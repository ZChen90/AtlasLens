"""Fluorescence-image pipeline: nuclei segmentation and per-nucleus marker quantification.

Written for multi-channel sections (DAPI + antibody channels). It runs on the
10x Visium IF section used as a stand-in here and on confocal data: for a
z-stack, pass a single plane or a maximum projection as a (C, Y, X) array.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import results_dir, write_run_record
from . import data


# ----------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------
class LazyImage:
    """Multi-channel 2D image read one channel at a time, so a multi-GB slide scan
    never has to sit in memory whole. Shape is reported as (C, Y, X)."""

    def __init__(self, path: Path):
        import tifffile
        self.path = Path(path)
        self._mm = None
        try:                                   # fast path: uncompressed, contiguous file
            mm = tifffile.memmap(self.path, mode="r")
            if mm.ndim == 2:
                mm = mm[None]
            if mm.ndim == 3 and mm.shape[-1] <= 4 < mm.shape[0]:
                mm = np.moveaxis(mm, -1, 0)
            if mm.ndim == 3:
                self._mm = mm
        except Exception:
            pass
        if self._mm is not None:
            self.shape, self.mode = tuple(self._mm.shape), "memmap"
            return
        self._tif = tifffile.TiffFile(self.path)
        series = self._tif.series[0]
        shp = tuple(series.shape)
        self._pages = None
        if len(shp) == 2:
            # one 2D page per channel, possibly stored as separate series; pyramid levels
            # (different shapes) are ignored
            self._pages = [pg for pg in self._tif.pages if tuple(pg.shape) == shp]
            self.shape, self.mode = (len(self._pages),) + shp, "pages"
        elif len(shp) == 3 and len(series.pages) == shp[0]:
            self._pages = list(series.pages)
            self.shape, self.mode = shp, "pages"          # one page per channel
        elif len(shp) == 3 and shp[-1] <= 4:
            self.shape, self.mode = (shp[2], shp[0], shp[1]), "interleaved"
            self._full = None
        else:
            raise ValueError(f"Unsupported image layout: axes {series.axes}, shape {shp}. "
                             "For z-stacks export one plane or a projection as (C, Y, X).")

    def channel(self, c: int) -> np.ndarray:
        """Full-resolution 2D array for one channel."""
        if self.mode == "memmap":
            return self._mm[c]
        if self.mode == "pages":
            return self._pages[c].asarray()
        if self._full is None:
            self._full = self._tif.series[0].asarray()
        return self._full[..., c]

    def overview(self, step: int) -> np.ndarray:
        return np.stack([np.asarray(self.channel(c)[::step, ::step]) for c in range(self.shape[0])])

    def crop(self, x0: int, y0: int, w: int, h: int) -> np.ndarray:
        return np.stack([np.asarray(self.channel(c)[y0:y0 + h, x0:x0 + w]) for c in range(self.shape[0])])


def open_image(path: Path) -> LazyImage:
    return LazyImage(path)


def _norm(a, lo=1, hi=99.8):
    a = a.astype(np.float32)
    p0, p1 = np.percentile(a, [lo, hi])
    return np.clip((a - p0) / max(p1 - p0, 1e-6), 0, 1)


def composite(img: np.ndarray, channels: dict) -> np.ndarray:
    """RGB preview: DAPI blue, first marker green, second marker red."""
    rgb = np.zeros(img.shape[1:] + (3,), np.float32)
    names = list(channels)
    rgb[..., 2] = _norm(img[channels["DAPI"]])
    markers = [n for n in names if n != "DAPI"]
    if markers:
        rgb[..., 1] = _norm(img[channels[markers[0]]])
    if len(markers) > 1:
        rgb[..., 0] = _norm(img[channels[markers[1]]])
    return rgb


def auto_roi(dapi: np.ndarray, size: int) -> tuple[int, int, int, int]:
    """Window with the highest DAPI signal: a cell-dense area. Verify on the overview."""
    from skimage.transform import integral_image
    step = max(1, min(dapi.shape) // 1500)
    small = np.asarray(dapi[::step, ::step], np.float32)
    small = np.clip(small - np.percentile(small, 50), 0, np.percentile(small, 99.9))
    w = max(1, min(size // step, small.shape[0], small.shape[1]))
    ii = np.pad(integral_image(small), ((1, 0), (1, 0)))
    sums = ii[w:, w:] - ii[:-w, w:] - ii[w:, :-w] + ii[:-w, :-w]
    y, x = np.unravel_index(np.argmax(sums), sums.shape)
    h = min(size, dapi.shape[0]); wd = min(size, dapi.shape[1])
    return int(min(x * step, dapi.shape[1] - wd)), int(min(y * step, dapi.shape[0] - h)), wd, h


# ----------------------------------------------------------------------------
# Segmentation
# ----------------------------------------------------------------------------
def segment_classical(dapi: np.ndarray, diameter: float | None = None) -> np.ndarray:
    from scipy import ndimage as ndi
    from skimage import filters, measure, morphology, segmentation, feature
    img = _norm(dapi, 1, 99.9)
    sm = filters.gaussian(img, 1)
    bg = filters.gaussian(img, 25)
    fg = np.clip(sm - bg, 0, None)
    mask = fg > filters.threshold_otsu(fg)
    mask = ndi.binary_fill_holes(ndi.binary_opening(mask, morphology.disk(1)))
    if diameter is None:   # estimate from connected components
        areas = [r.area for r in measure.regionprops(measure.label(mask)) if r.area >= 12]
        diameter = 2 * np.sqrt(np.median(areas) / np.pi) if areas else 10.0
    lab = measure.label(mask)
    small = np.bincount(lab.ravel()) < int(0.2 * np.pi * (diameter / 2) ** 2)
    small[0] = False
    mask[small[lab]] = False
    dist = ndi.gaussian_filter(ndi.distance_transform_edt(mask), 1)
    peaks = feature.peak_local_max(dist, min_distance=max(2, int(diameter * 0.35)),
                                   labels=measure.label(mask), exclude_border=False)
    markers = np.zeros(mask.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return segmentation.watershed(-dist, markers, mask=mask)


def segment_cellpose(dapi: np.ndarray, diameter: float | None = None) -> np.ndarray:
    from cellpose import models
    try:
        import torch
        gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    except Exception:
        gpu = False
    model = models.CellposeModel(gpu=gpu)
    return model.eval(np.asarray(dapi), diameter=diameter)[0].astype(np.int32)


def segment(dapi, method="auto", diameter=None) -> tuple[np.ndarray, str]:
    if method in ("auto", "cellpose"):
        try:
            return segment_cellpose(dapi, diameter), "cellpose"
        except Exception as e:
            if method == "cellpose":
                raise
            print(f"cellpose unavailable ({type(e).__name__}); using classical segmentation")
    return segment_classical(dapi, diameter), "classical"


# ----------------------------------------------------------------------------
# Measurement and QC
# ----------------------------------------------------------------------------
def measure_nuclei(labels: np.ndarray, img: np.ndarray, channels: dict, ring_px: int = 3) -> pd.DataFrame:
    from skimage import measure, segmentation
    props = measure.regionprops_table(labels, properties=(
        "label", "area", "centroid", "eccentricity", "solidity", "bbox"))
    df = pd.DataFrame(props).rename(columns={"centroid-0": "y", "centroid-1": "x"})
    h, w = labels.shape
    df["touches_border"] = ((df["bbox-0"] == 0) | (df["bbox-1"] == 0) |
                            (df["bbox-2"] == h) | (df["bbox-3"] == w))
    df = df.drop(columns=[c for c in df if c.startswith("bbox")])
    ring = segmentation.expand_labels(labels, ring_px)
    ring[labels > 0] = 0
    for name, c in channels.items():
        ch = np.asarray(img[c], np.float32)
        bg = float(np.percentile(ch[labels == 0], 20)) if (labels == 0).any() else 0.0
        nuc = measure.regionprops_table(labels, ch, properties=("label", "mean_intensity"))
        df[f"{name}_nucleus"] = pd.Series(nuc["mean_intensity"], index=nuc["label"]).reindex(df["label"]).to_numpy() - bg
        rng = measure.regionprops_table(ring, ch, properties=("label", "mean_intensity"))
        df[f"{name}_ring"] = pd.Series(rng["mean_intensity"], index=rng["label"]).reindex(df["label"]).to_numpy() - bg
    return df


def classify_positive(values: pd.Series) -> tuple[pd.Series, float]:
    """Otsu threshold on log intensity. Returns (is_positive, threshold)."""
    from skimage.filters import threshold_otsu
    v = np.log1p(np.clip(values.fillna(0).to_numpy(), 0, None))
    if np.ptp(v) == 0:
        return pd.Series(False, index=values.index), float("nan")
    t = threshold_otsu(v)
    return pd.Series(v > t, index=values.index), float(np.expm1(t))


def qc_flags(df: pd.DataFrame, roi_area_px: int) -> list[str]:
    flags = []
    if len(df) < 50:
        flags.append("fewer than 50 nuclei: check ROI, channel order and threshold")
    else:
        cv = df["area"].std() / df["area"].mean()
        if cv > 1.0:
            flags.append(f"nucleus area CV {cv:.2f} > 1.0: likely merged or fragmented objects")
        if df["solidity"].median() < 0.85:
            flags.append("low median solidity: under-segmented clumps likely")
        if df["area"].sum() / roi_area_px > 0.6:
            flags.append("nuclei cover >60% of ROI: threshold too permissive")
    return flags


def analyse(img: np.ndarray, channels: dict, method="auto", diameter=None, ring_px=3):
    """Core entry point on an in-memory (C, Y, X) crop. Returns (labels, table, summary)."""
    labels, used = segment(np.asarray(img[channels["DAPI"]]), method, diameter)
    df = measure_nuclei(labels, img, channels, ring_px)
    summary = {"segmentation": used, "n_nuclei": int(len(df)),
               "median_area_px": float(df["area"].median()) if len(df) else 0.0}
    for name in channels:
        if name == "DAPI" or df.empty:
            continue
        # nuclear markers (NeuN) read from the nucleus, cytoplasmic ones (GFAP) from the ring
        col = f"{name}_nucleus" if name.lower() in ("neun", "sox10", "olig2", "prox1") else f"{name}_ring"
        df[f"{name}_positive"], thr = classify_positive(df[col])
        summary[f"{name}_measured_in"] = col.split("_")[-1]
        summary[f"{name}_threshold"] = round(thr, 2)
        summary[f"fraction_{name}_positive"] = round(float(df[f"{name}_positive"].mean()), 3)
    summary["qc_flags"] = qc_flags(df, labels.size)
    return labels, df, summary


def save_overlay(path: Path, rgb: np.ndarray, labels: np.ndarray, df: pd.DataFrame, marker: str | None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    out = rgb.copy()
    edge = find_boundaries(labels, mode="outer")
    out[edge] = (1, 1, 1)
    if marker and f"{marker}_positive" in df:
        pos = np.isin(labels, df.loc[df[f"{marker}_positive"], "label"])
        out[edge & pos] = (1, 1, 0)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(rgb); axes[0].set_title("composite")
    axes[1].imshow(out)
    axes[1].set_title("nuclei outlines" + (f" (yellow = {marker}+)" if marker else ""))
    for a in axes:
        a.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(cfg, image_path: Path | None = None) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = cfg["imaging"]
    out = results_dir(cfg, "imaging")
    path = Path(image_path or data.tenx_if_image_path(cfg))
    img = open_image(path)
    channels = {k: v for k, v in p["channels"].items() if v < img.shape[0]}

    # Low-resolution overview, read one channel at a time
    step = max(1, max(img.shape[1:]) // 2000)
    small = img.overview(step)

    roi = p["roi"]
    if roi in (None, "auto"):
        sx, sy, sw, sh = auto_roi(small[channels["DAPI"]], max(1, p["roi_size"] // step))
        w, h = min(p["roi_size"], img.shape[2]), min(p["roi_size"], img.shape[1])
        roi = (min(sx * step, img.shape[2] - w), min(sy * step, img.shape[1] - h), w, h)
    x0, y0, w, h = map(int, roi)

    # Overview with the ROI box, so a human or the agent can check where we are
    over = composite(small, channels)
    fig, ax = plt.subplots(figsize=(10, 10 * over.shape[0] / over.shape[1]))
    ax.imshow(over, extent=(0, img.shape[2], img.shape[1], 0))
    ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, ec="yellow", lw=1.5))
    ax.set_title(f"{path.name}  (full-resolution pixel coordinates; yellow = ROI)")
    fig.savefig(out / "overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    crop = img.crop(x0, y0, w, h)
    labels, df, summary = analyse(crop, channels, p["segmentation"],
                                  p["nucleus_diameter_px"], p["ring_px"])
    df["x_full"], df["y_full"] = df["x"] + x0, df["y"] + y0
    df.to_csv(out / "nuclei.csv", index=False)
    np.save(out / "labels.npy", labels.astype(np.int32))
    markers = [n for n in channels if n != "DAPI"]
    save_overlay(out / "overlay.png", composite(crop, channels), labels, df,
                 markers[0] if markers else None)

    summary = {"image": path.name, "image_shape_cyx": list(img.shape), "read_mode": img.mode, "roi_xywh": [x0, y0, w, h], **summary}
    write_run_record(out, "imaging", p, summary)
    print(json.dumps(summary, indent=2))
    return summary
