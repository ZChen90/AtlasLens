"""Data access: which files are needed, and loaders that return tidy tables.

Downloading and file lookup live in fetch.py (standard library only). Loaders read the large
metadata tables in chunks and keep only the rows and columns the pipelines use.
"""
from __future__ import annotations
from pathlib import Path
from .common import data_dir


# ----------------------------------------------------------------------------
# What we need
# ----------------------------------------------------------------------------
def allen_files(cfg) -> list[tuple[str, str]]:
    """(directory, file_name) pairs in abc_atlas_access naming."""
    r, m = cfg["region"], cfg["merfish"]
    return [
        ("WMB-10X", "cell_metadata_with_cluster_annotation"),
        ("WMB-10X", "gene"),
        (r["tenx_dataset_dir"], f'{r["tenx_feature_matrix"]}/log2'),
        (m["dataset_dir"], "cell_metadata_with_cluster_annotation"),
        (m["dataset_dir"], "gene"),
        (m["dataset_dir"], m["matrix"]),
        (m["dataset_dir"] + "-CCF", "ccf_coordinates"),
        ("Allen-CCF-2020", "parcellation_to_parcellation_term_membership_acronym"),
    ]


def tenx_files(cfg) -> dict[str, str]:
    s, base = cfg["imaging"]["sample"], cfg["imaging"]["base_url"]
    return {f"{s}_{suffix}": f"{base}/{s}/{s}_{suffix}"
            for suffix in ("image.tif", "spatial.tar.gz", "filtered_feature_bc_matrix.h5")}


def get_cache(cfg, refresh: bool = False):
    from .fetch import MirrorCache
    return MirrorCache(data_dir(cfg) / "abc_atlas", refresh=refresh)


def plan(cfg, cache=None) -> int:
    """Print every file with its size and whether it is already on disk. Downloads nothing
    except the small release manifest."""
    from . import fetch
    return fetch.plan(cache or get_cache(cfg), allen_files(cfg), tenx_files(cfg), data_dir(cfg) / "tenx_if")


def download(cfg, assume_yes: bool = False):
    from . import fetch
    fetch.fetch_all(get_cache(cfg), allen_files(cfg), tenx_files(cfg), data_dir(cfg) / "tenx_if", assume_yes)


# ----------------------------------------------------------------------------
# Loaders. `cache` only needs get_file_path() and get_metadata_dataframe(),
# which lets tests substitute a fake cache backed by synthetic files.
# ----------------------------------------------------------------------------
TENX_COLUMNS = {"cell_label", "library_label", "feature_matrix_label", "region_of_interest_acronym",
                "donor_label", "donor_sex", "neurotransmitter", "class", "subclass", "supertype",
                "cluster", "class_color", "subclass_color"}
MERFISH_COLUMNS = {"cell_label", "brain_section_label", "x", "y", "z", "neurotransmitter", "class",
                   "subclass", "supertype", "cluster", "class_color", "subclass_color"}


def load_tenx_cells(cfg, cache):
    """10x cell metadata with Allen taxonomy labels, restricted to the region matrix."""
    import pandas as pd
    want = cfg["region"]["tenx_feature_matrix"]
    chunks = cache.get_metadata_dataframe(
        directory="WMB-10X", file_name="cell_metadata_with_cluster_annotation",
        dtype={"cell_label": str, "neurotransmitter": str},
        usecols=lambda c: c in TENX_COLUMNS, chunksize=500_000)   # 4 million rows in the full table
    cell = pd.concat(c[c["feature_matrix_label"] == want] for c in chunks)
    return cell.set_index("cell_label")


def tenx_matrix_path(cfg, cache) -> Path:
    r = cfg["region"]
    return Path(cache.get_file_path(directory=r["tenx_dataset_dir"],
                                    file_name=f'{r["tenx_feature_matrix"]}/log2'))


def load_merfish_cells(cfg, cache):
    """Returns (region, background, divisions).

    region     : MERFISH cells inside the configured CCF division, with taxonomy labels, section
                 coordinates (x, y) and parcellation_division / _structure / _substructure
    background : section label and x, y for every cell (for drawing whole sections in grey)
    divisions  : all division names, for error messages
    """
    import pandas as pd
    m = cfg["merfish"]
    terms = cache.get_metadata_dataframe(
        directory="Allen-CCF-2020",
        file_name="parcellation_to_parcellation_term_membership_acronym").set_index("parcellation_index")
    terms.columns = [f"parcellation_{c}" for c in terms.columns]
    wanted = terms.index[terms["parcellation_division"] == cfg["region"]["parcellation_division"]]
    ccf = cache.get_metadata_dataframe(
        directory=m["dataset_dir"] + "-CCF", file_name="ccf_coordinates",
        dtype={"cell_label": str}, usecols=["cell_label", "parcellation_index"])
    ccf = ccf[ccf["parcellation_index"].isin(wanted)].set_index("cell_label").join(terms, on="parcellation_index")

    labels = set(ccf.index)
    region, background = [], []
    for c in cache.get_metadata_dataframe(
            directory=m["dataset_dir"], file_name="cell_metadata_with_cluster_annotation",
            dtype={"cell_label": str, "neurotransmitter": str},
            usecols=lambda c: c in MERFISH_COLUMNS, chunksize=500_000):
        background.append(c[["brain_section_label", "x", "y"]].astype({"x": "float32", "y": "float32"}))
        region.append(c[c["cell_label"].isin(labels)])
    region = pd.concat(region).set_index("cell_label").join(ccf, how="inner")
    background = pd.concat(background, ignore_index=True)
    background["brain_section_label"] = background["brain_section_label"].astype("category")
    return region, background, sorted(terms["parcellation_division"].dropna().unique())


def merfish_matrix_path(cfg, cache) -> Path:
    m = cfg["merfish"]
    return Path(cache.get_file_path(directory=m["dataset_dir"], file_name=m["matrix"]))


def tenx_if_image_path(cfg) -> Path:
    return data_dir(cfg) / "tenx_if" / f'{cfg["imaging"]["sample"]}_image.tif'
