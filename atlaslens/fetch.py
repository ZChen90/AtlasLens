"""Download and locate Allen Brain Cell Atlas files. Standard library only, so it runs with any
Python 3.8+ before the analysis environment exists:

    python3 atlaslens/fetch.py            # list files and sizes, ask, download
    python3 atlaslens/fetch.py --plan     # list only
    python3 atlaslens/fetch.py --yes      # download without asking

Files are mirrored under data/abc_atlas/ with the same relative paths as the public bucket, and
the release manifest is stored next to them, so later runs work offline and stay pinned to
the release that was downloaded.
"""
from __future__ import annotations
import json, re, shutil, sys, time, urllib.request
from pathlib import Path

BUCKET = "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com"
ROOT = Path(__file__).resolve().parents[1]

# Defaults used when config.yaml cannot be read (PyYAML missing). Keep in step with config.yaml.
DEFAULT_ALLEN = [
    ("WMB-10X", "cell_metadata_with_cluster_annotation"),
    ("WMB-10X", "gene"),
    ("WMB-10Xv3", "WMB-10Xv3-HPF/log2"),
    ("MERFISH-C57BL6J-638850", "cell_metadata_with_cluster_annotation"),
    ("MERFISH-C57BL6J-638850", "gene"),
    ("MERFISH-C57BL6J-638850", "C57BL6J-638850/log2"),
    ("MERFISH-C57BL6J-638850-CCF", "ccf_coordinates"),
    ("Allen-CCF-2020", "parcellation_to_parcellation_term_membership_acronym"),
]
_S = "V1_Adult_Mouse_Brain_Coronal_Section_2"
DEFAULT_TENX = {f"{_S}_{x}": f"https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/{_S}/{_S}_{x}"
                for x in ("image.tif", "spatial.tar.gz", "filtered_feature_bc_matrix.h5")}


def _open(url, headers=None, method="GET", timeout=60):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}, method=method),
                                  timeout=timeout)


def http_size(url: str):
    try:
        with _open(url, method="HEAD", timeout=30) as r:
            return int(r.headers["Content-Length"])
    except Exception:
        return None


def download(url: str, dest: Path, size: int | None = None, label: str | None = None) -> Path:
    """Download with resume. A finished file of the expected size is left alone."""
    dest = Path(dest)
    label = label or dest.name
    size = size if size is not None else http_size(url)
    if dest.exists() and (size is None or dest.stat().st_size == size):
        print(f"  have      {label}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, 6):
        done = part.stat().st_size if part.exists() else 0
        try:
            headers = {"Range": f"bytes={done}-"} if done else {}
            with _open(url, headers) as r:
                if done and r.status != 206:          # server ignored the range: start over
                    done = 0
                with open(part, "ab" if done else "wb") as fh:
                    last = 0.0
                    while True:
                        chunk = r.read(1 << 22)
                        if not chunk:
                            break
                        fh.write(chunk)
                        done += len(chunk)
                        if size and time.time() - last > 1:
                            last = time.time()
                            print(f"\r  {100 * done / size:5.1f}%    {label}", end="", flush=True)
            if size is None or done == size:
                break
            raise IOError(f"incomplete: {done} of {size} bytes")
        except Exception as e:
            print(f"\n  attempt {attempt} failed for {label}: {e}", file=sys.stderr)
            if attempt == 5:
                raise
            time.sleep(5 * attempt)
    part.rename(dest)
    print(f"\r  done      {label}" + " " * 12)
    return dest


class MirrorCache:
    """Resolves (directory, file_name) to a local file using the atlas release manifest.

    Same call names as the Allen `abc_atlas_access` cache for the parts we use:
    get_file_path(directory=..., file_name=...) and get_metadata_dataframe(...).
    """

    def __init__(self, cache_dir, bucket: str = BUCKET, refresh: bool = False):
        self.cache_dir = Path(cache_dir)
        self.bucket = bucket.rstrip("/")
        local = sorted(self.cache_dir.glob("releases/*/manifest.json"))
        if local and not refresh:
            path = local[-1]
        else:
            path = self._fetch_latest_manifest()
        self.current_manifest = f"releases/{path.parent.name}/manifest.json"
        self._data = json.loads(path.read_text())

    def _fetch_latest_manifest(self) -> Path:
        with _open(f"{self.bucket}/?list-type=2&prefix=releases/&delimiter=/") as r:
            releases = sorted(re.findall(r"<Prefix>releases/(\d+)/</Prefix>", r.read().decode()))
        if not releases:
            raise RuntimeError("Could not list atlas releases.")
        rel = f"releases/{releases[-1]}/manifest.json"
        return download(f"{self.bucket}/{rel}", self.cache_dir / rel, label=rel)

    def entry(self, directory: str, file_name: str) -> dict:
        """Manifest record {'url', 'size', 'relative_path', ...} for one file."""
        try:
            listing = self._data["file_listing"][directory]
        except KeyError:
            raise KeyError(f"Directory {directory!r} not in manifest. "
                           f"Available: {sorted(self._data['file_listing'])}") from None
        name, _, kind = file_name.partition("/")
        for section in listing.values():
            if name in section:
                node = section[name][kind] if kind else section[name]
                leaves = list(_leaves(node))
                if len(leaves) != 1:
                    raise KeyError(f"{file_name!r} in {directory!r} is ambiguous; add '/log2' or '/raw'.")
                return leaves[0]
        raise KeyError(f"{file_name!r} not found in {directory!r}.")

    def size(self, directory: str, file_name: str) -> int:
        return int(self.entry(directory, file_name)["size"])

    def local_path(self, directory: str, file_name: str) -> Path:
        return self.cache_dir / self.entry(directory, file_name)["relative_path"]

    def get_file_path(self, directory: str, file_name: str, download_missing: bool = True, **_) -> Path:
        e = self.entry(directory, file_name)
        path = self.cache_dir / e["relative_path"]
        if path.exists() and path.stat().st_size == int(e["size"]):
            return path
        if not download_missing:
            raise FileNotFoundError(f"{path} is missing or incomplete. Run `python3 atlaslens/fetch.py`.")
        return download(e["url"], path, int(e["size"]), label=f"{directory} :: {file_name}")

    def get_metadata_dataframe(self, directory: str, file_name: str, **kwargs):
        import pandas as pd
        return pd.read_csv(self.get_file_path(directory, file_name), **kwargs)


def _leaves(node):
    if isinstance(node, dict):
        if "url" in node and "size" in node:
            yield node
        else:
            for v in node.values():
                yield from _leaves(v)


def _gb(n):
    return "  unknown" if n is None else f"{n / 1e9:7.2f} GB"


def plan(cache: MirrorCache, allen: list, tenx: dict, tenx_dir: Path) -> int:
    print(f"Allen Brain Cell Atlas, {cache.current_manifest}")
    total = todo = 0
    for d, f in allen:
        n = cache.size(d, f)
        have = cache.local_path(d, f).exists() and cache.local_path(d, f).stat().st_size == n
        total += n
        todo += 0 if have else n
        print(f"  {_gb(n)}  {'have' if have else '    '}  {d} :: {f}")
    print("10x Genomics Visium immunofluorescence section")
    for name, url in tenx.items():
        n = http_size(url)
        have = n is not None and (tenx_dir / name).exists() and (tenx_dir / name).stat().st_size == n
        total += n or 0
        todo += 0 if have else (n or 0)
        print(f"  {_gb(n)}  {'have' if have else '    '}  {name}")
    cache.cache_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache.cache_dir).free
    print(f"Total {total / 1e9:.2f} GB, still to download {todo / 1e9:.2f} GB, free space {free / 1e9:.0f} GB")
    print("Licences: Allen 10x data CC BY-NC 4.0; Allen MERFISH and 10x Genomics data CC BY 4.0.")
    return todo


def fetch_all(cache: MirrorCache, allen: list, tenx: dict, tenx_dir: Path, assume_yes=False, only_plan=False):
    todo = plan(cache, allen, tenx, tenx_dir)
    if only_plan:
        return
    if todo and not assume_yes and input("Download the missing files? [y/N] ").strip().lower() != "y":
        print("Nothing downloaded.")
        return
    for d, f in allen:
        cache.get_file_path(d, f)
    for name, url in tenx.items():
        try:
            download(url, tenx_dir / name)
        except Exception as e:      # the Allen data alone is still useful
            print(f"  FAILED {name}: {e}", file=sys.stderr)
    print("All done.")


def main():
    args = set(sys.argv[1:])
    allen, tenx, data_root = DEFAULT_ALLEN, DEFAULT_TENX, ROOT / "data"
    try:                                   # use config.yaml when the environment allows it
        sys.path.insert(0, str(ROOT))
        from atlaslens.common import load_config, data_dir
        from atlaslens import data
        cfg = load_config()
        allen, tenx, data_root = data.allen_files(cfg), data.tenx_files(cfg), data_dir(cfg)
    except ImportError:
        print("(config.yaml not read: PyYAML missing; using the built-in hippocampus defaults)")
    cache = MirrorCache(data_root / "abc_atlas", refresh="--refresh" in args)
    fetch_all(cache, allen, tenx, data_root / "tenx_if", "--yes" in args, "--plan" in args)


if __name__ == "__main__":
    main()
