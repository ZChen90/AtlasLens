from __future__ import annotations
import json, platform, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    import yaml
    cfg = yaml.safe_load(Path(path or ROOT / "config.yaml").read_text())
    cfg["_root"] = str(ROOT)
    return cfg


def data_dir(cfg) -> Path:
    p = Path(cfg["_root"]) / cfg["data_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir(cfg, step: str) -> Path:
    p = Path(cfg["_root"]) / cfg["results_dir"] / step
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_run_record(out: Path, step: str, params: dict, summary: dict) -> Path:
    """Every pipeline run leaves a JSON record: parameters, package versions, key numbers."""
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    from importlib import metadata
    for pkg in ("scanpy", "anndata", "numpy", "pandas", "scikit-image", "tifffile",
                "cellpose", "langgraph", "langchain"):
        try:
            versions[pkg] = metadata.version(pkg)
        except Exception:
            pass
    rec = {"step": step, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "params": params, "versions": versions, "summary": summary}
    path = out / "run.json"
    path.write_text(json.dumps(rec, indent=2, default=str))
    return path
