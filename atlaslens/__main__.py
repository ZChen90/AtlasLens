"""Command line: python -m atlaslens {plan,download,scrna,merfish,imaging,validate,all}

validate checks the cell identity model against MERFISH ground truth, e.g.
  python -m atlaslens validate --structure DG --genes Slc17a7 Gad2 Aqp4 Mog
"""
import argparse
from .common import load_config


def main():
    ap = argparse.ArgumentParser(prog="atlaslens", description=__doc__)
    ap.add_argument("step", choices=["plan", "download", "scrna", "merfish", "imaging", "validate", "all"])
    ap.add_argument("--config", default=None, help="path to a config.yaml (default: project root)")
    ap.add_argument("--yes", action="store_true", help="download without the confirmation prompt")
    ap.add_argument("--image", default=None, help="imaging: analyse this TIFF instead of the 10x section")
    ap.add_argument("--roi", type=int, nargs=4, metavar=("X0", "Y0", "W", "H"),
                    help="imaging: region in full-resolution pixels")
    ap.add_argument("--structure", default="DG", help="validate: Allen CCF structure")
    ap.add_argument("--genes", nargs="+", default=["Slc17a7", "Gad2", "Aqp4", "Mog"],
                    help="validate: marker genes (must be in the MERFISH panel)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.roi:
        cfg["imaging"]["roi"] = a.roi

    from . import data
    if a.step == "plan":
        data.plan(cfg)
    elif a.step == "download":
        data.download(cfg, assume_yes=a.yes)
    if a.step in ("scrna", "all"):
        from . import scrna
        scrna.run(cfg)
    if a.step in ("merfish", "all"):
        from . import merfish
        merfish.run(cfg)
    if a.step == "validate":
        import json
        from . import reference
        r = reference.validate_on_merfish(cfg, a.structure, a.genes)
        print(json.dumps({k: v for k, v in r.items() if k != "phenotypes"}, indent=1))
        for code, ph in r["phenotypes"].items():
            print(f"{code:40s} n={ph['n_nuclei']:6d}  -> {ph['best_class']} ({ph['best_class_posterior']}), "
                  f"resolved={ph['resolved']}, truth={ph['true_class_composition']}")
    if a.step in ("imaging", "all"):
        from . import imaging
        imaging.run(cfg, a.image)


if __name__ == "__main__":
    main()
