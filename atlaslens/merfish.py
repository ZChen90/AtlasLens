"""MERFISH pipeline: spatial layout of taxonomy cell types in one CCF division,
and a cross-modal comparison with the 10x scRNA-seq composition of the same region.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .common import results_dir, write_run_record
from . import data

# Subclass name fragment -> CCF structure where most of those cells should sit.
# Used as an automatic sanity check of labels + registration (hippocampus only).
EXPECTED_HOME = {"DG Glut": "DG", "CA1-ProS Glut": "CA1", "CA3 Glut": "CA3"}


def run(cfg, cache=None) -> dict:
    import anndata as ad
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    p = cfg["merfish"]
    division = cfg["region"]["parcellation_division"]
    out = results_dir(cfg, "merfish")
    cache = cache or data.get_cache(cfg)

    region, background, divisions = data.load_merfish_cells(cfg, cache)
    if region.empty:
        raise ValueError(f"No MERFISH cells in division {division!r}. Available: {divisions}")

    # Composition by anatomical structure
    comp = pd.crosstab(region["parcellation_structure"], region["subclass"])
    comp.to_csv(out / "subclass_by_structure_counts.csv")

    # Sanity check: do signature subclasses sit in their home structure?
    home = {}
    for frag, structure in EXPECTED_HOME.items():
        sel = region[region["subclass"].astype(str).str.contains(frag, regex=False)]
        if len(sel) >= 20:
            home[f"{frag} in {structure}"] = round(
                float((sel["parcellation_structure"] == structure).mean()), 3)

    # Cross-modal comparison: subclass fractions, MERFISH vs dissociated 10x
    tenx = data.load_tenx_cells(cfg, cache)
    both = pd.DataFrame({
        "merfish": region["subclass"].value_counts(normalize=True),
        "scrna": tenx["subclass"].value_counts(normalize=True)}).fillna(0)
    both["log2_ratio_scrna_over_merfish"] = np.log2((both["scrna"] + 1e-5) / (both["merfish"] + 1e-5))
    both.sort_values("merfish", ascending=False).to_csv(out / "subclass_fraction_merfish_vs_scrna.csv")
    shared = both[(both["merfish"] > 0) & (both["scrna"] > 0)]
    rho = float(spearmanr(shared["merfish"], shared["scrna"])[0]) if len(shared) > 2 else float("nan")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(both["merfish"] + 1e-5, both["scrna"] + 1e-5, s=14)
    for name, row in both.reindex(both["log2_ratio_scrna_over_merfish"].abs()
                                  .sort_values(ascending=False).index[:8]).iterrows():
        ax.annotate(str(name), (row["merfish"] + 1e-5, row["scrna"] + 1e-5), fontsize=6)
    lim = [1e-5, 1]
    ax.plot(lim, lim, lw=0.8, color="grey")
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
           xlabel=f"fraction of MERFISH cells in {division}",
           ylabel="fraction of 10x scRNA-seq cells",
           title=f"Subclass composition (Spearman {rho:.2f})")
    fig.savefig(out / "composition_merfish_vs_scrna.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Spatial maps for the section with the most cells of this region
    section = region["brain_section_label"].value_counts().idxmax()
    sec_all = background[background["brain_section_label"] == section]
    sec = region[region["brain_section_label"] == section]

    def base(ax):
        ax.scatter(sec_all["x"], sec_all["y"], s=0.3, color="#dddddd", rasterized=True)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")

    for key in ("subclass", "parcellation_structure"):
        fig, ax = plt.subplots(figsize=(9, 7))
        base(ax)
        color_col = f"{key}_color"
        if color_col in sec.columns:
            ax.scatter(sec["x"], sec["y"], s=1.5, c=sec[color_col], rasterized=True)
        else:
            codes = sec[key].astype("category")
            ax.scatter(sec["x"], sec["y"], s=1.5, c=codes.cat.codes, cmap="tab20", rasterized=True)
        ax.set_title(f"{section}: {division} cells by {key}")
        fig.savefig(out / f"section_{key}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

    # Expression: keep region cells, save as h5ad, plot marker genes on the section
    backed = ad.read_h5ad(data.merfish_matrix_path(cfg, cache), backed="r")
    pos = backed.obs_names.get_indexer(region.index)
    pos = np.sort(pos[pos >= 0])
    adata = backed[pos].to_memory()
    backed.file.close()
    if "gene_symbol" in adata.var:
        adata.var["gene_identifier"] = adata.var_names
        adata.var_names = adata.var["gene_symbol"].astype(str)
        adata.var_names_make_unique()
        adata.var.index.name = None      # the gene_symbol column keeps the original names
    keep = ["brain_section_label", "x", "y", "class", "subclass", "supertype", "cluster",
            "parcellation_structure", "parcellation_substructure"]
    adata.obs = adata.obs[[]].join(region[[c for c in keep if c in region.columns]])
    adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy()
    adata.write_h5ad(out / "region_merfish.h5ad")

    genes = [g for g in p["genes_to_plot"] if g in adata.var_names]
    in_sec = (adata.obs["brain_section_label"] == section).to_numpy()
    if genes and in_sec.any():
        n = len(genes)
        ncol = min(4, n)
        nrow = -(-n // ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow), squeeze=False)
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        sub = adata[in_sec]
        for ax, g in zip(axes.ravel(), genes):
            v = np.asarray(sub[:, g].X.todense()).ravel() if hasattr(sub.X, "todense") \
                else np.asarray(sub[:, g].X).ravel()
            order = np.argsort(v)
            ax.scatter(sub.obs["x"].to_numpy()[order], sub.obs["y"].to_numpy()[order], s=1.5,
                       c=v[order], cmap="magma", rasterized=True)
            ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off"); ax.set_title(g)
        fig.savefig(out / "section_marker_genes.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

    summary = {
        "division": division, "n_cells_division": int(len(region)),
        "n_sections_with_cells": int(region["brain_section_label"].nunique()),
        "example_section": str(section), "n_cells_example_section": int(len(sec)),
        "n_subclasses": int(region["subclass"].nunique()),
        "genes_in_panel": int(adata.n_vars), "marker_genes_plotted": genes,
        "home_structure_check": home,
        "spearman_subclass_fraction_merfish_vs_scrna": round(rho, 3),
        "most_depleted_in_scrna": both["log2_ratio_scrna_over_merfish"].nsmallest(5).round(2).to_dict(),
        "most_enriched_in_scrna": both["log2_ratio_scrna_over_merfish"].nlargest(5).round(2).to_dict(),
    }
    write_run_record(out, "merfish", p, summary)
    print(json.dumps(summary, indent=2))
    return summary
