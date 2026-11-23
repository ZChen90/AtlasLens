"""scRNA-seq pipeline for one Allen 10x region matrix.

Input matrix is the Allen `log2` file: log2(CPM + 1), genes indexed by Ensembl ID.
The Allen taxonomy labels (class / subclass / supertype / cluster) come with the
metadata, so unsupervised clustering here doubles as a validation: Leiden
clusters are scored against the reference labels.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .common import results_dir, write_run_record
from . import data


def subsample(cells: pd.DataFrame, n: int | None, min_per_group: int, seed: int,
              group: str = "subclass") -> pd.Index:
    """Random subsample that keeps rare groups (up to `min_per_group` cells each)."""
    if n is None or n >= len(cells):
        return cells.index
    rng = np.random.default_rng(seed)
    frac = n / len(cells)
    keep = []
    for _, idx in cells.groupby(group, observed=True).groups.items():
        k = min(len(idx), max(min_per_group, int(round(frac * len(idx)))))
        keep.append(rng.choice(np.asarray(idx), size=k, replace=False))
    return pd.Index(np.concatenate(keep))


def neighbors(adata, k: int, seed: int, force_fallback: bool = False):
    """kNN graph on X_pca. Uses scanpy; if scanpy's neighbour search fails (some scanpy /
    pynndescent / scipy version combinations do), builds the same UMAP-style graph directly."""
    import scanpy as sc
    if not force_fallback:
        try:
            sc.pp.neighbors(adata, n_neighbors=k, random_state=seed)
            return "scanpy"
        except (AttributeError, TypeError) as e:
            print(f"scanpy neighbours failed ({e}); using exact kNN fallback")
    from scipy.sparse import csr_matrix
    from sklearn.neighbors import NearestNeighbors
    from umap.umap_ import fuzzy_simplicial_set
    X = adata.obsm["X_pca"]
    dist, idx = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
    conn = fuzzy_simplicial_set(X, k, np.random.RandomState(seed), "euclidean",
                                knn_indices=idx, knn_dists=dist)[0]
    n = X.shape[0]
    rows = np.repeat(np.arange(n), k - 1)
    adata.obsp["distances"] = csr_matrix((dist[:, 1:].ravel(), (rows, idx[:, 1:].ravel())), shape=(n, n))
    adata.obsp["connectivities"] = csr_matrix(conn)
    adata.uns["neighbors"] = {"connectivities_key": "connectivities", "distances_key": "distances",
                              "params": {"n_neighbors": k, "method": "umap", "metric": "euclidean",
                                         "random_state": seed, "use_rep": "X_pca"}}
    return "exact-knn fallback"


def load_region_adata(cfg, cache, labels: pd.Index, cells: pd.DataFrame):
    import anndata as ad
    backed = ad.read_h5ad(data.tenx_matrix_path(cfg, cache), backed="r")
    pos = backed.obs_names.get_indexer(labels)
    pos = np.sort(pos[pos >= 0])
    adata = backed[pos].to_memory()
    backed.file.close()
    adata.var["gene_identifier"] = adata.var_names
    adata.var_names = adata.var["gene_symbol"].astype(str)
    adata.var_names_make_unique()
    adata.var.index.name = None      # the gene_symbol column keeps the original names
    cols = [c for c in ("class", "subclass", "supertype", "cluster", "neurotransmitter",
                        "region_of_interest_acronym", "donor_label", "donor_sex",
                        "library_label") if c in cells.columns]
    adata.obs = adata.obs[[]].join(cells[cols])
    for c in cols:
        adata.obs[c] = adata.obs[c].astype("category")
    return adata


def run(cfg, cache=None) -> dict:
    import scanpy as sc
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

    p = cfg["scrna"]
    out = results_dir(cfg, "scrna")
    cache = cache or data.get_cache(cfg)

    cells = data.load_tenx_cells(cfg, cache)
    cells = cells[cells["subclass"].notna()]
    labels = subsample(cells, p["n_cells"], p["min_cells_per_subclass"], p["seed"])
    adata = load_region_adata(cfg, cache, labels, cells)

    # Embedding and clustering. X is already library-size normalised and logged.
    sc.pp.highly_variable_genes(adata, n_top_genes=min(p["n_hvg"], adata.n_vars), flavor="seurat")
    n_pcs = min(p["n_pcs"], int(adata.var["highly_variable"].sum()) - 1, adata.n_obs - 1)
    hv = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.pca(hv, n_comps=n_pcs, random_state=p["seed"])
    adata.obsm["X_pca"] = hv.obsm["X_pca"]
    del hv
    neighbors(adata, 15, p["seed"])
    sc.tl.umap(adata, random_state=p["seed"])
    sc.tl.leiden(adata, resolution=p["leiden_resolution"], random_state=p["seed"],
                 flavor="igraph", n_iterations=2, directed=False)

    # Validation against the Allen taxonomy
    agreement = {
        f"{metric}_leiden_vs_{level}": float(fn(adata.obs[level], adata.obs["leiden"]))
        for level in ("class", "subclass")
        for metric, fn in (("ARI", adjusted_rand_score), ("AMI", adjusted_mutual_info_score))
    }

    # Marker genes per subclass (subclasses with too few cells are skipped)
    counts = adata.obs["subclass"].value_counts()
    groups = counts[counts >= 20].index.tolist()
    sc.tl.rank_genes_groups(adata, "subclass", groups=groups, method="t-test_overestim_var")
    markers = sc.get.rank_genes_groups_df(adata, group=None)
    markers = markers.groupby("group", observed=True).head(25)
    markers.to_csv(out / "markers_by_subclass.csv", index=False)

    comp = (adata.obs.groupby(["class", "subclass"], observed=True).size()
            .rename("n_cells_sampled").reset_index())
    full = cells["subclass"].value_counts()
    comp["n_cells_region"] = comp["subclass"].map(full).astype(int)
    comp["fraction_region"] = comp["n_cells_region"] / len(cells)
    comp.sort_values("n_cells_region", ascending=False).to_csv(out / "composition.csv", index=False)

    # Figures
    for key in ("class", "subclass", "leiden"):
        legend = "right margin" if adata.obs[key].nunique() <= 30 else None
        sc.pl.umap(adata, color=key, legend_loc=legend, show=False, frameon=False,
                   title=f"{cfg['region']['name']} scRNA-seq: {key}")
        plt.savefig(out / f"umap_{key}.png", dpi=150, bbox_inches="tight")
        plt.close("all")
    panel = {k: [g for g in v if g in adata.var_names] for k, v in p["markers"].items()}
    panel = {k: v for k, v in panel.items() if v}
    if panel:
        sc.pl.dotplot(adata, panel, groupby="class", show=False)
        plt.savefig(out / "dotplot_markers_by_class.png", dpi=150, bbox_inches="tight")
        plt.close("all")

    adata.write_h5ad(out / "region_subsample.h5ad")
    summary = {
        "n_cells_region": int(len(cells)), "n_cells_analysed": int(adata.n_obs),
        "n_genes": int(adata.n_vars), "n_classes": int(adata.obs["class"].nunique()),
        "n_subclasses": int(adata.obs["subclass"].nunique()),
        "n_leiden_clusters": int(adata.obs["leiden"].nunique()),
        **agreement,
        "top_subclasses": comp.sort_values("n_cells_region", ascending=False)
                              .head(8).set_index("subclass")["fraction_region"].round(4).to_dict(),
    }
    write_run_record(out, "scrna", p, summary)
    print(json.dumps(summary, indent=2))
    return summary
