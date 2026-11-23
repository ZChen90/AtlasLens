"""Reference-based cell identity: which cell types can the marker-positive cells in an image be?

Evidence used
  prior       : cell-type composition of the anatomical structure, from MERFISH (intact tissue)
  likelihood  : probability that a cell type is positive for each marker, from scRNA-seq
                (fraction of cells of that subclass in which the marker's gene is detected)
  observation : per-nucleus marker calls from the image

posterior(type | phenotype, structure)  is proportional to
    prior(type | structure) * product over markers of P(marker status | type)

This is a transparent naive-Bayes model. It assumes markers are independent given the cell
type and that mRNA detection predicts antibody staining; both are approximations, and the
report states them.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .common import results_dir

EPS = 0.02   # no marker is treated as perfectly sensitive or specific
MIN_MARGINAL = 0.02   # below this, no cell type expected in the structure accounts for the phenotype


def _need(path: Path, step: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m atlaslens {step}` first.")
    return path


def marker_profile(cfg, genes: list[str]) -> pd.DataFrame:
    """Per subclass: class, n_cells and the fraction of cells with each gene detected."""
    import anndata as ad
    a = ad.read_h5ad(_need(results_dir(cfg, "scrna") / "region_subsample.h5ad", "scrna"), backed="r")
    missing = [g for g in genes if g not in a.var_names]
    if missing:
        a.file.close()
        raise KeyError(f"Genes not in the scRNA-seq matrix: {missing}")
    X = a[:, genes].to_memory().X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    obs = a.obs[["class", "subclass"]].astype(str)
    a.file.close()
    det = pd.DataFrame(X > 0, columns=genes, index=obs.index)
    g = det.groupby(obs["subclass"])
    out = g.mean()
    out.insert(0, "n_cells", g.size())
    out.insert(0, "class", obs.groupby("subclass")["class"].first())
    return out


def structure_composition(cfg, structure: str | None = None) -> tuple[pd.Series, int, list[str]]:
    """Subclass fractions among MERFISH cells in a CCF structure (None = whole division)."""
    import anndata as ad
    a = ad.read_h5ad(_need(results_dir(cfg, "merfish") / "region_merfish.h5ad", "merfish"), backed="r")
    obs = a.obs[["subclass", "parcellation_structure"]].astype(str)
    a.file.close()
    available = sorted(obs["parcellation_structure"].unique())
    if structure:
        obs = obs[obs["parcellation_structure"] == structure]
        if obs.empty:
            raise ValueError(f"No MERFISH cells in structure {structure!r}. Available: {available}")
    return obs["subclass"].value_counts(normalize=True), len(obs), available


def assign(nuclei: pd.DataFrame, marker_genes: dict[str, str], profile: pd.DataFrame,
           prior: pd.Series, min_prior: float = 0.002) -> tuple[pd.DataFrame, dict]:
    """Posterior over cell types for every marker phenotype seen in the image.

    marker_genes maps image marker name -> gene symbol, e.g. {"NeuN": "Rbfox3", "GFAP": "Gfap"}.
    Returns the nuclei table with assignments added, and a summary dictionary.
    """
    markers = [m for m in marker_genes if f"{m}_positive" in nuclei.columns]
    if not markers:
        raise ValueError("No marker positivity calls in the nuclei table.")
    genes = [marker_genes[m] for m in markers]

    types = [t for t in prior.index if t in profile.index and prior[t] >= min_prior]
    pi = prior[types] / prior[types].sum()
    P = profile.loc[types, genes].clip(EPS, 1 - EPS).to_numpy()
    cls = profile.loc[types, "class"]

    flags = nuclei[[f"{m}_positive" for m in markers]].astype(bool).to_numpy()
    codes = ["".join(f"{m}{'+' if f else '-'} " for m, f in zip(markers, row)).strip() for row in flags]
    out = nuclei.copy()
    out["phenotype"] = codes

    phenotypes = {}
    for code in sorted(set(codes)):
        row = flags[codes.index(code)]
        like = np.prod(np.where(row, P, 1 - P), axis=1)
        post = pi.to_numpy() * like
        marginal = float(post.sum())       # how well any cell type of this structure explains the phenotype
        explained = marginal >= MIN_MARGINAL
        post = post / post.sum()
        sub = pd.Series(post, index=types).sort_values(ascending=False)
        by_class = pd.Series(post, index=types).groupby(cls).sum().sort_values(ascending=False)
        n = codes.count(code)
        phenotypes[code] = {
            "n_nuclei": n, "fraction_of_nuclei": round(n / len(codes), 3),
            "class_posterior": by_class.head(4).round(3).to_dict(),
            "subclass_posterior": sub.head(5).round(3).to_dict(),
            "best_class": by_class.index[0], "best_class_posterior": round(float(by_class.iloc[0]), 3),
            "phenotype_probability_under_reference": round(marginal, 4),
            "explained_by_reference": bool(explained),
            "resolved": bool(explained and by_class.iloc[0] >= 0.8),
        }
    out["assigned_class"] = out["phenotype"].map(lambda c: phenotypes[c]["best_class"])
    out["assigned_class_posterior"] = out["phenotype"].map(lambda c: phenotypes[c]["best_class_posterior"])

    # Consistency: does the image show as many positive cells as the references predict?
    consistency = {}
    for m, g, col in zip(markers, genes, P.T):
        expected = float((pi.to_numpy() * col).sum())
        observed = float(nuclei[f"{m}_positive"].mean())
        ratio = observed / max(expected, 1e-6)
        consistency[m] = {"gene": g, "expected_fraction": round(expected, 3),
                          "observed_fraction": round(observed, 3),
                          "consistent": bool(0.5 <= ratio <= 2 or abs(observed - expected) < 0.1)}
    summary = {
        "markers": dict(zip(markers, genes)), "n_nuclei": int(len(nuclei)),
        "n_cell_types_considered": len(types),
        "prior_mass_covered": round(float(prior[types].sum()), 3),
        "phenotypes": phenotypes, "consistency": consistency,
        "positive_subclasses_per_marker": {
            m: profile.loc[types, g].sort_values(ascending=False).head(6).round(2).to_dict()
            for m, g in zip(markers, genes)},
    }
    return out, summary


def validate_on_merfish(cfg, structure: str, genes: list[str], test_section: str | None = None) -> dict:
    """Ground-truth check of the identity model using MERFISH itself.

    MERFISH cells have both measured marker expression and a known cell-type label. Cells of one
    section in `structure` are treated like nuclei in an image: each gene is called positive or
    negative (Otsu on log expression, as in the imaging step), the model assigns a class from those
    calls alone, and the assignment is compared with the true label. The prior comes from the other
    sections, the likelihood from scRNA-seq, so nothing about the test cells' labels is used.
    """
    import anndata as ad
    from .imaging import classify_positive
    a = ad.read_h5ad(_need(results_dir(cfg, "merfish") / "region_merfish.h5ad", "merfish"), backed="r")
    obs = a.obs[["brain_section_label", "class", "subclass", "parcellation_structure"]].astype(str)
    missing = [g for g in genes if g not in a.var_names]
    if missing:
        a.file.close()
        raise KeyError(f"Not in the MERFISH panel: {missing}")
    in_struct = (obs["parcellation_structure"] == structure).to_numpy()
    if not in_struct.any():
        a.file.close()
        raise ValueError(f"No MERFISH cells in structure {structure!r}.")
    if test_section is None:
        test_section = obs.loc[in_struct, "brain_section_label"].value_counts().idxmax()
    test = in_struct & (obs["brain_section_label"] == test_section).to_numpy()
    X = a[np.where(test)[0], genes].to_memory().X
    a.file.close()
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    cells = obs[test].copy()
    thresholds = {}
    for j, g in enumerate(genes):
        cells[f"{g}_positive"], thresholds[g] = classify_positive(pd.Series(np.expm1(X[:, j]), index=cells.index))
    prior = obs[in_struct & ~test]["subclass"].value_counts(normalize=True)
    profile = marker_profile(cfg, genes)
    table, summary = assign(cells, {g: g for g in genes}, profile, prior)

    correct = table["assigned_class"] == table["class"]
    resolved = table["phenotype"].map(lambda c: summary["phenotypes"][c]["resolved"])
    per_class = (pd.DataFrame({"true": table["class"], "ok": correct & resolved})
                 .groupby("true")["ok"].agg(["size", "mean"]).sort_values("size", ascending=False))
    for code, ph in summary["phenotypes"].items():
        true = table.loc[table["phenotype"] == code, "class"].value_counts(normalize=True)
        ph["true_class_composition"] = true.head(3).round(3).to_dict()
    result = {
        "structure": structure, "test_section": test_section, "genes": genes,
        "n_test_cells": int(len(table)), "n_prior_cells": int((in_struct & ~test).sum()),
        "accuracy_all_cells": round(float(correct.mean()), 3),
        "fraction_cells_in_resolved_phenotypes": round(float(resolved.mean()), 3),
        "accuracy_within_resolved": round(float(correct[resolved].mean()), 3) if resolved.any() else None,
        "baseline_always_most_common_class": round(float(table["class"].value_counts(normalize=True).iloc[0]), 3),
        "recall_by_true_class": {k: {"n": int(v["size"]), "correctly_resolved": round(float(v["mean"]), 3)}
                                 for k, v in per_class.head(8).iterrows()},
        "phenotypes": summary["phenotypes"], "consistency": summary["consistency"],
    }
    out = results_dir(cfg, "validate")
    (out / f"merfish_{structure}_{'_'.join(genes)}.json").write_text(__import__("json").dumps(result, indent=1))
    return result
