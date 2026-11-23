"""Runs all three pipelines on small synthetic data that mimics the real file layout.
No downloads needed. Usage:  python tests/smoke_test.py
Use it as a regression test whenever a pipeline, a skill file or the agent changes.
"""
import sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlaslens.common import load_config          # noqa: E402
from atlaslens import scrna, merfish, imaging     # noqa: E402

rng = np.random.default_rng(0)
SUBCLASSES = {  # subclass -> (class, home structure)
    "037 DG Glut": ("04 DG-IMN Glut", "DG"), "016 CA1-ProS Glut": ("03 HPF Glut", "CA1"),
    "017 CA3 Glut": ("03 HPF Glut", "CA3"), "052 Pvalb Gaba": ("07 CTX-MGE GABA", "CA1"),
    "319 Astro-TE NN": ("30 Astro-Epen", "DG"), "327 Oligo NN": ("31 OPC-Oligo", "CA3"),
}


class FakeCache:
    current_manifest = "synthetic"

    def __init__(self, root):
        self.root = Path(root)

    def _p(self, directory, file_name, ext):
        return self.root / directory / (file_name.replace("/", "-") + ext)

    def get_file_path(self, directory, file_name, **_):
        return self._p(directory, file_name, ".h5ad")

    def get_metadata_dataframe(self, directory, file_name, **kw):
        kw.pop("skip_hash_check", None)
        return pd.read_csv(self._p(directory, file_name, ".csv"), **kw)

    def put_csv(self, directory, file_name, df):
        p = self._p(directory, file_name, ".csv"); p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)

    def put_h5ad(self, directory, file_name, adata):
        p = self._p(directory, file_name, ".h5ad"); p.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(p)


def expression(labels, n_genes, names):
    """Each subclass gets its own block of high genes."""
    k = len(names)
    X = rng.poisson(0.3, (len(labels), n_genes)).astype(np.float32)
    for i, name in enumerate(names):
        rows = np.where(labels == name)[0]
        cols = np.arange(i * (n_genes // k), i * (n_genes // k) + n_genes // (2 * k))
        X[np.ix_(rows, cols)] += rng.poisson(6, (len(rows), len(cols)))
    return sp.csr_matrix(np.log2(X + 1))


def build(cache, cfg):
    names = list(SUBCLASSES)
    # ---- 10x
    n = 1800
    sub = rng.choice(names, n, p=[.35, .25, .15, .05, .12, .08])
    cells = pd.DataFrame({
        "cell_label": [f"c{i}" for i in range(n)],
        "feature_matrix_label": cfg["region"]["tenx_feature_matrix"],
        "region_of_interest_acronym": "HIP", "donor_label": "d1", "donor_sex": "M",
        "library_label": "L1", "subclass": sub, "class": [SUBCLASSES[s][0] for s in sub],
        "supertype": sub, "cluster": sub, "neurotransmitter": "Glut"})
    other = cells.head(50).assign(feature_matrix_label="WMB-10Xv3-TH",
                                  cell_label=[f"o{i}" for i in range(50)])
    cache.put_csv("WMB-10X", "cell_metadata_with_cluster_annotation", pd.concat([cells, other]))
    genes = [f"ENSMUSG{i:011d}" for i in range(300)]
    symbols = ["Prox1", "Slc17a7", "Gad2", "Aqp4", "Mog"] + [f"Gene{i}" for i in range(295)]
    symbols[10] = symbols[11] = "Dup1"      # the real gene table has duplicated symbols
    X = expression(sub, 300, names).tolil()
    symbols[298], symbols[299] = "Rbfox3", "Gfap"     # NeuN gene in neurons, Gfap in astrocytes
    neuron = np.array([("Glut" in s_) or ("Gaba" in s_) for s_ in sub])
    astro = np.array(["Astro" in s_ for s_ in sub])
    X[:, 298] = (np.log2(rng.poisson(5, n) + 1) * (neuron & (rng.random(n) < 0.9)))[:, None]
    X[:, 299] = (np.log2(rng.poisson(5, n) + 1) * (astro & (rng.random(n) < 0.9)))[:, None]
    a = ad.AnnData(X.tocsr(),
                   obs=pd.DataFrame(index=cells["cell_label"].to_numpy()),
                   var=pd.DataFrame({"gene_symbol": symbols}, index=genes))
    cache.put_h5ad(cfg["region"]["tenx_dataset_dir"], cfg["region"]["tenx_feature_matrix"] + "/log2", a)

    # ---- MERFISH
    m = cfg["merfish"]
    n = 3000
    sub = rng.choice(names, n, p=[.30, .25, .15, .05, .15, .10])
    struct = np.array([SUBCLASSES[s][1] for s in sub])
    centers = {"DG": (2, 2), "CA1": (5, 2), "CA3": (3.5, 4)}
    xy = np.array([centers[s] for s in struct]) + rng.normal(0, .4, (n, 2))
    mc = pd.DataFrame({
        "cell_label": [f"m{i}" for i in range(n)],
        "brain_section_label": rng.choice(["sec.01", "sec.02"], n, p=[.7, .3]),
        "x": xy[:, 0], "y": xy[:, 1], "z": 1.0,
        "subclass": sub, "class": [SUBCLASSES[s][0] for s in sub], "supertype": sub,
        "cluster": sub, "neurotransmitter": "Glut", "subclass_color": "#3366aa"})
    cache.put_csv(m["dataset_dir"], "cell_metadata_with_cluster_annotation", mc)
    idx = {"DG": 1, "CA1": 2, "CA3": 3}
    pidx = np.array([idx[s] for s in struct]); pidx[:200] = 4      # 200 cells outside HPF
    cache.put_csv(m["dataset_dir"] + "-CCF", "ccf_coordinates", pd.DataFrame({
        "cell_label": mc["cell_label"], "x": 1.0, "y": 1.0, "z": 1.0, "parcellation_index": pidx}))
    cache.put_csv("Allen-CCF-2020", "parcellation_to_parcellation_term_membership_acronym", pd.DataFrame({
        "parcellation_index": [1, 2, 3, 4], "organ": "brain", "category": "grey",
        "division": ["HPF", "HPF", "HPF", "TH"], "structure": ["DG", "CA1", "CA3", "LP"],
        "substructure": ["DG-sg", "CA1", "CA3", "LP"]}))
    mg = [f"ENSMUSG{i:011d}" for i in range(60)]
    a = ad.AnnData(expression(sub, 60, names), obs=pd.DataFrame(index=mc["cell_label"].to_numpy()),
                   var=pd.DataFrame({"gene_symbol": symbols[:60]}, index=mg))
    cache.put_h5ad(m["dataset_dir"], m["matrix"], a)


def synthetic_image(path, size=1200, n=260):
    import tifffile
    from skimage.draw import disk
    from skimage.filters import gaussian
    img = np.zeros((3, size, size), np.float32)
    centres = rng.uniform(20, size - 20, (n, 2))
    neun = rng.random(n) < 0.5
    for (y, x), pos in zip(centres, neun):
        rr, cc = disk((y, x), rng.uniform(5, 7), shape=(size, size))
        img[0, rr, cc] = 1.0
        if pos:
            img[1, rr, cc] = 0.8
        else:
            rr, cc = disk((y, x), 11, shape=(size, size)); img[2, rr, cc] = 0.5
    img = np.stack([gaussian(c, 1) for c in img])
    img = (img * 3000 + 200 + rng.normal(0, 25, img.shape)).clip(0, 65535).astype(np.uint16)
    tifffile.imwrite(path, img, photometric="minisblack")
    return int(neun.sum()), n


def main():
    keep = len(sys.argv) > 1   # pass a folder to keep the outputs for inspection
    with tempfile.TemporaryDirectory() as tmp:
        if keep:
            tmp = sys.argv[1]; Path(tmp).mkdir(parents=True, exist_ok=True)
        cfg = load_config()
        cfg["_root"] = tmp
        cfg["scrna"].update(n_cells=1200, n_hvg=200, n_pcs=20)
        cfg["imaging"].update(roi_size=800, segmentation="classical")
        cache = FakeCache(Path(tmp) / "cache")
        build(cache, cfg)

        s = scrna.run(cfg, cache)
        assert s["n_cells_region"] == 1800 and s["n_subclasses"] == 6
        assert s["ARI_leiden_vs_subclass"] > 0.5, s

        m = merfish.run(cfg, cache)
        assert m["n_cells_division"] == 2800
        assert all(v > 0.9 for v in m["home_structure_check"].values()), m
        assert m["spearman_subclass_fraction_merfish_vs_scrna"] > 0.5

        img = Path(tmp) / "synthetic.tif"
        synthetic_image(img)
        i = imaging.run(cfg, img)
        assert i["n_nuclei"] > 50 and not i["qc_flags"], i
        assert 0.3 < i["fraction_NeuN_positive"] < 0.7, i

        # ---- identity model against MERFISH ground truth ----
        from atlaslens import reference
        v = reference.validate_on_merfish(cfg, "DG", ["Slc17a7", "Aqp4"])
        assert v["n_test_cells"] > 100 and 0 <= v["accuracy_all_cells"] <= 1, v

        # ---- LangGraph identify workflow (no language model needed) ----
        from atlaslens import agent
        asked = []

        def human(payload):
            asked.append(payload)
            return {"approve": True, "structure": "DG"}

        st_ = agent.run_identify(cfg, image=str(img), use_llm=False, ask=human)
        a = st_["assignment"]
        assert asked and "ROI was chosen automatically" in asked[0]["reasons"], asked
        assert a["structure"] == "DG" and st_["context"], a
        ph = a["phenotypes"]
        assert ph["NeuN+ GFAP-"]["best_class"] == "04 DG-IMN Glut" and ph["NeuN+ GFAP-"]["resolved"], ph
        assert ph["NeuN- GFAP+"]["best_class"] == "30 Astro-Epen" and ph["NeuN- GFAP+"]["resolved"], ph
        assert all(c["consistent"] for c in a["consistency"].values()), a["consistency"]
        # wrong structure: CA1 has no astrocytes in the synthetic reference, so GFAP-only cells
        # must be reported as unexplained, and a new ROI must loop back through quantify
        answers = [{"roi": [200, 200, 700, 700]}, {"approve": True, "structure": "CA1"}]
        st_ = agent.run_identify(cfg, image=str(img), use_llm=False, ask=lambda p: answers.pop(0))
        assert st_["imaging"]["roi_xywh"] == [200, 200, 700, 700] and not answers
        bad = st_["assignment"]["phenotypes"]["NeuN- GFAP+"]
        assert not bad["explained_by_reference"] and not bad["resolved"], bad
        # rejecting at the checkpoint stops the workflow
        st_ = agent.run_identify(cfg, image=str(img), use_llm=False, ask=lambda p: {"approve": False})
        assert st_["status"] == "stopped at human review"

        # ---- chat agent wiring, with a scripted stand-in for the language model ----
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        class Scripted(BaseChatModel):
            script: list = []

            @property
            def _llm_type(self):
                return "scripted"

            def bind_tools(self, tools, **kw):
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kw):
                return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])

        fake = Scripted(script=[
            AIMessage(content="", tool_calls=[{"name": "search_knowledge", "id": "1",
                                               "args": {"query": "GFAP ring over-calls astrocytes"}}]),
            AIMessage(content="", tool_calls=[{"name": "identify_cell_populations", "id": "2",
                                               "args": {"structure": "DG", "image": str(img)}}]),
            AIMessage(content="done")])
        out = agent.build_chat_agent(cfg, fake).invoke(
            {"messages": [("user", "Which cells are NeuN positive in this DG image?")]},
            {"configurable": {"thread_id": "t"}})
        tool_msgs = [m for m in out["messages"] if m.type == "tool"]
        assert len(tool_msgs) == 2 and "cell_identity.md" in tool_msgs[0].content, tool_msgs[0].content[:300]
        assert "needs_human_review" in tool_msgs[1].content and "04 DG-IMN Glut" in tool_msgs[1].content
        # the report node with a language model
        fake2 = Scripted(script=[AIMessage(content="REPORT TEXT")])
        g = agent.build_identify_graph(cfg, fake2)
        th = {"configurable": {"thread_id": "r"}}
        s2 = g.invoke({"image": str(img), "structure": "DG", "roi": [100, 100, 800, 800]}, th)
        assert s2["report"] == "REPORT TEXT" and not s2.get("__interrupt__"), s2.get("__interrupt__")

        for f in ("identify/report.md", "identify/nuclei_assigned.csv", "scrna/umap_subclass.png", "scrna/markers_by_subclass.csv", "scrna/run.json",
                  "merfish/section_subclass.png", "merfish/composition_merfish_vs_scrna.png",
                  "merfish/region_merfish.h5ad", "imaging/overlay.png", "imaging/nuclei.csv"):
            assert (Path(tmp) / "results" / f).exists(), f
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
