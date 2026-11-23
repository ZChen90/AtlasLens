"""AtlasLens: a LangGraph agent that identifies cell populations in brain-section images
using scRNA-seq and MERFISH references.

Two entry points
  identify : a fixed LangGraph workflow with a human checkpoint
             retrieve knowledge -> quantify image -> human review -> build references
             -> assign cell types -> write report
  chat     : a tool-calling agent for open questions; it can call every pipeline and the
             identify workflow, and searches the lab knowledge base (RAG) before interpreting.

Numbers are always computed by the pipelines. The language model plans, explains and writes;
it is instructed to use only numbers present in the tool results.

  python -m atlaslens.agent identify --image section.tif --structure DG
  python -m atlaslens.agent identify --no-llm          # deterministic run, template report
  python -m atlaslens.agent chat
"""
from __future__ import annotations
import argparse, json, os, uuid
from pathlib import Path
from typing import Any, Optional, TypedDict

import pandas as pd
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from . import imaging, knowledge, reference
from .common import ROOT, load_config, results_dir, write_run_record

SYSTEM_PROMPT = (ROOT / "prompts" / "system.md")


def system_prompt() -> str:
    return SYSTEM_PROMPT.read_text() if SYSTEM_PROMPT.exists() else "You are a careful neuroscience data analyst."


def get_llm(cfg):
    from langchain.chat_models import init_chat_model
    a = cfg.get("agent", {})
    name = os.environ.get("ATLASLENS_LLM") or a.get("llm")
    if not name:
        raise SystemExit('No language model configured. Set agent.llm in config.yaml or the ATLASLENS_LLM '
                         'variable to "provider:model", or run with --no-llm.')
    return init_chat_model(name, temperature=a.get("temperature", 0))


# ----------------------------------------------------------------------------
# identify workflow
# ----------------------------------------------------------------------------
class IdentifyState(TypedDict, total=False):
    image: Optional[str]
    structure: Optional[str]
    roi: Any                      # "auto" or [x0, y0, w, h]
    context: list                 # retrieved knowledge chunks
    imaging: dict                 # imaging summary (run.json content)
    review: dict                  # what the human decided
    assignment: dict
    report: str
    status: str


def build_identify_graph(cfg, llm=None, checkpointer=None):
    marker_genes = cfg["imaging"]["marker_genes"]

    def retrieve(state: IdentifyState):
        q = (f"identify cell population markers {' '.join(marker_genes)} {' '.join(marker_genes.values())} "
             f"structure {state.get('structure') or ''} segmentation threshold positivity limits")
        return {"context": knowledge.search(q, k=5)}

    def quantify(state: IdentifyState):
        run_cfg = {**cfg, "imaging": {**cfg["imaging"], "roi": state.get("roi") or cfg["imaging"]["roi"]}}
        return {"imaging": imaging.run(run_cfg, state.get("image"))}

    def review(state: IdentifyState):
        """Human checkpoint. Runs when the ROI was automatic, QC raised flags or no structure is known."""
        im = state["imaging"]
        reasons = []
        if (state.get("roi") or cfg["imaging"]["roi"]) in (None, "auto"):
            reasons.append("ROI was chosen automatically")
        if im.get("qc_flags"):
            reasons.append("segmentation QC flags: " + "; ".join(im["qc_flags"]))
        if not state.get("structure"):
            reasons.append("the anatomical structure of the ROI is not set")
        if not reasons:
            return {"review": {"needed": False}}
        answer = interrupt({
            "reasons": reasons, "roi_xywh": im["roi_xywh"], "n_nuclei": im["n_nuclei"],
            "look_at": [str(results_dir(cfg, "imaging") / f) for f in ("overview.png", "overlay.png")],
            "question": "Approve this ROI and segmentation? Reply with JSON, e.g. "
                        '{"approve": true, "structure": "DG"}  or  {"roi": [x0, y0, w, h]} to redo.',
        })
        answer = answer if isinstance(answer, dict) else {"approve": bool(answer)}
        update: dict = {"review": {"needed": True, "reasons": reasons, **answer}}
        if answer.get("structure"):
            update["structure"] = answer["structure"]
        if answer.get("roi"):
            update["roi"] = list(answer["roi"])
        return update

    def after_review(state: IdentifyState):
        r = state.get("review", {})
        if r.get("roi"):
            return "quantify"                      # new ROI: measure again, then review again
        if r.get("needed") and not r.get("approve"):
            return "rejected"
        return "assign"

    def rejected(state: IdentifyState):
        return {"status": "stopped at human review", "report": "Stopped: ROI or segmentation was not approved."}

    def assign(state: IdentifyState):
        structure = state.get("structure") or cfg["imaging"].get("structure")
        nuclei = pd.read_csv(results_dir(cfg, "imaging") / "nuclei.csv")
        profile = reference.marker_profile(cfg, list(marker_genes.values()))
        prior, n_ref, _ = reference.structure_composition(cfg, structure)
        table, summary = reference.assign(nuclei, marker_genes, profile, prior)
        summary = {"structure": structure or cfg["region"]["parcellation_division"],
                   "n_merfish_cells_in_structure": n_ref, **summary}
        out = results_dir(cfg, "identify")
        table.to_csv(out / "nuclei_assigned.csv", index=False)
        profile.to_csv(out / "marker_detection_by_subclass.csv")
        prior.rename("fraction").to_csv(out / "structure_composition.csv")
        write_run_record(out, "identify", {"marker_genes": marker_genes, "image": state.get("image")}, summary)
        return {"assignment": summary, "status": "ok"}

    def report(state: IdentifyState):
        a, im = state["assignment"], state["imaging"]
        if llm is None:
            text = template_report(a, im)
        else:
            ctx = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in state.get("context", []))
            msg = llm.invoke([
                ("system", system_prompt()),
                ("user",
                 "Write the analysis report for this image.\n"
                 "Use only numbers that appear in the JSON below. Structure: 1) answer in two sentences: "
                 "which cell population each marker phenotype is, and how sure; 2) evidence per phenotype; "
                 "3) consistency between image and references; 4) what is not resolved and which extra "
                 "marker would resolve it; 5) checks the scientist should do. Cite knowledge sources in "
                 "square brackets when you use them.\n\n"
                 f"IMAGING SUMMARY\n{json.dumps(im, indent=1)}\n\n"
                 f"CELL IDENTITY RESULT\n{json.dumps(a, indent=1)}\n\n"
                 f"HUMAN REVIEW\n{json.dumps(state.get('review', {}))}\n\n"
                 f"LAB KNOWLEDGE (retrieved)\n{ctx}")])
            text = msg.content if isinstance(msg.content, str) else "".join(
                b.get("text", "") for b in msg.content if isinstance(b, dict))
        (results_dir(cfg, "identify") / "report.md").write_text(text)
        return {"report": text}

    g = StateGraph(IdentifyState)
    for name, fn in [("retrieve", retrieve), ("quantify", quantify), ("review", review),
                     ("rejected", rejected), ("assign", assign), ("report", report)]:
        g.add_node(name, fn)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "quantify")
    g.add_edge("quantify", "review")
    g.add_conditional_edges("review", after_review, ["quantify", "rejected", "assign"])
    g.add_edge("assign", "report")
    g.add_edge("report", END)
    g.add_edge("rejected", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def template_report(a: dict, im: dict) -> str:
    lines = [f"# Cell identity report: {im['image']}", "",
             f"Structure: {a['structure']} ({a['n_merfish_cells_in_structure']} MERFISH reference cells). "
             f"ROI {im['roi_xywh']}, {a['n_nuclei']} nuclei, segmentation: {im['segmentation']}.", "",
             "| Phenotype | Nuclei | Fraction | Most likely class | Posterior | Resolved |", "|---|---|---|---|---|---|"]
    for code, p in a["phenotypes"].items():
        lines.append(f"| {code} | {p['n_nuclei']} | {p['fraction_of_nuclei']} | {p['best_class']} | "
                     f"{p['best_class_posterior']} | "
                     f"{'yes' if p['resolved'] else 'no' if p['explained_by_reference'] else 'no: not explained by the cell types of this structure'} |")
    lines += ["", "## Image versus references", ""]
    for m, c in a["consistency"].items():
        lines.append(f"- {m} ({c['gene']}): observed {c['observed_fraction']}, expected {c['expected_fraction']}"
                     f"{'' if c['consistent'] else '  <- mismatch, check threshold, ROI and structure'}")
    if im.get("qc_flags"):
        lines += ["", "## Segmentation QC flags", ""] + [f"- {f}" for f in im["qc_flags"]]
    lines += ["", "Model: naive Bayes over MERFISH composition (prior) and scRNA-seq detection fractions "
                  "(likelihood). mRNA detection is used as a proxy for antibody staining."]
    return "\n".join(lines)


def run_identify(cfg, image=None, structure=None, roi=None, use_llm=True, ask=None) -> dict:
    """Run the workflow, pausing at the human checkpoint. `ask(payload) -> dict` supplies the
    human answer (default: prompt in the terminal)."""
    llm = get_llm(cfg) if use_llm else None
    graph = build_identify_graph(cfg, llm)
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = graph.invoke({"image": image, "structure": structure or cfg["imaging"].get("structure"),
                          "roi": roi}, thread)
    while state.get("__interrupt__"):
        payload = state["__interrupt__"][0].value
        answer = (ask or _ask_terminal)(payload)
        state = graph.invoke(Command(resume=answer), thread)
    return state


def _ask_terminal(payload: dict) -> dict:
    print("\n=== HUMAN REVIEW NEEDED ===")
    for r in payload["reasons"]:
        print(" -", r)
    print(f"ROI (x0, y0, w, h): {payload['roi_xywh']}   nuclei: {payload['n_nuclei']}")
    print("Open:", *payload["look_at"], sep="\n  ")
    raw = input(payload["question"] + "\n> ").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"approve": raw.lower() in ("y", "yes", "ok", "approve")}


# ----------------------------------------------------------------------------
# chat agent (tool-calling)
# ----------------------------------------------------------------------------
def make_tools(cfg):
    def _j(x) -> str:
        text = json.dumps(x, default=str)
        return text if len(text) <= 12000 else text[:12000] + " ...[truncated; read the files for the rest]"

    @tool
    def search_knowledge(query: str) -> str:
        """Search the lab knowledge base (conventions, pitfalls, protocols, antibody notes).
        Call this before interpreting any result."""
        return _j(knowledge.search(query, k=4))

    @tool
    def run_pipeline(step: str) -> str:
        """Run a validated pipeline on the configured data and return its summary.
        step is one of: 'scrna' (cluster the region's scRNA-seq, markers, agreement with the Allen
        taxonomy), 'merfish' (spatial cell-type maps, composition per structure, comparison with scRNA-seq)."""
        if step == "scrna":
            from . import scrna
            return _j(scrna.run(cfg))
        if step == "merfish":
            from . import merfish
            return _j(merfish.run(cfg))
        return "Unknown step. Use 'scrna' or 'merfish'."

    @tool
    def read_results(step: str) -> str:
        """Read the stored summary of the last run of a step: 'scrna', 'merfish', 'imaging' or 'identify'."""
        p = results_dir(cfg, step) / "run.json"
        return p.read_text()[:6000] if p.exists() else f"No results for {step} yet."

    @tool
    def marker_expression_by_cell_type(genes: list[str]) -> str:
        """For each gene, the cell-type subclasses in which it is most often detected in the region's
        scRNA-seq reference (fraction of cells with the gene detected)."""
        prof = reference.marker_profile(cfg, genes)
        return _j({g: prof[g].sort_values(ascending=False).head(10).round(2).to_dict() for g in genes})

    @tool
    def structure_composition(structure: str = "") -> str:
        """Cell-type composition of an Allen CCF structure (e.g. 'DG', 'CA1', 'CA3') from MERFISH.
        Empty string = whole region. Also lists the available structure names."""
        prior, n, available = reference.structure_composition(cfg, structure or None)
        return _j({"n_cells": n, "top_subclasses": prior.head(12).round(3).to_dict(), "available": available})

    @tool
    def identify_cell_populations(structure: str, image: str = "", roi: Optional[list[int]] = None) -> str:
        """Main task. Segment nuclei in a fluorescence image, call marker-positive cells, and infer which
        cell population each marker phenotype is, using MERFISH composition of `structure` as prior and
        scRNA-seq marker detection as likelihood. image: path to a (C,Y,X) TIFF, empty = configured image.
        roi: [x0, y0, width, height] in pixels; omit for automatic. If the result contains
        'needs_human_review', show the reasons and files to the user and ask before relying on it."""
        notes = {}

        def auto_ask(payload):          # in chat, surface the checkpoint instead of blocking
            notes["needs_human_review"] = {k: payload[k] for k in ("reasons", "roi_xywh", "look_at")}
            return {"approve": True, "structure": structure}

        state = run_identify(cfg, image or None, structure, roi, use_llm=False, ask=auto_ask)
        a = state.get("assignment") or {}
        compact = {k: v for k, v in a.items() if k != "positive_subclasses_per_marker"}
        return _j({**notes, "imaging": state.get("imaging"), "assignment": compact,
                   "files": str(results_dir(cfg, "identify"))})

    return [search_knowledge, run_pipeline, read_results, marker_expression_by_cell_type,
            structure_composition, identify_cell_populations]


def build_chat_agent(cfg, llm=None):
    from langchain.agents import create_agent
    return create_agent(model=llm or get_llm(cfg), tools=make_tools(cfg),
                        system_prompt=system_prompt(), checkpointer=InMemorySaver())


def chat(cfg):
    agent = build_chat_agent(cfg)
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    print("AtlasLens chat. Empty line or Ctrl-C to quit.")
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        out = agent.invoke({"messages": [("user", q)]}, thread)
        msg = out["messages"][-1]
        text = msg.content if isinstance(msg.content, str) else "".join(
            b.get("text", "") for b in msg.content if isinstance(b, dict))
        print("\nagent>", text)


def main():
    ap = argparse.ArgumentParser(prog="atlaslens.agent", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["identify", "chat"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--image", default=None, help="(C, Y, X) TIFF; default is the configured 10x section")
    ap.add_argument("--structure", default=None, help="Allen CCF structure shown in the ROI, e.g. DG")
    ap.add_argument("--roi", type=int, nargs=4, metavar=("X0", "Y0", "W", "H"))
    ap.add_argument("--no-llm", action="store_true", help="skip the language model; write a template report")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.mode == "chat":
        chat(cfg)
        return
    state = run_identify(cfg, a.image, a.structure, a.roi, use_llm=not a.no_llm)
    print("\n" + state.get("report", ""))
    print(f"\nFiles: {results_dir(cfg, 'identify')}")


if __name__ == "__main__":
    main()
