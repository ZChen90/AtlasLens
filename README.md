# AtlasLens

A standalone LangGraph agent for mouse-brain data. It analyses single-cell RNA-seq and MERFISH data and
uses them as references to identify which cell population the marker-positive cells in a fluorescence
image of a brain section belong to.

## How it is built
| Part | Technology | Where |
|---|---|---|
| Workflow with human checkpoint | LangGraph `StateGraph`, `interrupt`, checkpointer | `atlaslens/agent.py` (`identify`) |
| Open-question agent with tools | LangChain `create_agent` + tool calling | `atlaslens/agent.py` (`chat`) |
| Lab knowledge retrieval (RAG) | TF-IDF retriever over `skills/` and `knowledge/` | `atlaslens/knowledge.py` |
| Language model | any LangChain chat model, set in `config.yaml` (`agent.llm`) | needs your API key |
| Analysis (all numbers) | scanpy, anndata, scikit-image, optional Cellpose | `scrna.py`, `merfish.py`, `imaging.py` |
| Cell identity model | naive Bayes: MERFISH composition x scRNA-seq marker detection | `atlaslens/reference.py` |

The language model plans, explains and writes the report. It never produces the numbers: those come from
the pipelines, and the workflow can run without a model (`--no-llm`).

### The identify workflow
    retrieve knowledge -> quantify image -> HUMAN REVIEW -> build references -> assign -> report
- **quantify**: segment nuclei, measure each marker per nucleus, call positive cells.
- **human review**: pauses when the ROI was automatic, QC raised flags or the structure is unknown.
  You approve, name the structure (e.g. DG) or give a new ROI, which sends it back to quantify.
- **assign**: for each marker phenotype (e.g. `NeuN+ GFAP-`), posterior over cell classes and subclasses,
  using the MERFISH composition of that structure as prior and scRNA-seq detection of the marker genes
  as likelihood. Also checks that the observed positive fractions match what the references predict.
- **report**: answer, evidence, unresolved phenotypes and which extra marker would resolve them.

## Setup
    git clone https://github.com/ZChen90/AtlasLens.git
    cd AtlasLens
    bash setup.sh                      # creates the conda env "atlaslens"
    conda activate atlaslens
    python tests/smoke_test.py         # synthetic data, should end with SMOKE TEST PASSED
    python -m atlaslens plan          # lists every file and its size; downloads nothing
    python -m atlaslens download      # 21.3 GB, asks y/N (or double-click download_data.command in Finder)
    python -m atlaslens scrna         # builds the scRNA-seq reference
    python -m atlaslens merfish       # builds the spatial reference
    python -m atlaslens validate --structure DG --genes Slc17a7 Gad2 Aqp4 Mog   # check the identity model

## Running the agent
    export ATLASLENS_LLM="provider:model"   # e.g. an OpenAI, Google or local Ollama model
    export OPENAI_API_KEY=...             # the key for the provider you chose
    python -m atlaslens.agent identify --structure DG              # the 10x DAPI/NeuN/GFAP section
    python -m atlaslens.agent identify --image my.tif --structure CA1 --roi 4000 6000 2048 2048
    python -m atlaslens.agent identify --structure DG --no-llm     # no model, template report
    python -m atlaslens.agent chat                                 # open questions

Results: `results/identify/report.md`, `nuclei_assigned.csv` (one row per nucleus with phenotype and
assigned class), `marker_detection_by_subclass.csv`, `structure_composition.csv`, `run.json`.

For your own images set the channel order (`imaging.channels`) and the antibody-to-gene mapping
(`imaging.marker_genes`) in `config.yaml`. Add protocols and antibody notes to `knowledge/`.

## Data
| Modality | Dataset | Source |
|---|---|---|
| scRNA-seq | Allen whole mouse brain 10Xv3, hippocampal formation (`WMB-10Xv3-HPF/log2`) + taxonomy labels | Allen Brain Cell Atlas, CC BY-NC 4.0 |
| Spatial | Allen whole-brain MERFISH `C57BL6J-638850` (500 genes) + CCF parcellation | Allen Brain Cell Atlas, CC BY 4.0 |
| Images | 10x Visium IF coronal section 2: DAPI, NeuN, GFAP (20x widefield, stand-in for confocal) | 10x Genomics, CC BY 4.0 |

Exact sizes are printed by `plan`. To change region, edit `region:` in `config.yaml`
(e.g. `WMB-10Xv3-TH` with division `TH`).

## Layout
    config.yaml          parameters, channel and marker mapping, model choice
    atlaslens/          agent.py, reference.py, knowledge.py, scrna.py, merfish.py, imaging.py, data.py, fetch.py
    download_data.command  double-click downloader (standard-library Python, resumable)
    prompts/system.md    the agent's instructions
    skills/              method conventions and pitfalls (searched by the agent)
    knowledge/           your own notes and protocols (searched by the agent)
    tests/smoke_test.py  regression test on synthetic data, including the agent graphs
    data/  results/      created on first use

## Download size (checked 22 Nov 2026, atlas release 20261130)
| File | Size |
|---|---|
| `WMB-10Xv3-HPF-log2.h5ad` | 7.41 GB |
| `C57BL6J-638850-log2.h5ad` (MERFISH) | 7.63 GB |
| 10x cell metadata with annotation | 1.39 GB |
| MERFISH cell metadata with annotation | 1.02 GB |
| MERFISH CCF coordinates | 0.30 GB |
| Gene tables, parcellation terms | < 0.01 GB |
| 10x IF `image.tif` | 3.53 GB |
| 10x `spatial.tar.gz`, `filtered_feature_bc_matrix.h5` | 0.02 GB |
| **Total** | **21.3 GB** |

## Status
Tested:
- All three pipelines on synthetic data with the real file layout (`tests/smoke_test.py`).
- Every Allen file resolves in the current manifest; metadata column names, the `HPF` division and its
  structures, the taxonomy subclass names and the MERFISH gene panel were checked against the real tables.
- The three 10x download links respond; the image is a 3.5 GB BigTIFF.
- The image reader on four TIFF layouts (uncompressed, tiled-compressed pages, one series per channel,
  interleaved), giving identical results. It reads one channel at a time, so peak memory is about one channel.

- The LangGraph workflow end to end on synthetic data: checkpoint, approval, rejection, ROI loop wiring,
  assignment and report; and the chat agent's tool calls, using a scripted stand-in for the language model.

- **On the real Allen data** (22 Nov 2026, on a 4 GB machine, scRNA-seq subsampled to 12,239 cells):
  the scRNA-seq pipeline (Leiden clusters versus Allen subclasses: ARI 0.75, AMI 0.90), the MERFISH pipeline
  (304,736 hippocampal-formation cells, 31 sections) and the identity model against MERFISH ground truth
  (86-88% correct within resolved phenotypes; details and limits in `skills/cell_identity.md`).

Not yet tested:
- The scRNA-seq pipeline at the full 60,000-cell setting, and the image workflow on a real image
  (the 10x section was not in `data/` when the tests ran).
- The agent with a real language model (no API access in the test environment).
- Cellpose inference. Without Cellpose the classical segmentation is used.
- The cell identity model on antibody images. It was validated on MERFISH transcripts only; validate it
  on stained sections where you know the answer before relying on it.

