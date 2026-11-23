# Skill: identifying cell populations in images from sequencing references

## Method
`identify` combines three sources: MERFISH gives the cell-type composition of the structure the image
shows (prior), scRNA-seq gives how likely each subclass is to be positive for each marker gene
(likelihood), and the image gives per-nucleus marker calls. A naive-Bayes posterior is reported for every
marker phenotype (for example `NeuN+ GFAP-`), at class and subclass level.

## Antibody to gene mapping (set in config.yaml, imaging.marker_genes)
NeuN = Rbfox3, GFAP = Gfap, Iba1 = Aif1, parvalbumin = Pvalb, somatostatin = Sst, calbindin = Calb1,
calretinin = Calb2, S100b = S100b, Olig2 = Olig2, Sox10 = Sox10, Prox1 = Prox1, c-Fos = Fos, VIP = Vip,
tyrosine hydroxylase = Th, ChAT = Chat.

## How to read the output
- `resolved: true` means one class has posterior of at least 0.8. Otherwise the marker panel cannot
  separate the candidates in that structure: say so and name the marker that would separate them.
- `explained_by_reference: false` means no cell type expected in that structure produces this phenotype
  (probability under the reference below 0.02). The posterior is then meaningless. Usual causes: wrong
  structure, a threshold artefact, or a cell type missing from the reference. Never report a best class for it.
- Subclass posteriors within a class mostly restate the MERFISH prior. Two or three markers resolve
  classes (neuron, astrocyte, oligodendrocyte), rarely subclasses. Do not over-read them.
- `consistency` compares the observed positive fraction with the fraction the references predict.
  A mismatch points to the image first: threshold, bleed-through, wrong structure, ROI at a border.
- The structure label drives the prior. A wrong structure gives a confident wrong answer. Confirm the ROI
  on the overview image before trusting the result.

## Known limits
- mRNA detection is not protein staining. Some transcripts are abundant with little protein and the
  reverse. Rbfox3 (NeuN) is absent or low in some neurons (for example Purkinje, mitral, some interneurons).
- scRNA-seq detection fractions suffer dropout, so sensitivity of a marker is underestimated for lowly
  expressed genes; nuclei data lose cytoplasmic transcripts.
- GFAP is measured in a perinuclear ring. Astrocyte processes pass close to neuronal nuclei, so `GFAP+`
  over-calls. A `NeuN+ GFAP+` phenotype is usually a neuron next to an astrocyte process, not a new cell type.
- Reactive states, disease models and development change marker expression; the Allen references are
  healthy adult mouse.
- Nuclei are assigned by phenotype, so all nuclei with the same marker calls get the same answer.

## Validation on Allen MERFISH (22 Nov 2026)
`python -m atlaslens validate --structure DG --genes Slc17a7 Gad2 Aqp4 Mog` treats MERFISH cells of one
section as if they were nuclei in an image and compares the assigned class with the true label.
Reference: 12,239-cell scRNA-seq subsample; prior from the other sections.

| Structure | Test cells | Cells in resolved phenotypes | Accuracy within resolved | Always-most-common baseline |
|---|---|---|---|---|
| DG | 7,188 | 68% | 86% | 45% |
| CA1 | 8,905 | 59% | 88% | 43% |
| CA3 | 7,985 | 21% | 87% | 33% |

What this showed
- Resolved calls are right 86-88% of the time, but the stated posteriors are too confident: the
  Aqp4-only phenotype gets 0.96-0.97 for astrocytes while 79-85% truly are (11-16% are vascular cells,
  which sit next to astrocyte endfeet). Treat posteriors above 0.9 as "about 85%".
- Cells positive for both a neuronal and a glial marker (Slc17a7+ Aqp4+) are mostly neurons (65-71%), not
  astrocytes: signal from a neighbouring cell. The model leans to astrocyte but marks them unresolved.
  The same will happen with GFAP next to NeuN+ nuclei in images.
- Slc17a7 cannot separate CA3 pyramidal cells (class 01 IT-ET Glut) from DG granule cells (04 DG-IMN Glut).
  In the atlas registration 29% of DG granule cells fall inside the CA3 label, so in CA3 most excitatory
  cells stay unresolved at class level. Add Prox1 (DG) or report "glutamatergic neuron" only.
- MGE versus CGE interneurons are not separable with Gad2 alone; vascular and immune cells are never
  resolved without their own marker. All-negative cells are a mixture, not a cell type.
