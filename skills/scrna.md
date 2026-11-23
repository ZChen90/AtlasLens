# Skill: scRNA-seq (Allen 10x whole-mouse-brain data and the lab's own data)

## Reference data facts
- `WMB-10Xv3-HPF/log2`: hippocampal-formation dissections, log2(CPM+1), var index = Ensembl ID,
  `gene_symbol` column. Labels come from `WMB-10X / cell_metadata_with_cluster_annotation`
  (`class` > `subclass` > `supertype` > `cluster`). Only QC-passed cells have labels.
- Dissection label (`region_of_interest_acronym`: HIP, RHP, ...) is where tissue was cut, not where each
  cell truly sat. Neighbouring-region cell types are expected at low frequency.

## Pipeline (`python -m atlaslens scrna`)
Subsample (rare subclasses kept) -> HVG -> PCA -> neighbours -> UMAP -> Leiden -> markers.
Outputs in `results/scrna/`: UMAPs, `dotplot_markers_by_class.png`, `markers_by_subclass.csv`,
`composition.csv`, `region_subsample.h5ad`, `run.json`.

## What to check
- `ARI_leiden_vs_class` in `run.json`. On this reference data expect clear agreement; a low value
  means the embedding or the label join is broken, not that the biology is new.
- Dot plot sanity: Prox1 in DG granule cells, Gad1/Gad2 in GABAergic classes, Aqp4 in astrocytes,
  Mog/Plp1 in oligodendrocytes, Cx3cr1 in microglia, Cldn5 in endothelium.
- UMAP by `donor_label` / `library_label` for batch structure before interpreting clusters.

## For the lab's own data (not yet wired in)
- Start from raw counts. QC per sample: genes/cell, UMIs/cell, mitochondrial fraction; pick thresholds
  from the distributions and record them. Nuclei: expect near-zero mito, flag samples that are not.
- Doublets: scrublet or scDblFinder per sample. Ambient RNA matters in brain (neuronal transcripts in glia).
- Annotate by mapping to the Allen taxonomy (MapMyCells, or label transfer from this reference), then
  confirm with markers. Report mapping confidence; never force a label on low-confidence cells.
- Differential expression across conditions: pseudobulk per animal, then edgeR/DESeq2-style tests.
