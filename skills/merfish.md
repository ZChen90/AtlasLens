# Skill: MERFISH / spatial transcriptomics (Allen whole-brain MERFISH)

## Reference data facts
- `MERFISH-C57BL6J-638850`: one brain, 59 coronal sections, about 4 million cells, 500 genes plus
  50 blank codewords. `x`, `y` in the cell metadata are section coordinates; `z` is the section position.
- CCF registration is in `MERFISH-C57BL6J-638850-CCF / ccf_coordinates` (`parcellation_index`), named via
  `Allen-CCF-2020 / parcellation_to_parcellation_term_membership_acronym`
  (`division` > `structure` > `substructure`, e.g. HPF > DG > DG-sg).
- Cell-type labels were transferred from the 10x taxonomy; the panel cannot resolve every cluster, so
  trust `class`/`subclass` more than `cluster`.

## Pipeline (`python -m atlaslens merfish`)
Outputs in `results/merfish/`: `section_subclass.png`, `section_parcellation_structure.png`,
`section_marker_genes.png`, `subclass_by_structure_counts.csv`, `region_merfish.h5ad` (with `obsm['spatial']`),
`subclass_fraction_merfish_vs_scrna.csv`, `composition_merfish_vs_scrna.png`, `run.json`.

## What to check
- `home_structure_check`: DG Glut cells in DG, CA1-ProS Glut in CA1, CA3 Glut in CA3. Expect high but not
  perfect fractions (CA1-ProS includes prosubiculum; boundaries are registration-limited). Low values mean
  a broken join or a registration problem; inspect the section plots.
- Blank codewords estimate the false-positive rate; a gene near blank level is not detected.
- Genes requested but not in the 500-gene panel are skipped silently in the plot: list which were plotted.
  Not in the panel: Prox1, Sst, Cx3cr1, Gad1, Wfs1, C1ql2, Cdh24, Olig1, Sncg. In the panel: Slc17a7, Slc17a6,
  Gad2, Pvalb, Vip, Lamp5, Calb1, Calb2, Fibcd1, Aqp4, Gfap, Mog, Opalin, Sox10, Igf2. Check `gene.csv` for others.
- The `HPF` division includes the retrohippocampal region (ENTl, ENTm, SUB, ProS, PRE, POST, PAR, APr, HATA)
  as well as CA1, CA2, CA3, DG, FC and IG. The 10x HPF dissections likewise cover HIP and RHP.
- Sections differ in quality. Before pooling sections, look at per-section cell counts.
