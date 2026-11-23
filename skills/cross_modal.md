# Skill: linking sequencing and imaging

## What the starter does
`merfish` compares subclass fractions between MERFISH cells in the CCF division and dissociated 10x cells
from the matching dissection (`subclass_fraction_merfish_vs_scrna.csv`, log2 ratio per subclass).

## How to read it
- Differences mix biology-free effects: dissociation survival (large projection neurons and some
  interneurons are lost, glia and small neurons survive), FACS gating or enrichment used for the 10x
  libraries, dissection borders versus CCF borders, and label-transfer error. Name these before biology.
- MERFISH counts are closer to true tissue composition; use them as the reference for abundance.

## Useful next analyses
- From scRNA-seq markers to a staining panel: for a subclass of interest, pick 2-3 genes that are specific
  in `markers_by_subclass.csv` and have good antibodies or probes; check their spatial pattern in MERFISH.
- From images to cell types: compare NeuN+ fraction per structure with the neuronal fraction per
  `parcellation_structure` in MERFISH (neuronal classes versus `NN` classes).
- With the lab's own spatial data, map scRNA-seq types into space with Tangram or cell2location.
