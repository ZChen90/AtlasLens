# Skill: fluorescence image segmentation and quantification

## Stand-in data
10x Visium IF section `V1_Adult_Mouse_Brain_Coronal_Section_2`: 20x widefield, channel 0 DAPI, 1 NeuN (FITC),
2 GFAP plus the Visium fiducial frame (TRITC). It is not confocal: more out-of-focus haze, no z-stack.
The fiducial dots and the slide frame in channel 2 are not biology; keep ROIs inside the tissue.

## Pipeline (`python -m atlaslens imaging [--image file.tif] [--roi X0 Y0 W H]`)
Overview -> ROI -> nuclei segmentation (Cellpose if installed, else threshold + watershed) ->
per-nucleus intensities (nucleus and perinuclear ring, background-subtracted) -> Otsu positivity call.
Outputs in `results/imaging/`: `overview.png`, `overlay.png`, `nuclei.csv`, `labels.npy`, `run.json`.

## What to check, every time
1. `overview.png`: is the ROI in the intended structure? `roi: auto` only finds the densest DAPI window.
2. `overlay.png`: outlines on nuclei, one per nucleus. Dense layers (DG granule, CA pyramidal) are where
   classical segmentation merges nuclei; use Cellpose there and set `nucleus_diameter_px`.
3. `qc_flags` in `run.json` must be empty, or be explained.
4. Positivity thresholds: plot the per-nucleus intensity histogram. Otsu is only valid if it is bimodal.
   Otherwise use a negative control or a fixed threshold agreed with the human.
5. Report the denominator: nuclei counted, area analysed, and the fraction positive.

## Conventions
- Nuclear markers (NeuN, Sox10, Olig2, Prox1) are read from the nucleus mask; cytoplasmic or fibrous
  markers (GFAP, Iba1) from the ring. GFAP ring signal says an astrocyte process is near the nucleus,
  which over-calls astrocytes; treat `GFAP_positive` as a screening value.
- Confocal z-stacks: analyse a single plane or a projection of a thin slab, state which, and never mix them.
  For 3D counting use 3D segmentation (Cellpose 3D / StarDist 3D), not per-plane counts summed.
- Never compare raw intensities across images acquired with different laser power, gain or exposure.
- Record pixel size (um/px) with every result; report densities per mm2 or mm3, not per image.
- Atlas-level questions need registration to the Allen CCF (brainreg / ABBA) before counting per region.
