# Reproducing the CV bootcamp submission

This folder contains the notebooks, source scripts, and model definition for the CV bootcamp submission. The required architecture is `model.py`; models are trained from scratch using the supplied data. Experiment artifacts are organized separately under the submission's `../artifacts/` folder: checkpoints in `checkpoints/`, training histories in `training_histories/`, retained indices in `indices/`, and optional Task 5 method predictions in `supplemental_predictions/`.

## Snapshot status

The prediction files and reported metrics for Tasks 1–5 are present. The submission folder passes `check_format.py` for all five tasks. The final ZIP has not yet been rebuilt or validated.

Task 4's best recorded method is Co-Teaching: 56.22% noisy-label validation accuracy at epoch 40 (net1), versus 55.10% for label smoothing and 54.36% for filtering. Task 5 compares the Task 1 cross-entropy baseline (49.84%) with the best Task 4 method, Co-Teaching (49.26%), retrained on synthetic labels. Other Task 5 experiments are retained as supplemental artifacts but are not the required Task 5 comparison.

## Data

The supplied data files are in the submission's sibling `../data/` folder:

- `train_images.npy`, `test_images.npy`, `labels.csv`
- `train_idx.txt`, `val_idx.txt`, `classes.txt`

The model definition, `model.py`, is in this `code/` folder.

## Commands

`task5.ipynb` contains training and inference cells for the Task 5 baseline, label smoothing, filtering, and Co-Teaching methods. The required Task 5 prediction files are `../predictions/t5_baseline_synth.csv` and `../predictions/t5_best_synth.csv`; the latter contains predictions from Task 4's best method, Co-Teaching, retrained on synthetic labels. Optional Task 5 filtering, label-smoothing, and co-teaching CSVs are stored in `../artifacts/supplemental_predictions/`. The separate Task 4 Co-Teaching training notebook or script was not supplied; its checkpoint and history are in `../artifacts/`, but that run cannot be fully reproduced from this folder alone.

## Reproducibility note

The notebook does not set a seed for the Task 1 or Task 2 runs. The five-fold Task 3 script uses seed 2026. The Task 4 label-smoothing run and the Task 5 notebook use seed 42. The template has one seed field; `results.json` records 42, so do not interpret it as a shared seed for every experiment. Seeds for the other runs were not confirmed and are not inferred here.

## Outputs

Prediction CSVs are stored in the submission's sibling `../predictions/` folder. Training checkpoints and histories are kept in `../artifacts/` rather than mixed with source files. Run the supplied `check_format.py` against the complete submission folder and ZIP before submitting.
