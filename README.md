# Inter IIT CV Bootcamp — Submission Overview

**Participant:** Tanish Jain  
**Roll number:** 25B4244

This repository contains the complete working submission for the Inter IIT Tech Meet 15.0 Computer Vision bootcamp problem statement. It includes the dataset files, notebooks and scripts, task predictions, experiment records, `results.json`, and the written analysis in `report.md`.

## Results at a glance

| Task | Method / result | Validation result |
|---|---|---:|
| 1 | Clean-label reference model | 59.06% accuracy on noisy-label validation |
| 1 | Noisy-label cross-entropy baseline | 50.10% accuracy on noisy-label validation |
| 2 | Early stopping | Best noisy-label validation accuracy: 55.50% at epoch 35 |
| 2 | Memorization evidence | Mislabeled-example training accuracy rises to 67.1% by epoch 60; memorization becomes evident around epoch 44 |
| 3 | Five-fold out-of-fold score, `1 − P(noisy label)` | ROC-AUC 0.8848; precision@K 77.09% |
| 4 | Co-Teaching — best Task 4 method | 56.22% noisy-label validation accuracy at epoch 40 (net1) |
| 4 | Label Smoothing | 55.10% noisy-label validation accuracy |
| 4 | Filtering | 54.36% noisy-label validation accuracy |
| 5 | Task 1 cross-entropy baseline retrained on synthetic labels | 49.84% synthetic-label validation accuracy |
| 5 | Best Task 4 method (Co-Teaching) retrained on synthetic labels | 49.26% synthetic-label validation accuracy |

Task 5 follows the problem statement: it compares the Task 1 baseline against the **best Task 4 method**, Co-Teaching, with both retrained on the synthetic-label column. `t5_best_synth.csv` therefore contains Task 5 Co-Teaching predictions, even though the synthetic validation baseline scored 0.58 percentage points higher. Optional Task 5 label-smoothing and filtering runs are retained as supplementary experiments, not substituted for the required comparison.

## Approach and findings

### Task 1 — Label-noise baseline

The clean-reference model was trained using clean labels and reached 59.06% accuracy when evaluated against noisy validation labels. The Task 1 cross-entropy baseline trained on noisy labels reached 50.10%. The observed training-set noise rate was 40.24%. Predictions are provided as `predictions/t1_clean_ref.csv` and `predictions/t1_noisy_ce.csv`.

### Task 2 — Training dynamics and early stopping

The training-dynamics analysis tracks performance on clean and mislabeled examples over epochs. Noisy-label validation accuracy peaks at epoch 35 (55.50%); the analysis identifies memorization of wrong labels around epoch 44. Training accuracy on mislabeled examples grows from 18.6% at epoch 41 to 21.8% at epoch 44 and 67.1% by epoch 60. The dynamics and early-stopping outputs are `predictions/t2_dynamics.csv` and `predictions/t2_early_stop.csv`.

### Task 3 — Noise scoring

The five-fold out-of-fold model assigns each training example a score based on `1 − P(noisy label)`. This score reaches ROC-AUC 0.8848 and precision@K 77.09%, with K set to the known number of mislabeled training examples. High scores can also flag difficult but correctly labeled images, so the score is best interpreted as a ranking signal rather than proof that an individual label is wrong. The required output is `predictions/t3_noise_scores.csv`.

### Task 4 — Noise-robust methods

Three methods were evaluated against noisy-label validation accuracy. Co-Teaching ranks first at 56.22% (epoch 40, net1), followed by Label Smoothing at 55.10% and Filtering at 54.36%. Co-Teaching leads Label Smoothing by 1.12 percentage points and Filtering by 1.86 points. The three prediction files are `predictions/t4_coteaching.csv`, `predictions/t4_label_smoothing.csv`, and `predictions/t4_filtering.csv`.

### Task 5 — Real versus synthetic noise

The required Task 5 pair is the Task 1 cross-entropy baseline and the Task 4 winner, Co-Teaching, retrained on `synthetic_label`. Their synthetic-label validation accuracies are 49.84% and 49.26%, respectively. Thus Co-Teaching leads on the real human-noise validation in Task 4, while the baseline is ahead by 0.58 percentage points on synthetic noise in Task 5. This difference indicates the ranking depends on the noise setting; the results alone do not isolate its cause. Required prediction files: `predictions/t5_baseline_synth.csv` and `predictions/t5_best_synth.csv`.

## Training setup and reproducibility

The reported primary training setup uses ResNet-18, SGD (learning rate 0.1, momentum 0.9, weight decay `5e-4`), batch size 128, 60 epochs, cosine annealing, random crop with padding 4, and random horizontal flip. The reported GPU is a Tesla T4. Run-specific configurations are recorded in the notebooks, scripts, and histories; do not assume every run shares one seed. The Task 3 five-fold script sets seed 2026, while Task 4 and Task 5 runs include seed 42. Task 1 and Task 2 seed settings were not recorded. `results.json` preserves the required template fields and its seed value should not be read as a universal seed for every experiment.

The separate Task 4 Co-Teaching training notebook or script was not supplied. Its best/latest checkpoints, history, and prediction file are included, but that particular run is not fully reproducible from this package alone. The Task 3 five-fold run and Task 5 workflows are represented by their supplied scripts/notebooks and artifacts.

## Directory guide

```text
25B4244_cv_bootcamp/
├── README.md                    # This overview
├── report.md                    # Required Q1–Q5 written answers
├── results.json                 # Metrics in the supplied results schema
├── data/                        # Classes, labels, image arrays, and split indices
├── code/                        # Notebooks, model definition, and scripts
├── predictions/                 # Required prediction CSVs for Tasks 1–5
└── artifacts/
    ├── checkpoints/             # Best/latest model checkpoints
    ├── training_histories/      # Per-run training history CSVs
    ├── indices/                 # Retained training indices
    └── supplemental_predictions/ # Optional Task 5 method predictions
```

### Important code files

- `code/task1_task2_task3_task4.ipynb` — combined exploration and training workflow for Tasks 1–4.
- `code/task3_task4.ipynb` — Task 3 five-fold scoring, Task 4 filtering, and Co-Teaching workflow.
- `code/task5.ipynb` — Task 5 baseline and method experiments, including prediction generation.
- `code/task3_5fold_oof_colab.py` — reusable Task 3 five-fold out-of-fold scoring script.
- `code/model.py` — ResNet-18 model definition.
- `code/predict_t4_filtering.py` — Task 4 filtering prediction utility.

`code/README.md` contains additional implementation notes. `artifacts/supplemental_predictions/` contains optional Task 5 Co-Teaching, Filtering, and Label Smoothing prediction files; only the baseline and `t5_best_synth.csv` belong in the required Task 5 prediction pair.

## Browsing the GitHub repository

The public repository is [tanishjain0911-beep/25B4244-cv-bootcamp](https://github.com/tanishjain0911-beep/25B4244-cv-bootcamp). On GitHub, open the **Code** tab to see the repository files. Select a folder such as `code`, `data`, `predictions`, or `artifacts` to browse its contents; select a file to view it. Use **Go to file** to jump directly to a filename, or click **README.md** to return to this overview.

**Repository browsing:** the `main` branch is the canonical view of the extracted submission files. Use the **Code** tab to inspect the current tree and the **Releases** section for any downloadable ZIP archive; a release archive is separate from the files committed to `main`.

## Validation

The submission folder passes the supplied `check_format.py` validation for all five tasks, with completeness 1.00 for each. The report answers remain within the 100-word limit. Re-run the checker against the complete extracted submission folder and the final ZIP after packaging.
