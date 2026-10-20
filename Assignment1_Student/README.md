# Assignment 1 execution notes

Student: **LIU, Yixuan (50027321)**.

The original assignment requirements and evaluation notebook are preserved.
Course data is extracted into `student_data/` and excluded from Git/submission.

## Results and brief analysis

Final training (v2) used all 50,000 course training images and independently
rebuilt a frozen seven-member sklearn MLP ensemble for each required seed. Each
member combines learned local-patch features with HOG and color statistics.
Four members differ in patch size, dictionary size, smoothing, and augmentation
strength. Three see a fixed preprocessed image before all features: 3x3 median,
grayscale, or 2x2 low resolution. Inference averages probabilities over members
and horizontal flips. No external data or pretrained models were used.

| Seed | Full validation (5,000) | Reserved audit (1,500) | Model size (decimal MB) |
| --- | ---: | ---: | ---: |
| 42 | 66.94% | 67.27% | 43.20 |
| 2026 | 67.34% | 68.00% | 43.22 |
| 3102 | 66.72% | 66.93% | 43.34 |
| Mean | **67.00%** | **67.40%** | — |

The previous four-member submission (v1) scored 65.46% on full validation
and 66.13% on audit. The audit subset was never used for any selection, so
the +1.27-point audit gain is the independent estimate of the improvement.
The full validation set includes the development subset and is optimistic.
v1 artifacts are kept in `submission_v1_backup/`.

Seven-member Run All took 2,953 seconds with three concurrent seed workers on
this machine. The official `Model_Evaluation.ipynb` code passed every cell in a
separate directory and fresh kernel and reproduced all three scores and the
67.00% mean (`Model_Evaluation_executed.ipynb`). Nine behavioral tests passed.
Frozen-source/data hashes, seed metadata for all seven members, ZIP contents,
and ZIP integrity passed (`submission_verification.json`). The ZIP is 129.8 MB.

## Development experiments after v1

All comparisons use the 3,500-image development subset. The primary
comparison was fixed before each run finished and scored with a paired
McNemar test. Retraining required a gain of about one point.

| Idea | Primary result (dev) | Decision |
| --- | --- | --- |
| Broader `robust3` augmentation in raw, largepatch, smooth (`augment_v3.py`) | 65.74% vs 65.80%, p = 0.95 | Rejected |
| Median-filter test-time views (`tta_experiments.py`) | 65.23% vs 65.80%, p = 0.24 | Rejected |
| Add median3, gray, lowres members (`preprocess_members.py`) | **66.77% vs 65.80%, p = 0.014** | **Adopted (v2)** |
| Train on validation images, 2-fold out-of-fold (`validation_training.py`) | raw 61.69% vs 61.91% (repeat 10); largepatch 61.51% vs 63.46%, p = 0.004 | Rejected |
| Logistic stacker on member validation probabilities (`stacking_experiment.py`) | 64.03% vs 66.77%, p < 0.001 | Rejected |
| Add median5 and autocontrast members (`compare_extra_members.py`) | 66.83% vs 66.77%, p = 0.91 | Rejected |
| Higher-capacity raw member, K=1600 or stride 1 (`capacity_experiments.py`) | K=1600 replacing raw: 66.80% vs 66.77%, p = 1.0 | Rejected |
| Other classifier family as eighth member, on PCA-256 of raw features (`family_experiments.py`) | HistGradientBoosting: 66.69% vs 66.77%, p = 0.83; ExtraTrees: 66.83%, p = 0.73 | Rejected |

The TA confirmed that validation images may be used for training. In
out-of-fold tests they did not help this model family: repeated validation
images were memorized rather than generalized. Individual robust3 members were
stronger, but more correlated with each other, so the ensemble did not improve.
Member diversity was the effective lever, and it saturated at seven members.
More feature capacity raised clean accuracy, up to 76.4% for stride 1 versus 75.1%,
but not accuracy on corrupted development images (K=1600: 60.51% versus 62.06%
with flip TTA). The extra detail it captures is mostly destroyed by corruption.

## Environment

The workspace-local interpreter is:

```powershell
..\.cache\assignment1-runtime\python\python.exe
```

It uses Python 3.14.6, NumPy 2.4.6, SciPy 1.18.0, scikit-learn 1.9.0,
and joblib 1.5.3. `requirements-final.txt` pins the model libraries. The final
notebook also requires Jupyter/IPython to execute; these are not model inference
dependencies. The existing Conda/Habitat environment has not been changed.

From the workspace root, execute all notebook cells and retain outputs with:

```powershell
& .\.cache\assignment1-runtime\python\python.exe Assignment1_Student\execute_notebook.py
```

The default trains one seed at a time. With sufficient available RAM, set
`$env:ASSIGNMENT1_N_JOBS='3'` before that command to train the three seeds in
independent processes. Each uses the same frozen recipe and its own seed.

## Files and reproducibility

- `assignment_model.py`: deterministic image features, training augmentation,
  portable export, and probability averaging.
- `run_experiments.py`, `advanced_experiments.py`: development experiments.
- `preprocess_members.py`, `augment_v3.py`, `tta_experiments.py`, `validation_training.py`,
  `stacking_experiment.py`, `capacity_experiments.py`, `family_experiments.py`, and
  `compare_*.py`: post-v1 development experiments (dev subset only; not used by
  the notebook). Their feature caches in `.cache/` were deleted after submission;
  rerunning a comparison first regenerates its inputs, and measured results stay
  in `experiment_results.jsonl`.
- `experiment_results.jsonl`: actual experiment measurements, appended as runs finish.
- `data_audit.json`: exact-image duplicate and overlap checks.
- `test_assignment.py`: behavioral tests, including independent-process loading.
- `smoke_final_training.py`: tiny end-to-end run of the notebook training and export path.
- `final_config.json`: frozen final recipe (created after development selection).
- `final_training.py`, `build_notebook.py`: build the self-contained training notebook.
- `execute_notebook.py`: run every cell in a new kernel and retain its real output.
- `verify_submission.py`: run the unmodified official evaluation code in a separate
  directory/kernel, then package exactly the four required files.

The final `train_models.ipynb` embeds all necessary model and training code; it
does not import these development scripts or load experiment caches. Run it with
the extracted course dataset and the exact environment. Each required seed starts
from scratch. Identical deterministic feature recipes can share an in-memory
cache within one seed, but nothing is reused between seeds or notebook runs. Each
cached feature matrix is freed after the last member that needs it, which bounds
peak memory.

Development uses stratified splits with `random_state=1729`: 45,000/5,000 original
training images and 3,500/1,500 official development/audit images. Augmentation is
performed only after splitting original images. No official validation images fit
feature statistics, dictionaries, classifiers, or calibration. Audit scores are
reported only after the recipe is locked. Final models train on all 50,000 course
training images. The three reported validation accuracies are averaged, matching
the instructor's evaluation; predictions are not averaged across seed files.

Two LinearSVC trials were stopped for excessive runtime without a completed score.
The smoothing search was bounded after its late epochs became unusually slow;
its best completed checkpoint was preserved. These interrupted runs are not
presented as completed measurements.

Later MLP epochs accumulated float32 subnormal values. Between epochs the final
recipe rounds weights and Adam state below `1e-20` to zero to avoid this CPU
slowdown. This numerical cleanup is applied consistently to all seeds and checked
against the development predictions. Optimizer moments and unused early-stopping
backups are removed from final inference files; prediction equality is asserted.

The local notebook runner uses an explicit successful exit after saving and
closing the kernel to avoid a native handle cleanup failure observed with this
embedded Windows Python/Jupyter runtime. Execution exceptions are not suppressed.
This runner is not needed by or included in the submission.

## Submission

The final ZIP must contain only `train_models.ipynb`, `model_seed42.joblib`,
`model_seed2026.joblib`, and `model_seed3102.joblib`. The verification script uses
the filename `LIU, Yixuan-50027321-AIAA3102-Assignment1.zip`. Dataset, development
scripts, caches, and reports are excluded. Upload to Canvas is a separate action.
