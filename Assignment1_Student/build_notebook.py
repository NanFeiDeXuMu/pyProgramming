"""Build a standalone submission notebook from reviewed training source."""
import json
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
configuration = json.loads((ROOT / 'final_config.json').read_text(encoding='utf-8'))
nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = dict(display_name='Python 3', language='python', name='python3')
nb.metadata.language_info = dict(name='python', version='3.14.6')
cells = []
cells.append(nbf.v4.new_markdown_cell('''# AIAA 3102 Assignment 1 — CIFAR-10 classification

**Student:** LIU, Yixuan — **Student ID:** 50027321

## Final method and rationale

We learn local patch dictionaries exclusively from the provided training images.
Local contrast normalization and regularized ZCA whitening precede MiniBatchKMeans;
triangle coding and quadrant pooling preserve local image structure. HOG and
spatial color statistics add complementary information. Scikit-learn MLP classifiers
learn from clean images and seeded, label-preserving corruption augmentations.
The frozen configuration below specifies all model members, training epochs,
ensemble weights, and whether horizontal-flip prediction averaging is enabled.

The ensemble has seven members. Four differ in patch size, dictionary size,
smoothing, and augmentation strength. Three see a fixed preprocessed image
before all features: a 3x3 median filter, grayscale, or 2x2 low resolution.
This diversity makes members fail on different corruptions. On the development
subset, the seven-member ensemble reached 66.77% versus 65.80% for the previous
four-member recipe (McNemar p = 0.014), while extra test-time views and a
broader augmentation did not help.

Development uses a fixed 45,000/5,000 split of the provided training set and a
3,500/1,500 development/audit split of the official validation set. The audit labels
are not used to choose the model. All final components are retrained from scratch
on the 50,000 training images for each required seed; official validation images
are used only for reporting. No external data, pretrained weights, or custom
classification algorithm is used. Feature learning is preprocessing; all supervised
learners are `sklearn.neural_network.MLPClassifier` instances.

To avoid CPU slowdowns from float32 subnormals, values below `1e-20` in weights
and optimizer state are rounded to zero between epochs. Development prediction
probabilities are unchanged by this cleanup. Training-only optimizer state is
omitted from exported files; prediction equality is checked before each export.

The patch-feature design is inspired by Coates, Lee and Ng (AISTATS 2011):
https://proceedings.mlr.press/v15/coates11a.html . Paper results are not our results.

## How to run

Use Python 3.14.6, NumPy 2.4.6, SciPy 1.18.0, scikit-learn 1.9.0 and joblib 1.5.3,
with a Jupyter kernel for that environment. Set `DATA_ROOT` below if needed. The
default expects `student_data/CIFAR10_Data` and
`student_data/validation_set_to_students` next to this notebook (or one folder up).
Training uses CPU BLAS, not CUDA. Approximately 16 GB free RAM is recommended. Training one seed at a time takes
several hours on a 6-thread CPU; actual time depends on the machine and is printed below.
Execution defaults to one seed at a time. On a machine with at least 32 GB free
RAM, set the environment variable `ASSIGNMENT1_N_JOBS=3` before launching Jupyter
to train the three independent seeds in parallel. This does not change the recipe.

**Run All rebuilds all three models from raw course data.** No experiment cache,
existing model file, local Python source file, internet connection, or user input
is required. The outputs are the three required `.joblib` files in the working
directory. Model prediction includes preprocessing and needs no training files.
Custom preprocessing code is embedded using joblib's cloudpickle wrapper; the
exported files are checked in a new interpreter without importing notebook code.
'''))
cells.append(nbf.v4.new_code_cell('''import os
os.environ['LOKY_MAX_CPU_COUNT'] = '6'
os.environ['OMP_NUM_THREADS'] = '6'
os.environ['OPENBLAS_NUM_THREADS'] = '6'
import sys, platform, json, time, hashlib, tempfile, subprocess
from pathlib import Path
import numpy as np
import scipy, sklearn, joblib
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
versions = {'python': platform.python_version(), 'numpy': np.__version__,
            'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__, 'joblib': joblib.__version__}
expected = {'python':'3.14.6','numpy':'2.4.6','scipy':'1.18.0','scikit-learn':'1.9.0','joblib':'1.5.3'}
print(json.dumps(versions, indent=2))
assert versions == expected, 'Use the course evaluation environment before exporting final models.'
DATA_ROOT = Path('student_data')  # EDIT HERE if your extracted data is elsewhere.
if not (DATA_ROOT / 'CIFAR10_Data/x_train.npy').exists():
    for candidate in [Path('../student_data'), Path('.')]:
        if (candidate / 'CIFAR10_Data/x_train.npy').exists():
            DATA_ROOT = candidate
            break
assert (DATA_ROOT / 'CIFAR10_Data/x_train.npy').exists(), 'Extract student_data.zip and configure DATA_ROOT.'
X_train = np.load(DATA_ROOT / 'CIFAR10_Data/x_train.npy')
y_train = np.load(DATA_ROOT / 'CIFAR10_Data/y_train.npy').ravel()
X_validation = np.load(DATA_ROOT / 'validation_set_to_students/images.npy')
y_validation = np.load(DATA_ROOT / 'validation_set_to_students/labels.npy').ravel()
assert X_train.shape == (50000,32,32,3) and y_train.shape == (50000,)
assert X_validation.shape == (5000,32,32,3) and y_validation.shape == (5000,)
assert np.array_equal(np.unique(y_train), np.arange(10))
assert X_train.dtype == np.uint8 and X_validation.dtype == np.uint8
print('Training:', X_train.shape, 'classes:', np.bincount(y_train))
print('Validation:', X_validation.shape, 'classes:', np.bincount(y_validation))
dev_indices, audit_indices = train_test_split(np.arange(5000), test_size=1500,
    stratify=y_validation, random_state=1729)
'''))
cells.append(nbf.v4.new_markdown_cell('## Self-contained preprocessing and export implementation'))
cells.append(nbf.v4.new_code_cell((ROOT / 'assignment_model.py').read_text(encoding='utf-8')))
cells.append(nbf.v4.new_markdown_cell('## Frozen training recipe\nAll three seeds use the same settings. Validation scores do not control final training.'))
cells.append(nbf.v4.new_code_cell('CONFIGURATION = ' + repr(configuration) + '\nprint(json.dumps(CONFIGURATION, indent=2))\nresults = []\n'))
cells.append(nbf.v4.new_code_cell((ROOT / 'final_training.py').read_text(encoding='utf-8')))
cells.append(nbf.v4.new_markdown_cell('## Train the three independent seeds from scratch'))
cells.append(nbf.v4.new_code_cell('''from joblib import Parallel, delayed
SEEDS = [42, 2026, 3102]
N_JOBS = int(os.environ.get('ASSIGNMENT1_N_JOBS', '1'))
assert N_JOBS in [1, 2, 3]
training_started = time.perf_counter()
print('Independent seed workers:', N_JOBS, flush=True)
with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix='training_temp_') as cache_directory:
    fitted_models = Parallel(n_jobs=N_JOBS, backend='loky', temp_folder=cache_directory)(
        delayed(fit_final_seed)(X_train, y_train, seed, CONFIGURATION) for seed in SEEDS)
training_wall_seconds = time.perf_counter() - training_started
print('All three seeds trained from scratch in', round(training_wall_seconds, 2), 'seconds')
'''))
for seed in (42, 2026, 3102):
    code = f'''seed = {seed}
started = time.perf_counter()
model, member_seconds = fitted_models.pop(0)
model_path = Path(f'model_seed{{seed}}.joblib')
save_portable(model, model_path)
file_bytes = model_path.stat().st_size
assert file_bytes < 100_000_000, 'Model exceeds the 100 MB limit.'
prediction = model.predict(X_validation)
assert prediction.shape == (5000,) and np.isin(prediction, np.arange(10)).all()
# The new process sees only the model and input images, not training definitions.
with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
    directory = Path(directory)
    np.save(directory / 'images.npy', X_validation[:37])
    check_code = ('import joblib,numpy as np; '
        + 'm=joblib.load(' + repr(str(model_path.resolve())) + '); '
        + "np.save('predictions.npy',m.predict(np.load('images.npy')))")
    subprocess.run([sys.executable, '-I', '-c', check_code], cwd=directory, check=True)
    np.testing.assert_array_equal(prediction[:37], np.load(directory / 'predictions.npy'))
row = {{'seed': seed, 'validation_accuracy': float(accuracy_score(y_validation, prediction)),
       'development_accuracy': float(accuracy_score(y_validation[dev_indices], prediction[dev_indices])),
       'audit_accuracy': float(accuracy_score(y_validation[audit_indices], prediction[audit_indices])),
       'model_bytes': file_bytes, 'member_seconds': member_seconds,
       'export_evaluation_seconds': round(time.perf_counter()-started, 2),
       'sha256': hashlib.sha256(model_path.read_bytes()).hexdigest()}}
results.append(row)
print(json.dumps(row, indent=2))
del model
_ = gc.collect()
'''
    cells.append(nbf.v4.new_markdown_cell(f'## Export and check seed {seed}'))
    cells.append(nbf.v4.new_code_cell(code))
cells.append(nbf.v4.new_markdown_cell('## Final validation summary\nThese are provided-validation results, not hidden-test results. The selection-development subset is optimistic; the reserved audit subset is the independent estimate. Final training uses all 50,000 training images.'))
cells.append(nbf.v4.new_code_cell('''summary = {'environment': versions, 'configuration': CONFIGURATION, 'models': results,
    'mean_validation_accuracy': float(np.mean([r['validation_accuracy'] for r in results])),
    'mean_audit_accuracy': float(np.mean([r['audit_accuracy'] for r in results])),
    'all_models_under_100MB': all(r['model_bytes'] < 100_000_000 for r in results),
    'fresh_process_loading_passed': True}
print(json.dumps(summary, indent=2))
Path('results_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
'''))
nb.cells = cells
nbf.validate(nb)
nbf.write(nb, ROOT / 'train_models.ipynb')
print('Created train_models.ipynb with', len(cells), 'cells.')
