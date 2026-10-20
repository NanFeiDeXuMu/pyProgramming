"""End-to-end smoke test of the notebook training and export path on tiny data.

Executes assignment_model.py and final_training.py the way the notebook does
(as top-level code), trains a three-member ensemble covering preprocessing,
a shared dictionary cache, and smoothing, then checks predictions after loading
the exported file in an isolated interpreter.
"""
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '6'
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
exec(compile((root / 'assignment_model.py').read_text(encoding='utf-8'), 'embedded_model', 'exec'))
exec(compile((root / 'final_training.py').read_text(encoding='utf-8'), 'embedded_training', 'exec'))
small = dict(clusters=8, patch_size=6, n_patches=128, augmentation_ratio=1, hidden_layers=[8],
             alpha=.1, learning_rate=.001, epochs=1)
configuration = dict(members=[dict(small, smooth_sigma=0, preprocess='median3'),
                              dict(small, smooth_sigma=0, preprocess='median3'),
                              dict(small, smooth_sigma=.6)],
                     weights=[1, 1, 1], horizontal_tta=True)
images = np.random.default_rng(42).integers(0, 256, (30, 32, 32, 3), dtype=np.uint8)
model, timing = fit_final_seed(images, np.arange(30) % 10, 42, configuration)
with tempfile.TemporaryDirectory() as directory:
    directory = Path(directory)
    save_portable(model, directory / 'model.joblib')
    np.save(directory / 'images.npy', images)
    np.save(directory / 'expected.npy', model.predict(images))
    code = ("import joblib,numpy as np; m=joblib.load('model.joblib'); "
            "np.testing.assert_array_equal(m.predict(np.load('images.npy')),np.load('expected.npy'))")
    subprocess.run([sys.executable, '-I', '-c', code], cwd=directory, check=True)
print('Final training and notebook-style export smoke passed')
