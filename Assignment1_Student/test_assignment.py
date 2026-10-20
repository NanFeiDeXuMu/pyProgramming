"""Behavioral tests including load in an isolated interpreter and directory."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import subprocess
import tempfile
import unittest
import numpy as np
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from assignment_model import (image_features, learn_dictionary, augment_images,
                              make_pipeline, save_portable, ImageClassifier, strip_training_state,
                              flush_tiny_training_values)


class AssignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = np.random.default_rng(42).integers(0, 256, (30, 32, 32, 3), dtype=np.uint8)
        cls.bank = learn_dictionary(cls.x, 42, n_clusters=8, n_patches=128)

    def test_shape_finiteness_and_batch_independence(self):
        args = dict(banks=[self.bank], include_hog=True)
        a = image_features(self.x[:3], batch_size=1, **args)
        b = image_features(self.x[:3], batch_size=3, **args)
        self.assertEqual(a.shape, (3, 1892))
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-5)
        self.assertTrue(np.isfinite(a).all())
        self.assertTrue(np.isfinite(image_features(np.zeros_like(self.x[:2]), **args)).all())
        self.assertEqual(image_features(self.x[:0], **args).shape, (0, 1892))

    def test_preprocessing_banks_are_batch_independent_and_portable(self):
        for mode in ('median3', 'gray', 'lowres'):
            bank = learn_dictionary(self.x, 42, n_clusters=8, n_patches=128, preprocess=mode)
            a = image_features(self.x[:4], banks=[bank], batch_size=1)
            b = image_features(self.x[:4], banks=[bank], batch_size=4)
            self.assertEqual(a.shape, (4, 1892))
            np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-5)
            self.assertFalse(np.allclose(a, image_features(self.x[:4], banks=[self.bank])))
        member = make_pipeline([bank], LogisticRegression(C=.1, max_iter=100))
        member.fit(self.x, np.arange(len(self.x)) % 10)
        model = ImageClassifier([member], horizontal_tta=True)
        root = Path(__file__).resolve().parent / '.cache'
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path = Path(directory)
            save_portable(model, path / 'model.joblib')
            np.save(path / 'images.npy', self.x[:5])
            code = "import joblib,numpy as np; m=joblib.load('model.joblib'); np.save('pred.npy',m.predict_proba(np.load('images.npy')))"
            subprocess.run([sys.executable, '-I', '-c', code], cwd=path,
                           check=True, capture_output=True, text=True)
            np.testing.assert_array_equal(model.predict_proba(self.x[:5]), np.load(path / 'pred.npy'))

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            image_features(np.zeros((3, 3072)))

    def test_augmentation_reproducible_without_mutation(self):
        original = self.x.copy()
        a = augment_images(self.x, 42)
        np.testing.assert_array_equal(a, augment_images(self.x, 42))
        np.testing.assert_array_equal(self.x, original)
        self.assertFalse(np.array_equal(a, augment_images(self.x, 43)))
        self.assertEqual(a.dtype, np.uint8)

    def test_robust3_augmentation_reproducible_and_valid(self):
        original = self.x.copy()
        from augment_v3 import augment_robust3
        a = augment_robust3(self.x, 42)
        np.testing.assert_array_equal(a, augment_robust3(self.x, 42))
        np.testing.assert_array_equal(self.x, original)
        self.assertEqual((a.shape, a.dtype), (self.x.shape, np.uint8))
        self.assertFalse(np.array_equal(a, augment_robust3(self.x, 43)))

    def test_dictionary_seed_reproducibility(self):
        bank = learn_dictionary(self.x, 42, n_clusters=8, n_patches=128)
        np.testing.assert_allclose(bank['centers'], self.bank['centers'])

    def test_portable_model_in_clean_subprocess(self):
        member = make_pipeline([self.bank], LogisticRegression(C=.1, max_iter=100))
        member.fit(self.x, np.arange(len(self.x)) % 10)
        model = ImageClassifier([member], horizontal_tta=True)
        expected = model.predict(self.x[:5])
        root = Path(__file__).resolve().parent / '.cache'
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path = Path(directory)
            save_portable(model, path / 'model.joblib')
            np.save(path / 'images.npy', self.x[:5])
            code = "import joblib,numpy as np; m=joblib.load('model.joblib'); np.save('pred.npy',m.predict(np.load('images.npy')))"
            subprocess.run([sys.executable, '-I', '-c', code], cwd=path,
                           check=True, capture_output=True, text=True)
            np.testing.assert_array_equal(expected, np.load(path / 'pred.npy'))

    def test_optimizer_removal_preserves_probabilities(self):
        model = MLPClassifier(hidden_layer_sizes=(8,), random_state=42)
        features = self.x.reshape(len(self.x), -1).astype(np.float32) / 255
        model.partial_fit(features, np.arange(len(features)) % 10, classes=np.arange(10))
        before = model.predict_proba(features)
        strip_training_state(model)
        self.assertFalse(hasattr(model, '_optimizer'))
        np.testing.assert_array_equal(before, model.predict_proba(features))

    def test_tiny_value_cleanup_preserves_probabilities(self):
        model = MLPClassifier(hidden_layer_sizes=(8,), random_state=42)
        features = self.x.reshape(len(self.x), -1).astype(np.float32) / 255
        model.partial_fit(features, np.arange(len(features)) % 10, classes=np.arange(10))
        model.coefs_[0][0, 0] = np.float32(1e-40)
        model._optimizer.ms[0][0, 0] = np.float32(1e-40)
        before = model.predict_proba(features)
        flush_tiny_training_values(model)
        self.assertEqual(model.coefs_[0][0, 0], 0)
        self.assertEqual(model._optimizer.ms[0][0, 0], 0)
        np.testing.assert_array_equal(before, model.predict_proba(features))


if __name__ == '__main__':
    unittest.main(verbosity=2)
