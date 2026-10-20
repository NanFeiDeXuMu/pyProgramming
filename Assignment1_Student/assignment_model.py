"""Self-contained CIFAR feature learning; no external data or model weights.

This source is embedded verbatim in the final notebook. Inference depends only on
NumPy, SciPy, scikit-learn and joblib. Training data and caches are never needed
by the exported model.
"""
import time
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from threadpoolctl import threadpool_limits


def preprocess_images(images, mode):
    """Fixed per-image view seen by a member before all features (float32 output).

    'median3' is a 3x3 spatial median, 'gray' replicates luminance to three
    channels, and 'lowres' averages 2x2 blocks then repeats them. Each view makes
    one ensemble member insensitive to a different corruption family.
    """
    import numpy as np
    from scipy.ndimage import median_filter
    x = np.asarray(images, dtype=np.float32)
    if mode == 'median3':
        return median_filter(x, size=(1, 3, 3, 1))
    if mode == 'gray':
        return np.repeat((x @ np.float32([.299, .587, .114]))[..., None], 3, axis=3)
    if mode == 'lowres':
        x = x.reshape(len(x), 16, 2, 16, 2, 3).mean(axis=(2, 4))
        return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)
    raise ValueError(f'Unknown preprocessing: {mode}')


def learn_dictionary(images, seed, n_clusters=512, patch_size=6, n_patches=60000, smooth_sigma=0,
                     preprocess=None):
    """Fit local contrast normalization, regularized ZCA, and a patch dictionary."""
    rng = np.random.default_rng(seed)
    if preprocess:
        images = preprocess_images(images, preprocess)
    if smooth_sigma:
        images = gaussian_filter(images.astype(np.float32), sigma=(0, smooth_sigma, smooth_sigma, 0))
    count = len(images)
    indices = rng.integers(count, size=n_patches)
    rows = rng.integers(33 - patch_size, size=n_patches)
    cols = rng.integers(33 - patch_size, size=n_patches)
    patches = np.empty((n_patches, patch_size * patch_size * 3), dtype=np.float32)
    for i, (index, row, col) in enumerate(zip(indices, rows, cols)):
        patch = images[index, row:row + patch_size, col:col + patch_size]
        if rng.random() < .5:
            patch = patch[:, ::-1]
        patches[i] = patch.reshape(-1)
    patches -= patches.mean(axis=1, keepdims=True)
    patches /= np.sqrt(patches.var(axis=1, keepdims=True) + 10.0)
    mean = patches.mean(axis=0)
    patches -= mean
    cov = (patches.T @ patches) / len(patches)
    values, vectors = np.linalg.eigh(cov)
    whitening = ((vectors * (1.0 / np.sqrt(np.maximum(values, 0) + .1))) @ vectors.T).astype(np.float32)
    patches = np.ascontiguousarray(patches @ whitening)
    km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=2048,
                        n_init=3, max_iter=80, max_no_improvement=15,
                        random_state=seed, reassignment_ratio=.01)
    started = time.perf_counter()
    with threadpool_limits(limits=6):
        km.fit(patches)
    print(f'Dictionary K={n_clusters}, p={patch_size}: {time.perf_counter()-started:.1f}s', flush=True)
    return dict(centers=km.cluster_centers_.astype(np.float32), mean=mean,
                whitening=whitening, patch_size=patch_size, smooth_sigma=smooth_sigma,
                preprocess=preprocess)


def image_features(images, banks=(), include_hog=True, stride=2, batch_size=48):
    """Deterministic per-image transform, with bounded temporary allocations.

    All imports and calculations needed by prediction are inside this function.
    The HOG implementation uses unsigned, linearly interpolated orientation bins
    and L2-Hys block normalization. Learned features use triangle coding and
    quadrant average pooling, followed by a square-root transform.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter
    from threadpoolctl import threadpool_limits
    x = np.asarray(images)
    if x.ndim != 4 or x.shape[1:] != (32, 32, 3):
        raise ValueError('Expected images with shape (N, 32, 32, 3).')
    if len(x) == 0:
        dimension = sum(4 * len(bank['centers']) for bank in banks) + 96
        if include_hog:
            dimension += 1764
        return np.empty((0, dimension), dtype=np.float32)
    result = []
    with threadpool_limits(limits=6):
        for start in range(0, len(x), batch_size):
            batch = x[start:start + batch_size].astype(np.float32)
            mode = banks[0].get('preprocess') if banks else None
            if mode:
                batch = preprocess_images(batch, mode)
            sigma = banks[0].get('smooth_sigma', 0) if banks else 0
            if sigma:
                batch = gaussian_filter(batch, sigma=(0, sigma, sigma, 0))
            n = len(batch)
            parts = []
            for bank in banks:
                p = bank['patch_size']
                view = np.lib.stride_tricks.sliding_window_view(batch, (p, p), axis=(1, 2))
                view = view[:, ::stride, ::stride].transpose(0, 1, 2, 4, 5, 3)
                h, w = view.shape[1:3]
                patches = np.ascontiguousarray(view).reshape(-1, p * p * 3)
                patches -= patches.mean(axis=1, keepdims=True)
                patches /= np.sqrt(patches.var(axis=1, keepdims=True) + 10.0)
                patches = (patches - bank['mean']) @ bank['whitening']
                centers = bank['centers']
                distances = patches @ centers.T
                distances *= -2
                distances += np.sum(patches * patches, axis=1, keepdims=True)
                distances += np.sum(centers * centers, axis=1)
                np.maximum(distances, 0, out=distances)
                np.sqrt(distances, out=distances)
                activation = np.maximum(distances.mean(axis=1, keepdims=True) - distances, 0)
                activation = activation.reshape(n, h, w, -1)
                quadrants = [activation[:, a:b, c:d].mean(axis=(1, 2))
                             for a, b in ((0, h//2), (h//2, h))
                             for c, d in ((0, w//2), (w//2, w))]
                parts.append(np.sqrt(np.concatenate(quadrants, axis=1)))
            # Spatial color means and standard deviations, 4x4 regions.
            color = batch.reshape(n, 4, 8, 4, 8, 3)
            parts.extend([color.mean(axis=(2, 4)).reshape(n, -1) / 255,
                          color.std(axis=(2, 4)).reshape(n, -1) / 255])
            if include_hog:
                # Choose the strongest color-channel gradient at each pixel.
                dx = np.zeros_like(batch)
                dy = np.zeros_like(batch)
                dx[:, :, 1:-1] = batch[:, :, 2:] - batch[:, :, :-2]
                dy[:, 1:-1] = batch[:, 2:] - batch[:, :-2]
                mag2 = dx*dx + dy*dy
                ch = mag2.argmax(axis=3)[..., None]
                gx = np.take_along_axis(dx, ch, axis=3)[..., 0]
                gy = np.take_along_axis(dy, ch, axis=3)[..., 0]
                mag = np.sqrt(np.take_along_axis(mag2, ch, axis=3)[..., 0])
                angle = (np.arctan2(gy, gx) % np.pi) * (9 / np.pi)
                lo = np.floor(angle).astype(np.int32)
                frac = angle - lo
                hist = np.empty((n, 8, 8, 9), np.float32)
                for k in range(9):
                    votes = mag * ((lo % 9 == k) * (1-frac) + ((lo+1) % 9 == k) * frac)
                    hist[..., k] = votes.reshape(n, 8, 4, 8, 4).sum(axis=(2, 4))
                blocks = np.lib.stride_tricks.sliding_window_view(hist, (2, 2), axis=(1, 2))
                blocks = blocks.transpose(0, 1, 2, 4, 5, 3).reshape(n, 7, 7, 36)
                blocks = blocks / np.sqrt(np.sum(blocks*blocks, axis=3, keepdims=True) + 1e-6)
                blocks = np.minimum(blocks, .2)
                blocks /= np.sqrt(np.sum(blocks*blocks, axis=3, keepdims=True) + 1e-6)
                parts.append(blocks.reshape(n, -1))
            result.append(np.concatenate(parts, axis=1).astype(np.float32))
    return np.concatenate(result)


def augment_images(images, seed, mode='mixed'):
    """Seeded, label-preserving augmentation of training images only."""
    rng = np.random.default_rng(seed)
    if mode == 'flip':
        return np.ascontiguousarray(images[:, :, ::-1])
    x = images.astype(np.float32).copy()
    flips = rng.random(len(x)) < .5
    x[flips] = x[flips, :, ::-1]
    for i in range(len(x)):
        dy, dx = rng.integers(-2, 3, size=2)
        padded = np.pad(x[i], ((2, 2), (2, 2), (0, 0)), mode='reflect')
        x[i] = padded[2+dy:34+dy, 2+dx:34+dx]
    if mode in ('robust', 'robust2'):
        strong = mode == 'robust2'
        for i in range(len(x)):
            kind = rng.integers(0, 7)
            if kind == 1:
                x[i] += rng.normal(0, rng.uniform(4, 40 if strong else 24), size=x[i].shape)
            elif kind == 2:
                x[i] = gaussian_filter(x[i], sigma=(rng.uniform(.4, 1.8 if strong else 1.15),) * 2 + (0,))
            elif kind == 3:
                x[i] *= rng.uniform(.45 if strong else .65, 1.55 if strong else 1.35, size=(1, 1, 3))
                x[i] += rng.uniform(-20, 20)
            elif kind == 4:
                mask = rng.random((32, 32)) < rng.uniform(.01, .10 if strong else .05)
                x[i][mask] = rng.choice([0, 255], size=(mask.sum(), 1))
            elif kind == 5:
                reduced = x[i].reshape(16, 2, 16, 2, 3).mean(axis=(1, 3))
                x[i] = np.repeat(np.repeat(reduced, 2, axis=0), 2, axis=1)
            elif kind == 6:
                size = int(rng.integers(4, 9))
                r, c = rng.integers(0, 33-size, size=2)
                x[i, r:r+size, c:c+size] = rng.integers(0, 256, size=(1, 1, 3))
        return np.clip(x, 0, 255).astype(np.uint8)
    # Keep several types separate; avoid always compounding all corruptions.
    kind = rng.integers(0, 4, size=len(x))
    for k in range(4):
        idx = np.flatnonzero(kind == k)
        if k == 1:
            x[idx] += rng.normal(0, 6, size=x[idx].shape).astype(np.float32)
        elif k == 2:
            x[idx] = gaussian_filter(x[idx], sigma=(0, .55, .55, 0))
        elif k == 3:
            x[idx] = x[idx] * rng.uniform(.8, 1.2, size=(len(idx), 1, 1, 1))
            x[idx] += rng.uniform(-12, 12, size=(len(idx), 1, 1, 1))
    return np.clip(x, 0, 255).astype(np.uint8)


def make_pipeline(banks, classifier, scaler=None, include_hog=True):
    transform = FunctionTransformer(image_features,
        kw_args=dict(banks=banks, include_hog=include_hog), validate=False)
    if scaler is None:
        scaler = StandardScaler()
    return Pipeline([('features', transform), ('scale', scaler), ('classifier', classifier)])


def strip_training_state(classifier):
    """Drop Adam moments and unused early-stopping backups, retaining all weights.

    These attributes are only used for continuing training, never prediction.
    The submission is an inference artifact; Run All always trains from scratch.
    Prediction equality before/after stripping is tested before every export.
    """
    for attribute in ('_optimizer', '_best_coefs', '_best_intercepts'):
        if hasattr(classifier, attribute):
            delattr(classifier, attribute)
    return classifier


def flush_tiny_training_values(classifier, threshold=1e-20):
    """Round numerically negligible float32 values to zero between epochs.

    Dead ReLU units plus Adam/regularization can generate CPU-slow subnormals.
    The threshold is far below float32 prediction precision. Apply identically
    for every seed, and test prediction equality for the developed candidates.
    """
    arrays = list(classifier.coefs_) + list(classifier.intercepts_)
    optimizer = getattr(classifier, '_optimizer', None)
    if optimizer is not None:
        arrays += list(getattr(optimizer, 'ms', [])) + list(getattr(optimizer, 'vs', []))
    for array in arrays:
        array[np.abs(array) < threshold] = 0
    return classifier


def save_portable(model, path):
    """Embed custom feature code, without requiring assignment_model.py at load."""
    import sys
    import joblib
    from joblib.externals import cloudpickle
    from joblib import wrap_non_picklable_objects
    module = sys.modules.get(image_features.__module__)
    registered = module is not None and module.__name__ != '__main__'
    if registered:
        cloudpickle.register_pickle_by_value(module)
    try:
        joblib.dump(wrap_non_picklable_objects(model), path, compress=3, protocol=5)
    finally:
        if registered:
            cloudpickle.unregister_pickle_by_value(module)


class ImageClassifier(ClassifierMixin, BaseEstimator):
    """Average fitted sklearn probability pipelines and optional horizontal views.

    Members are independently fitted with the frozen training recipe before this
    inference-only container is constructed. No model is fitted at prediction.
    """
    def __init__(self, members, weights=None, horizontal_tta=False):
        self.members = members
        self.weights = weights
        self.horizontal_tta = horizontal_tta
        self.classes_ = np.asarray(members[0].classes_)
        for member in members:
            if not np.array_equal(member.classes_, self.classes_):
                raise ValueError('All members must have identical class ordering.')

    def predict_proba(self, images):
        weights = np.ones(len(self.members)) if self.weights is None else np.asarray(self.weights)
        if len(weights) != len(self.members) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError('Invalid ensemble weights.')
        result = None
        for weight, model in zip(weights / weights.sum(), self.members):
            probability = model.predict_proba(images)
            if self.horizontal_tta:
                probability = (probability + model.predict_proba(np.asarray(images)[:, :, ::-1])) * .5
            result = weight * probability if result is None else result + weight * probability
        return result

    def predict(self, images):
        return self.classes_[self.predict_proba(images).argmax(axis=1)]
