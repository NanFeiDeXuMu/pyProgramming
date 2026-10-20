"""Development members that see a fixed image preprocessing before all features.

Each preprocessing is applied to dictionary-learning images, training images,
augmented images (after corruption, as at test time) and development images,
exactly as the 'smooth' member applies Gaussian smoothing. All other settings
match the 'raw' member: K=512, 6x6 patches, robust augmentation at ratio 2.0,
MLP (384, 128), alpha 5, learning rate 5e-4. Audit labels are never scored.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import time
import joblib
import numpy as np
from scipy.ndimage import median_filter
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from assignment_model import (learn_dictionary, image_features, augment_images,
                              flush_tiny_training_values, preprocess_images)
from run_experiments import data_and_splits, record, CACHE


def preprocess(images, mode, batch_size=5000):
    """Deterministic per-image transform returning float32 (N, 32, 32, 3).

    Adopted modes delegate to the frozen preprocess_images; median5 and
    autocontrast were development-only candidates (rejected).
    """
    out = np.empty(images.shape, np.float32)
    for start in range(0, len(images), batch_size):
        x = np.asarray(images[start:start + batch_size], dtype=np.float32)
        if mode == 'median5':
            x = median_filter(x, size=(1, 5, 5, 1))
        elif mode == 'autocontrast':
            low = np.percentile(x, 1, axis=(1, 2, 3), keepdims=True)
            high = np.percentile(x, 99, axis=(1, 2, 3), keepdims=True)
            x = np.clip((x - low) * (255 / np.maximum(high - low, 1)), 0, 255).astype(np.float32)
        else:
            x = preprocess_images(x, mode)
        out[start:start + len(x)] = x
    return out


def cached(path, compute):
    if not path.exists():
        started = time.perf_counter()
        np.save(path, compute())
        print(f'{path.name}: {time.perf_counter() - started:.1f}s', flush=True)
    return np.load(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preprocess', choices=['median3', 'gray', 'lowres', 'median5', 'autocontrast'], required=True)
    p.add_argument('--epochs', type=int, default=15)
    a = p.parse_args()
    x, y, v, z, train, clean, dev = data_and_splits()
    name = f'pre_{a.preprocess}_patch512_robust_ratio2.0_alpha5.0'
    bank_path = CACHE / f'pre_{a.preprocess}_patch512_bank.joblib'
    if not bank_path.exists():
        joblib.dump([learn_dictionary(preprocess(x[train], a.preprocess), 42, n_clusters=512)], bank_path)
    banks = joblib.load(bank_path)
    features = lambda images: image_features(preprocess(images, a.preprocess), banks=banks)
    stem = f'pre_{a.preprocess}_patch512'
    ft = cached(CACHE / f'{stem}_trainsplit.npy', lambda: features(x[train]))
    fc = cached(CACHE / f'{stem}_clean.npy', lambda: features(x[clean]))
    fd = cached(CACHE / f'{stem}_dev.npy', lambda: features(v[dev]))
    fdf = cached(CACHE / f'{stem}_dev_flip.npy', lambda: features(v[dev, :, ::-1]))
    chosen = np.random.default_rng(42).choice(train, size=int(len(train) * 2.0), replace=True)
    af = cached(CACHE / f'{stem}_robust_ratio2.0.npy',
                lambda: features(augment_images(x[chosen], 42, 'robust')))
    scaler = StandardScaler()
    train_features = scaler.fit_transform(np.concatenate([ft, af]))
    targets = np.concatenate([y[train], y[chosen]])
    del ft, af
    fc, fd, fdf = scaler.transform(fc), scaler.transform(fd), scaler.transform(fdf)
    model = MLPClassifier(hidden_layer_sizes=(384, 128), alpha=5.0, learning_rate_init=.0005,
                          batch_size=512, max_iter=1, random_state=42, shuffle=True)
    best_score, best_epoch, epoch_probabilities = -1, 0, {}
    started = time.perf_counter()
    for epoch in range(1, a.epochs + 1):
        with threadpool_limits(limits=6):
            model.partial_fit(train_features, targets, classes=np.arange(10))
            flush_tiny_training_values(model)
            proba = model.predict_proba(fd)
            tta = (proba + model.predict_proba(fdf)) / 2
            pc = model.predict(fc)
        epoch_probabilities[f'epoch{epoch}'] = tta
        score = float(np.mean(proba.argmax(1) == z[dev]))
        row = dict(experiment=name, classifier='mlp384', epoch=epoch,
                   clean_accuracy=float(np.mean(pc == y[clean])), dev_accuracy=score,
                   dev_tta_accuracy=float(np.mean(tta.argmax(1) == z[dev])),
                   loss=float(model.loss_), elapsed_seconds=round(time.perf_counter() - started, 2))
        record(row)
        if score > best_score:
            best_score, best_epoch = score, epoch
            joblib.dump(dict(scaler=scaler, model=model, row=row, banks=banks,
                             preprocess=a.preprocess), CACHE / f'{name}_mlp384.joblib')
        if epoch - best_epoch >= 7:
            break
    np.savez(CACHE / f'{name}_epoch_tta_probabilities.npz', best_epoch=best_epoch, **epoch_probabilities)


if __name__ == '__main__':
    main()
