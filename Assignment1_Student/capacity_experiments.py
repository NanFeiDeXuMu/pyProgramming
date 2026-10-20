"""Higher-capacity patch features for the 'raw' member (dev split only).

Variants: more dictionary atoms (--clusters 1600) or denser encoding
(--stride 1). Everything else matches 'raw': 6x6 patches, robust augmentation
at ratio 2.0, MLP (384, 128), alpha 5, learning rate 5e-4. Per-epoch flip-TTA
development probabilities are saved for ensemble comparison.
Audit labels are never scored.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import time
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from assignment_model import learn_dictionary, image_features, augment_images, flush_tiny_training_values
from run_experiments import data_and_splits, record, CACHE
from preprocess_members import cached


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clusters', type=int, default=512)
    p.add_argument('--stride', type=int, default=2)
    p.add_argument('--epochs', type=int, default=15)
    a = p.parse_args()
    x, y, v, z, train, clean, dev = data_and_splits()
    stem = f'cap_K{a.clusters}_s{a.stride}'
    name = f'{stem}_robust_ratio2.0_alpha5.0'
    bank_path = CACHE / f'{stem}_bank.joblib'
    if not bank_path.exists():
        joblib.dump([learn_dictionary(x[train], 42, n_clusters=a.clusters)], bank_path)
    banks = joblib.load(bank_path)
    features = lambda images: image_features(images, banks=banks, stride=a.stride)
    ft = cached(CACHE / f'{stem}_trainsplit.npy', lambda: features(x[train]))
    fc = cached(CACHE / f'{stem}_clean.npy', lambda: features(x[clean]))
    fd = cached(CACHE / f'{stem}_dev.npy', lambda: features(v[dev]))
    fdf = cached(CACHE / f'{stem}_dev_flip.npy', lambda: features(v[dev, :, ::-1]))
    chosen = np.random.default_rng(42).choice(train, size=int(len(train) * 2.0), replace=True)
    af = cached(CACHE / f'{stem}_robust_ratio2.0.npy',
                lambda: features(augment_images(x[chosen], 42, 'robust')))
    train_features = np.concatenate([ft, af])
    del ft, af
    scaler = StandardScaler(copy=False)
    train_features = scaler.fit_transform(train_features)
    targets = np.concatenate([y[train], y[chosen]])
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
        record(dict(experiment=name, classifier='mlp384', epoch=epoch,
                    clean_accuracy=float(np.mean(pc == y[clean])), dev_accuracy=score,
                    dev_tta_accuracy=float(np.mean(tta.argmax(1) == z[dev])),
                    loss=float(model.loss_), elapsed_seconds=round(time.perf_counter() - started, 2)))
        if score > best_score:
            best_score, best_epoch = score, epoch
        if epoch - best_epoch >= 7:
            break
    np.savez(CACHE / f'{name}_epoch_tta_probabilities.npz', best_epoch=best_epoch, **epoch_probabilities)


if __name__ == '__main__':
    main()
