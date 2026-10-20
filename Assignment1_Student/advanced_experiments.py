"""Nonlinear classifiers and corruption augmentation, with a fixed audit split."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import copy
import time
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from threadpoolctl import threadpool_limits
from assignment_model import augment_images, flush_tiny_training_values
from augment_v3 import augment_robust3
from run_experiments import data_and_splits, features_cached, evaluate, record, CACHE


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clusters', type=int, default=512)
    p.add_argument('--stage', choices=['mlp', 'kernel', 'robust'], default='mlp')
    p.add_argument('--width', type=int, default=384)
    p.add_argument('--epochs', type=int, default=35)
    p.add_argument('--smooth', type=float, default=0)
    p.add_argument('--patch-size', type=int, default=6)
    p.add_argument('--augmentation-ratio', type=float, default=.5)
    p.add_argument('--alpha', type=float, default=.1)
    p.add_argument('--learning-rate', type=float, default=.001)
    p.add_argument('--augmentation', choices=['none', 'mixed', 'robust', 'robust2', 'robust3'], default='robust')
    a = p.parse_args()
    x, y, v, z, train, clean, dev = data_and_splits()
    tag = f'patch{a.clusters}' if a.clusters else 'hog'
    if a.patch_size != 6:
        tag += f'_p{a.patch_size}'
    if a.smooth:
        tag += f'_smooth{a.smooth}'
    banks = joblib.load(CACHE / f'{tag}_bank.joblib') if a.clusters else []
    f = features_cached(tag + '_train', x, banks)
    vf = features_cached(tag + '_validation', v, banks)
    ft = np.asarray(f[train])
    yt = y[train]
    name = tag + '_' + a.augmentation
    if a.augmentation_ratio != .5:
        name += f'_ratio{a.augmentation_ratio}'
    augmentation_cache = name
    if a.alpha != .1:
        name += f'_alpha{a.alpha}'
    if a.augmentation != 'none':
        chosen = np.random.default_rng(42).choice(train, size=int(len(train)*a.augmentation_ratio), replace=a.augmentation_ratio>1)
        if a.augmentation == 'robust3':
            augmented = augment_robust3(x[chosen], 42)
        else:
            augmented = augment_images(x[chosen], 42, a.augmentation)
        af = features_cached(augmentation_cache, augmented, banks)
        ft = np.concatenate([ft, af])
        yt = np.concatenate([yt, y[chosen]])
    if a.stage == 'robust':
        evaluate(name, f, vf, y, z, train, clean, dev, (af, y[chosen]))
        return
    scaler = StandardScaler()
    ft = scaler.fit_transform(ft)
    fc, fd = scaler.transform(f[clean]), scaler.transform(vf[dev])
    if a.stage == 'kernel':
        for gamma in (.0001, .0003, .001):
            model = Pipeline([('pca', PCA(n_components=256, random_state=42)),
                              ('kernel', Nystroem(gamma=gamma, n_components=1536, random_state=42)),
                              ('ridge', RidgeClassifier(alpha=.1))])
            started = time.perf_counter()
            with threadpool_limits(limits=6):
                model.fit(ft, yt)
                pc, pd = model.predict(fc), model.predict(fd)
            row = dict(experiment=name, classifier=f'nystroem{gamma}',
                       clean_accuracy=float(np.mean(pc == y[clean])),
                       dev_accuracy=float(np.mean(pd == z[dev])),
                       fit_seconds=round(time.perf_counter()-started, 2))
            record(row)
            joblib.dump(dict(scaler=scaler, model=model, row=row), CACHE / f'{name}_kernel{gamma}.joblib')
        return
    model = MLPClassifier(hidden_layer_sizes=(a.width, 128), alpha=a.alpha,
                          learning_rate_init=a.learning_rate, batch_size=512,
                          max_iter=1, random_state=42, shuffle=True)
    best_score = -1
    best_epoch = 0
    started = time.perf_counter()
    for epoch in range(1, a.epochs + 1):
        with threadpool_limits(limits=6):
            model.partial_fit(ft, yt, classes=np.arange(10))
            flush_tiny_training_values(model)
            pd = model.predict(fd)
            pc = model.predict(fc)
        score = float(np.mean(pd == z[dev]))
        row = dict(experiment=name, classifier=f'mlp{a.width}', epoch=epoch,
                   clean_accuracy=float(np.mean(pc == y[clean])), dev_accuracy=score,
                   loss=float(model.loss_), elapsed_seconds=round(time.perf_counter()-started, 2))
        record(row)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            joblib.dump(dict(scaler=scaler, model=model, row=row), CACHE / f'{name}_mlp{a.width}.joblib')
        if epoch - best_epoch >= 7:
            break


if __name__ == '__main__':
    main()
