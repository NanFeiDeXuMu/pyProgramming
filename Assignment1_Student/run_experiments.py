"""Logged development experiments. Official audit labels are never scored here."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import argparse
import json
import time
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from threadpoolctl import threadpool_limits
from assignment_model import learn_dictionary, image_features, augment_images

CACHE = ROOT / '.cache'
CACHE.mkdir(exist_ok=True)
RESULTS = ROOT / 'experiment_results.jsonl'


def data_and_splits():
    x = np.load(ROOT / 'student_data/CIFAR10_Data/x_train.npy', mmap_mode='r')
    y = np.load(ROOT / 'student_data/CIFAR10_Data/y_train.npy').ravel()
    v = np.load(ROOT / 'student_data/validation_set_to_students/images.npy', mmap_mode='r')
    z = np.load(ROOT / 'student_data/validation_set_to_students/labels.npy')
    train, clean = train_test_split(np.arange(len(y)), test_size=5000, stratify=y, random_state=1729)
    dev, audit = train_test_split(np.arange(len(z)), test_size=1500, stratify=z, random_state=1729)
    np.savez(CACHE / 'splits.npz', train=train, clean=clean, dev=dev, audit=audit)
    return x, y, v, z, train, clean, dev


def record(row):
    print(json.dumps(row), flush=True)
    with RESULTS.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row) + '\n')


def features_cached(name, images, banks):
    path = CACHE / (name + '.npy')
    if not path.exists():
        t = time.perf_counter()
        f = image_features(images, banks=banks)
        np.save(path, f)
        print(f'Features {name}: {f.shape}, {time.perf_counter()-t:.1f}s', flush=True)
    return np.load(path, mmap_mode='r')


def evaluate(name, f, vf, y, z, train, clean, dev, augmented=None, selected=None):
    scaler = StandardScaler()
    ft = np.asarray(f[train])
    yt = y[train]
    if augmented is not None:
        af, ay = augmented
        ft = np.concatenate([ft, af])
        yt = np.concatenate([yt, ay])
    ft = scaler.fit_transform(ft)
    fc = scaler.transform(f[clean])
    fd = scaler.transform(vf[dev])
    candidates = selected or [('ridge100', RidgeClassifier(alpha=100, solver='lsqr', tol=1e-3)),
                             ('ridge1000', RidgeClassifier(alpha=1000, solver='lsqr', tol=1e-3)),
                             ('svm003', LinearSVC(C=.003, dual=False, tol=1e-3, max_iter=1500, random_state=42)),
                             ('svm03', LinearSVC(C=.03, dual=False, tol=1e-3, max_iter=1500, random_state=42))]
    for label, model in candidates:
        start = time.perf_counter()
        with threadpool_limits(limits=6):
            model.fit(ft, yt)
            pc, pd = model.predict(fc), model.predict(fd)
        row = dict(experiment=name, classifier=label, train_n=len(yt), features=ft.shape[1],
                   clean_accuracy=float(accuracy_score(y[clean], pc)),
                   dev_accuracy=float(accuracy_score(z[dev], pd)),
                   fit_seconds=round(time.perf_counter()-start, 2))
        record(row)
        joblib.dump(dict(scaler=scaler, model=model, row=row), CACHE / f'{name}_{label}.joblib')
        np.savez(CACHE / f'{name}_{label}_predictions.npz', clean=pc, dev=pd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['baseline', 'dictionary', 'augment'], required=True)
    parser.add_argument('--clusters', type=int, default=512)
    parser.add_argument('--smooth', type=float, default=0)
    parser.add_argument('--patch-size', type=int, default=6)
    parser.add_argument('--features-only', action='store_true')
    args = parser.parse_args()
    x, y, v, z, train, clean, dev = data_and_splits()
    if args.stage == 'baseline':
        f = features_cached('hog_train', x, [])
        vf = features_cached('hog_validation', v, [])
        evaluate('hog_color', f, vf, y, z, train, clean, dev)
        return
    tag = f'patch{args.clusters}'
    if args.patch_size != 6:
        tag += f'_p{args.patch_size}'
    if args.smooth:
        tag += f'_smooth{args.smooth}'
    bank_path = CACHE / f'{tag}_bank.joblib'
    if bank_path.exists():
        banks = joblib.load(bank_path)
    else:
        banks = [learn_dictionary(x[train], 42, n_clusters=args.clusters, smooth_sigma=args.smooth, patch_size=args.patch_size)]
        joblib.dump(banks, bank_path)
    f = features_cached(tag + '_train', x, banks)
    vf = features_cached(tag + '_validation', v, banks)
    if args.features_only:
        return
    if args.stage == 'dictionary':
        evaluate(tag, f, vf, y, z, train, clean, dev)
    else:
        rng = np.random.default_rng(42)
        chosen = rng.choice(train, size=22500, replace=False)
        for mode in ('flip', 'mixed'):
            af = features_cached(tag + '_' + mode, augment_images(x[chosen], 42, mode), banks)
            evaluate(tag + '_' + mode, f, vf, y, z, train, clean, dev, (af, y[chosen]),
                     [('ridge1000', RidgeClassifier(alpha=1000, solver='lsqr', tol=1e-3)),
                      ('svm003', LinearSVC(C=.003, dual=False, tol=1e-3, max_iter=1500, random_state=42))])


if __name__ == '__main__':
    main()
