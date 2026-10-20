"""Different classifier families as an eighth member (dev split only).

Input: the cached 'raw' features (K=512 patches + HOG + color) of the
45,000-image training split plus robust augmentation at ratio 2.0. After
standardization, a 256-component randomized PCA (fitted on training rows only)
makes tree models tractable. Flip-TTA development probabilities are saved.

Primary comparison, fixed before scoring: the seven-member v2 ensemble versus
v2 plus the HistGradientBoosting member (equal weights). v2 plus ExtraTrees is
secondary. Audit labels are never scored.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import time
import numpy as np
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from run_experiments import data_and_splits, record, CACHE

V2 = ('raw', 'strong', 'largepatch', 'smooth', 'median3', 'gray', 'lowres')


def pca_features(y, v, train, clean, dev):
    paths = {part: CACHE / f'family_pca256_{part}.npy' for part in ('train', 'clean', 'dev', 'dev_flip')}
    if not all(path.exists() for path in paths.values()):
        started = time.perf_counter()
        f = np.load(CACHE / 'patch512_train.npy', mmap_mode='r')
        af = np.load(CACHE / 'patch512_robust_ratio2.0.npy', mmap_mode='r')
        scaler = StandardScaler(copy=False)
        train_features = scaler.fit_transform(np.concatenate([np.asarray(f[train]), np.asarray(af)]))
        pca = PCA(n_components=256, svd_solver='randomized', random_state=42)
        with threadpool_limits(limits=16):
            np.save(paths['train'], pca.fit_transform(train_features).astype(np.float32))
            del train_features
            vf = np.load(CACHE / 'patch512_validation.npy', mmap_mode='r')
            np.save(paths['clean'], pca.transform(scaler.transform(np.asarray(f[clean]))).astype(np.float32))
            np.save(paths['dev'], pca.transform(scaler.transform(np.asarray(vf[dev]))).astype(np.float32))
            np.save(paths['dev_flip'], pca.transform(scaler.transform(np.load(CACHE / 'patch512_dev_flip.npy'))).astype(np.float32))
        print(f'PCA features ready: {time.perf_counter() - started:.1f}s', flush=True)
    return {part: np.load(path) for part, path in paths.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--families', default='hgb,extratrees')
    a = p.parse_args()
    _, y, v, z, train, clean, dev = data_and_splits()
    zd = z[dev]
    chosen = np.random.default_rng(42).choice(train, size=int(len(train) * 2.0), replace=True)
    targets = np.concatenate([y[train], y[chosen]])
    f = pca_features(y, v, train, clean, dev)
    models = {
        'hgb': HistGradientBoostingClassifier(max_iter=400, learning_rate=.1, max_leaf_nodes=31,
                                              l2_regularization=1.0, early_stopping=True,
                                              validation_fraction=.1, n_iter_no_change=20,
                                              random_state=42),
        'extratrees': ExtraTreesClassifier(n_estimators=400, max_features='sqrt', min_samples_leaf=2,
                                           n_jobs=16, random_state=42),
    }
    p_saved = {k[:-4]: v for k, v in np.load(CACHE / 'finalist_probabilities.npz').items() if k.endswith('_tta')}
    for name in ('median3', 'gray', 'lowres'):
        saved = np.load(CACHE / f'pre_{name}_patch512_robust_ratio2.0_alpha5.0_epoch_tta_probabilities.npz')
        p_saved[name] = saved[f"epoch{int(saved['best_epoch'])}"]
    base = np.mean([p_saved[m] for m in V2], axis=0).argmax(1) == zd
    for family in a.families.split(','):
        model = models[family]
        started = time.perf_counter()
        with threadpool_limits(limits=16):
            model.fit(f['train'], targets)
            tta = (model.predict_proba(f['dev']) + model.predict_proba(f['dev_flip'])) / 2
            clean_accuracy = float(np.mean(model.predict(f['clean']) == y[clean]))
        np.save(CACHE / f'family_{family}_dev_tta.npy', tta)
        new = np.mean([p_saved[m] for m in V2] + [tta], axis=0).argmax(1) == zd
        lost, gained = int((base & ~new).sum()), int((~base & new).sum())
        record(dict(experiment='family_comparison', classifier=f'{family}_pca256',
                    primary=family == 'hgb', fit_seconds=round(time.perf_counter() - started, 1),
                    iterations=int(getattr(model, 'n_iter_', 0)), clean_accuracy=clean_accuracy,
                    member_dev_accuracy=float(np.mean(tta.argmax(1) == zd)),
                    dev_accuracy=float(new.mean()), baseline_accuracy=float(base.mean()),
                    gained=gained, lost=lost,
                    mcnemar_p=float(binomtest(gained, gained + lost).pvalue) if gained + lost else 1.0))


if __name__ == '__main__':
    main()
