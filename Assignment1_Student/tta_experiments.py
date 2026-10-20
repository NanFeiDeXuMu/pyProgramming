"""Extra test-time views for the four frozen development finalists (dev split only).

Primary comparison, fixed before scoring: the current ensemble with views
{original, flip} versus {original, flip, median3, median3+flip}, equal weights.
Shift views are secondary. Audit labels are never scored.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import joblib
import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits
from run_experiments import data_and_splits, features_cached, record, CACHE

FINALISTS = [('raw', 'patch512', 'patch512_robust_ratio2.0_alpha5.0_mlp384.joblib'),
             ('smooth', 'patch1024_smooth0.6', 'patch1024_smooth0.6_robust_ratio1.0_alpha5.0_mlp384.joblib'),
             ('largepatch', 'patch512_p8', 'patch512_p8_robust_ratio2.0_alpha5.0_mlp384.joblib'),
             ('strong', 'patch512', 'patch512_robust2_ratio2.0_alpha5.0_mlp384.joblib')]


def shift(images, dx):
    """Shift columns by one pixel, replicating the exposed edge column."""
    out = images.copy()
    if dx > 0:
        out[:, :, 1:] = images[:, :, :-1]
    else:
        out[:, :, :-1] = images[:, :, 1:]
    return out


def compare(name, base, new, primary=False):
    lost, gained = int((base & ~new).sum()), int((~base & new).sum())
    record(dict(experiment='tta_comparison', classifier=name, primary=primary,
                dev_accuracy=float(new.mean()), baseline_accuracy=float(base.mean()),
                gained=gained, lost=lost,
                mcnemar_p=float(binomtest(gained, gained + lost).pvalue) if gained + lost else 1.0))


def main():
    _, _, v, z, _, _, dev = data_and_splits()
    images = np.asarray(v[dev])
    median = median_filter(images, size=(1, 3, 3, 1))
    views = {'median': median, 'median_flip': median[:, :, ::-1],
             'shift_r': shift(images, 1), 'shift_l': shift(images, -1)}
    probabilities = {}
    for name, tag, filename in FINALISTS:
        fitted = joblib.load(CACHE / filename)
        banks = joblib.load(CACHE / (tag + '_bank.joblib'))
        view_features = {'orig': np.load(CACHE / (tag + '_validation.npy'), mmap_mode='r')[dev],
                         'flip': np.load(CACHE / (tag + '_dev_flip.npy'))}
        for view, data in views.items():
            view_features[view] = features_cached(f'{tag}_dev_{view}', data, banks)
        with threadpool_limits(limits=6):
            for view, f in view_features.items():
                probabilities[f'{name}/{view}'] = fitted['model'].predict_proba(fitted['scaler'].transform(f))
    np.savez(CACHE / 'tta_view_probabilities.npz', **probabilities)
    members = [n for n, _, _ in FINALISTS]
    correct = lambda vs, ms=members: np.mean(
        [probabilities[f'{m}/{view}'] for m in ms for view in vs], axis=0).argmax(1) == z[dev]
    base = correct(['orig', 'flip'])
    compare('ensemble_orig+flip+median+median_flip', base,
            correct(['orig', 'flip', 'median', 'median_flip']), primary=True)
    compare('ensemble_orig+flip+shift_r+shift_l', base, correct(['orig', 'flip', 'shift_r', 'shift_l']))
    compare('ensemble_all_six_views', base,
            correct(['orig', 'flip', 'median', 'median_flip', 'shift_r', 'shift_l']))
    for m in members:
        compare(f'{m}_orig+flip+median+median_flip', correct(['orig', 'flip'], [m]),
                correct(['orig', 'flip', 'median', 'median_flip'], [m]))


if __name__ == '__main__':
    main()
