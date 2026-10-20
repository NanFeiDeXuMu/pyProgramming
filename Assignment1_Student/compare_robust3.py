"""Compare robust3 development members against frozen finalists (dev split only).

Primary comparison, fixed before the largepatch/smooth robust3 runs finished:
current ensemble (raw+strong+largepatch+smooth) versus the same ensemble with
raw, largepatch and smooth each replaced by its robust3 counterpart. Audit
labels are never scored here.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import joblib
import numpy as np
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits
from run_experiments import data_and_splits, features_cached, record, CACHE

ROBUST3 = [('raw3', 'patch512', 'patch512_robust3_ratio2.0_alpha5.0_mlp384.joblib'),
           ('largepatch3', 'patch512_p8', 'patch512_p8_robust3_ratio2.0_alpha5.0_mlp384.joblib'),
           ('smooth3', 'patch1024_smooth0.6', 'patch1024_smooth0.6_robust3_ratio1.0_alpha5.0_mlp384.joblib')]
BASELINE = ('raw', 'strong', 'largepatch', 'smooth')
PRIMARY = ('raw3', 'strong', 'largepatch3', 'smooth3')
SECONDARY = [('raw3', 'strong', 'largepatch', 'smooth'),
             ('raw', 'strong', 'largepatch3', 'smooth'),
             ('raw', 'strong', 'largepatch', 'smooth3'),
             ('raw3', 'largepatch3', 'smooth3')]


def main():
    _, _, v, z, _, _, dev = data_and_splits()
    p = dict(np.load(CACHE / 'finalist_probabilities.npz'))
    epochs = {}
    for name, tag, filename in ROBUST3:
        if not (CACHE / filename).exists():
            print('missing', filename)
            continue
        fitted = joblib.load(CACHE / filename)
        banks = joblib.load(CACHE / (tag + '_bank.joblib'))
        original = np.load(CACHE / (tag + '_validation.npy'), mmap_mode='r')[dev]
        flipped = features_cached(tag + '_dev_flip', v[dev, :, ::-1], banks)
        with threadpool_limits(limits=6):
            proba = fitted['model'].predict_proba(fitted['scaler'].transform(original))
            flip = fitted['model'].predict_proba(fitted['scaler'].transform(flipped))
        p[name + '_tta'] = (proba + flip) / 2
        epochs[name] = fitted['row']['epoch']
        record(dict(experiment='robust3_comparison', classifier=name + '_tta',
                    dev_accuracy=float(np.mean(p[name + '_tta'].argmax(1) == z[dev])),
                    epoch=epochs[name]))
    correct = lambda combo: np.mean([p[n + '_tta'] for n in combo], axis=0).argmax(1) == z[dev]
    base = correct(BASELINE)
    for combo in [PRIMARY] + SECONDARY:
        if not all(n + '_tta' in p for n in combo):
            continue
        new = correct(combo)
        lost, gained = int((base & ~new).sum()), int((~base & new).sum())
        record(dict(experiment='robust3_comparison', classifier='+'.join(combo) + '_tta',
                    primary=combo == PRIMARY, dev_accuracy=float(new.mean()),
                    baseline_accuracy=float(base.mean()), gained=gained, lost=lost,
                    mcnemar_p=float(binomtest(gained, gained + lost).pvalue) if gained + lost else 1.0,
                    epochs=epochs))


if __name__ == '__main__':
    main()
