"""Ensemble value of preprocessing members (dev split only, flip TTA everywhere).

Primary comparison, fixed before the member runs finished: the current four
finalists versus those four plus median3, gray and lowres (seven members, equal
weights). Single additions are secondary. Audit labels are never scored.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.stats import binomtest
from run_experiments import data_and_splits, record, CACHE

BASELINE = ('raw', 'strong', 'largepatch', 'smooth')
NEW = ('median3', 'gray', 'lowres')


def main():
    _, _, _, z, _, _, dev = data_and_splits()
    p = {k[:-4]: v for k, v in np.load(CACHE / 'finalist_probabilities.npz').items() if k.endswith('_tta')}
    for name in NEW:
        path = CACHE / f'pre_{name}_patch512_robust_ratio2.0_alpha5.0_epoch_tta_probabilities.npz'
        if path.exists():
            saved = np.load(path)
            p[name] = saved[f"epoch{int(saved['best_epoch'])}"]
            record(dict(experiment='preprocess_comparison', classifier=name + '_tta',
                        dev_accuracy=float(np.mean(p[name].argmax(1) == z[dev])),
                        epoch=int(saved['best_epoch'])))
    correct = lambda combo: np.mean([p[m] for m in combo], axis=0).argmax(1) == z[dev]
    base = correct(BASELINE)
    combos = [(BASELINE + NEW, True)] + [(BASELINE + (m,), False) for m in NEW]
    for combo, primary in combos:
        if not all(m in p for m in combo):
            continue
        new = correct(combo)
        lost, gained = int((base & ~new).sum()), int((~base & new).sum())
        record(dict(experiment='preprocess_comparison', classifier='+'.join(combo) + '_tta',
                    primary=primary, dev_accuracy=float(new.mean()),
                    baseline_accuracy=float(base.mean()), gained=gained, lost=lost,
                    mcnemar_p=float(binomtest(gained, gained + lost).pvalue) if gained + lost else 1.0))


if __name__ == '__main__':
    main()
