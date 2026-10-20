"""Ensemble value of higher-capacity members over the seven-member v2 recipe.

Primary comparison, fixed before the runs finished: v2 with 'raw' replaced by
the K=1600 member (still seven members, equal weights, flip TTA). Secondary:
raw replaced by the stride-1 member, and each variant added as an eighth member.
Audit labels are never scored.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.stats import binomtest
from run_experiments import data_and_splits, record, CACHE

V2 = ('raw', 'strong', 'largepatch', 'smooth', 'median3', 'gray', 'lowres')
VARIANTS = {'K1600': 'cap_K1600_s2', 'stride1': 'cap_K512_s1'}


def best_epoch_probabilities(path):
    saved = np.load(path)
    return saved[f"epoch{int(saved['best_epoch'])}"], int(saved['best_epoch'])


def main():
    _, _, _, z, _, _, dev = data_and_splits()
    zd = z[dev]
    p = {k[:-4]: v for k, v in np.load(CACHE / 'finalist_probabilities.npz').items() if k.endswith('_tta')}
    for name in ('median3', 'gray', 'lowres'):
        p[name], _ = best_epoch_probabilities(
            CACHE / f'pre_{name}_patch512_robust_ratio2.0_alpha5.0_epoch_tta_probabilities.npz')
    for name, stem in VARIANTS.items():
        path = CACHE / f'{stem}_robust_ratio2.0_alpha5.0_epoch_tta_probabilities.npz'
        if path.exists():
            p[name], epoch = best_epoch_probabilities(path)
            record(dict(experiment='capacity_comparison', classifier=name + '_tta', epoch=epoch,
                        dev_accuracy=float(np.mean(p[name].argmax(1) == zd)),
                        raw_dev_accuracy=float(np.mean(p['raw'].argmax(1) == zd))))
    correct = lambda combo: np.mean([p[m] for m in combo], axis=0).argmax(1) == zd
    base = correct(V2)
    combos = [(('K1600',) + V2[1:], True), (('stride1',) + V2[1:], False),
              (V2 + ('K1600',), False), (V2 + ('stride1',), False)]
    for combo, primary in combos:
        if not all(m in p for m in combo):
            continue
        new = correct(combo)
        lost, gained = int((base & ~new).sum()), int((~base & new).sum())
        record(dict(experiment='capacity_comparison', classifier='+'.join(combo) + '_tta',
                    primary=primary, dev_accuracy=float(new.mean()),
                    baseline_accuracy=float(base.mean()), gained=gained, lost=lost,
                    mcnemar_p=float(binomtest(gained, gained + lost).pvalue) if gained + lost else 1.0))


if __name__ == '__main__':
    main()
