"""Value of median5 and autocontrast members on top of the seven-member v2 recipe.

Primary comparison, fixed before the runs finished: the seven v2 members versus
those seven plus median5 and autocontrast (nine members, equal weights, flip
TTA). Single additions are secondary. Audit labels are never scored.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.stats import binomtest
from run_experiments import data_and_splits, record, CACHE

V2 = ('raw', 'strong', 'largepatch', 'smooth', 'median3', 'gray', 'lowres')
EXTRA = ('median5', 'autocontrast')


def main():
    _, _, _, z, _, _, dev = data_and_splits()
    zd = z[dev]
    p = {k[:-4]: v for k, v in np.load(CACHE / 'finalist_probabilities.npz').items() if k.endswith('_tta')}
    for name in ('median3', 'gray', 'lowres') + EXTRA:
        path = CACHE / f'pre_{name}_patch512_robust_ratio2.0_alpha5.0_epoch_tta_probabilities.npz'
        if path.exists():
            saved = np.load(path)
            p[name] = saved[f"epoch{int(saved['best_epoch'])}"]
            if name in EXTRA:
                record(dict(experiment='extra_member_comparison', classifier=name + '_tta',
                            dev_accuracy=float(np.mean(p[name].argmax(1) == zd)),
                            epoch=int(saved['best_epoch'])))
    correct = lambda combo: np.mean([p[m] for m in combo], axis=0).argmax(1) == zd
    base = correct(V2)
    for combo, primary in [(V2 + EXTRA, True)] + [(V2 + (m,), False) for m in EXTRA]:
        if not all(m in p for m in combo):
            continue
        new = correct(combo)
        lost, gained = int((base & ~new).sum()), int((~base & new).sum())
        record(dict(experiment='extra_member_comparison', classifier='+'.join(combo) + '_tta',
                    primary=primary, dev_accuracy=float(new.mean()),
                    baseline_accuracy=float(base.mean()), gained=gained, lost=lost,
                    mcnemar_p=float(binomtest(gained, gained + lost).pvalue) if gained + lost else 1.0))


if __name__ == '__main__':
    main()
