"""Compare out-of-fold ensembles with and without validation images in training.

Primary comparison, fixed before the seven-member runs finished: the
seven-member equal-weight ensemble trained with validation images (repeat r)
versus the identical recipe with repeat 0. Both use the same folds, epochs and
flip TTA, so the control has no epoch or ensemble selection on these labels.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import numpy as np
from scipy.stats import binomtest
from run_experiments import data_and_splits, record, CACHE
from validation_training import MEMBERS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repeat', type=int, required=True)
    a = p.parse_args()
    _, _, _, z, _, _, dev = data_and_splits()
    zd = z[dev]
    load = lambda r, m: np.load(CACHE / f'valtrain_r{r}_{m}_oof.npy')
    members = [m for m in MEMBERS if (CACHE / f'valtrain_r0_{m}_oof.npy').exists()
               and (CACHE / f'valtrain_r{a.repeat}_{m}_oof.npy').exists()]
    for m in members:
        record(dict(experiment='validation_training_member', classifier=m, repeat=a.repeat,
                    control_accuracy=float(np.mean(load(0, m).argmax(1) == zd)),
                    dev_accuracy=float(np.mean(load(a.repeat, m).argmax(1) == zd))))
    base = np.mean([load(0, m) for m in members], axis=0).argmax(1) == zd
    new = np.mean([load(a.repeat, m) for m in members], axis=0).argmax(1) == zd
    lost, gained = int((base & ~new).sum()), int((~base & new).sum())
    record(dict(experiment='validation_training_ensemble', classifier='+'.join(members), repeat=a.repeat,
                primary=len(members) == len(MEMBERS), dev_accuracy=float(new.mean()),
                control_accuracy=float(base.mean()), gained=gained, lost=lost,
                mcnemar_p=float(binomtest(gained, gained + lost).pvalue) if gained + lost else 1.0))


if __name__ == '__main__':
    main()
