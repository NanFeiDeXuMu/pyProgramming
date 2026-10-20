"""Two-fold estimate of a stacker fitted on member validation probabilities.

Members never train on validation images, so their validation probabilities
are held-out outputs. A multinomial logistic regression on the concatenated
log-probabilities of the seven members is fitted on one development half and
scored on the other. Primary setting, fixed before scoring: C = 0.1. The
control is the equal-weight average on the same images. Audit labels unused.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from run_experiments import data_and_splits, record, CACHE

ORIGINAL = ('raw', 'strong', 'largepatch', 'smooth')
PREPROCESSED = ('median3', 'gray', 'lowres')


def main():
    _, _, _, z, _, _, dev = data_and_splits()
    zd = z[dev]
    saved = np.load(CACHE / 'finalist_probabilities.npz')
    probabilities = [saved[m + '_tta'] for m in ORIGINAL]
    for m in PREPROCESSED:
        epochs = np.load(CACHE / f'pre_{m}_patch512_robust_ratio2.0_alpha5.0_epoch_tta_probabilities.npz')
        probabilities.append(epochs[f"epoch{int(epochs['best_epoch'])}"])
    features = np.log(np.concatenate(probabilities, axis=1) + 1e-6)
    base = np.mean(probabilities, axis=0).argmax(1) == zd
    halves = train_test_split(np.arange(len(dev)), test_size=.5, stratify=zd, random_state=1729)
    for C in (.01, .1, 1.0):
        predicted = np.zeros(len(dev), int)
        for fit_part, score_part in (halves, halves[::-1]):
            stacker = LogisticRegression(C=C, max_iter=2000)
            stacker.fit(features[fit_part], zd[fit_part])
            predicted[score_part] = stacker.predict(features[score_part])
        new = predicted == zd
        lost, gained = int((base & ~new).sum()), int((~base & new).sum())
        record(dict(experiment='stacking_oof', classifier=f'logistic_C{C}', primary=C == .1,
                    dev_accuracy=float(new.mean()), control_accuracy=float(base.mean()),
                    gained=gained, lost=lost,
                    mcnemar_p=float(binomtest(gained, gained + lost).pvalue) if gained + lost else 1.0))


if __name__ == '__main__':
    main()
