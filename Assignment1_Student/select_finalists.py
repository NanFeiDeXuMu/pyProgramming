"""Compare frozen candidate checkpoints on development data only, including TTA."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import json
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from run_experiments import data_and_splits, features_cached, record, CACHE, ROOT


def main():
    _, _, v, z, _, _, dev = data_and_splits()
    candidates = [
        ('raw', 'patch512', 'patch512_robust_ratio2.0_alpha5.0_mlp384.joblib'),
        ('smooth', 'patch1024_smooth0.6', 'patch1024_smooth0.6_robust_ratio1.0_alpha5.0_mlp384.joblib'),
        ('largepatch', 'patch512_p8', 'patch512_p8_robust_ratio2.0_alpha5.0_mlp384.joblib'),
        ('strong', 'patch512', 'patch512_robust2_ratio2.0_alpha5.0_mlp384.joblib')]
    probabilities = {}
    for name, tag, filename in candidates:
        fitted = joblib.load(CACHE / filename)
        banks = joblib.load(CACHE / (tag + '_bank.joblib'))
        original = np.load(CACHE / (tag + '_validation.npy'), mmap_mode='r')[dev]
        flipped = features_cached(tag + '_dev_flip', v[dev, :, ::-1], banks)
        with threadpool_limits(limits=6):
            proba = fitted['model'].predict_proba(fitted['scaler'].transform(original))
            flip_proba = fitted['model'].predict_proba(fitted['scaler'].transform(flipped))
        probabilities[name] = proba
        probabilities[name + '_tta'] = (proba + flip_proba) / 2
        for key in (name, name + '_tta'):
            record(dict(experiment='finalist_comparison', classifier=key,
                        dev_accuracy=float(np.mean(probabilities[key].argmax(axis=1) == z[dev])),
                        epoch=fitted['row']['epoch']))
    for use_tta in (False, True):
        suffix = '_tta' if use_tta else ''
        for raw_weight in (.25, .5, .75):
            probability = raw_weight * probabilities['raw' + suffix] + (1-raw_weight) * probabilities['smooth' + suffix]
            label = f'ensemble_raw{raw_weight}{suffix}'
            probabilities[label] = probability
            record(dict(experiment='finalist_comparison', classifier=label,
                        dev_accuracy=float(np.mean(probability.argmax(axis=1) == z[dev]))))
        for combination in [('strong', 'largepatch'), ('raw', 'largepatch'),
                            ('strong', 'largepatch', 'smooth'), ('raw', 'strong', 'largepatch', 'smooth')]:
            probability = np.mean([probabilities[name + suffix] for name in combination], axis=0)
            label = '+'.join(combination) + suffix
            probabilities[label] = probability
            record(dict(experiment='finalist_comparison', classifier=label,
                        dev_accuracy=float(np.mean(probability.argmax(axis=1) == z[dev]))))
    np.savez(CACHE / 'finalist_probabilities.npz', **probabilities)


if __name__ == '__main__':
    main()
