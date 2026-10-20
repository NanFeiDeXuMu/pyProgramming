"""Out-of-fold estimate of adding labelled validation images to training.

The course TA confirmed the provided validation set may be used for training.
The 3,500 development images are split into two stratified halves. Each fold
model trains on the 45,000-image training split, its augmentation, and the other
half (original + horizontally flipped, repeated `--repeat` times), using the
member's frozen hyperparameters and epochs. Every development image is then
predicted, with flip TTA, by the fold model that never saw it. Audit labels
are never used. All features come from the existing development caches.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from assignment_model import flush_tiny_training_values
from run_experiments import data_and_splits, record, CACHE

# member: (training features, augmented features, augmentation ratio,
#          development features, flipped development features, epochs)
MEMBERS = {
    'raw': ('patch512_train', 'patch512_robust_ratio2.0', 2.0, 'patch512_validation', 'patch512_dev_flip', 5),
    'strong': ('patch512_train', 'patch512_robust2_ratio2.0', 2.0, 'patch512_validation', 'patch512_dev_flip', 15),
    'largepatch': ('patch512_p8_train', 'patch512_p8_robust_ratio2.0', 2.0, 'patch512_p8_validation', 'patch512_p8_dev_flip', 15),
    'smooth': ('patch1024_smooth0.6_train', 'patch1024_smooth0.6_robust_ratio1.0', 1.0,
               'patch1024_smooth0.6_validation', 'patch1024_smooth0.6_dev_flip', 23),
}
for _name, _epochs in (('median3', 15), ('gray', 14), ('lowres', 7)):
    MEMBERS[_name] = (f'pre_{_name}_patch512_trainsplit', f'pre_{_name}_patch512_robust_ratio2.0', 2.0,
                      f'pre_{_name}_patch512_dev', f'pre_{_name}_patch512_dev_flip', _epochs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--member', choices=sorted(MEMBERS), required=True)
    p.add_argument('--repeat', type=int, default=4)
    a = p.parse_args()
    _, y, _, z, train, _, dev = data_and_splits()
    train_name, aug_name, ratio, dev_name, flip_name, epochs = MEMBERS[a.member]
    # Memory-mapped; rows are copied straight into one preallocated matrix per fold.
    ft = np.load(CACHE / f'{train_name}.npy', mmap_mode='r')
    train_rows = slice(None) if train_name.endswith('_trainsplit') else train
    af = np.load(CACHE / f'{aug_name}.npy', mmap_mode='r')
    chosen = np.random.default_rng(42).choice(train, size=int(len(train) * ratio), replace=ratio > 1)
    fd = np.load(CACHE / f'{dev_name}.npy', mmap_mode='r')
    fd = np.asarray(fd) if dev_name.endswith('_dev') else np.asarray(fd[dev])
    fdf = np.load(CACHE / f'{flip_name}.npy')
    zd = z[dev]
    halves = train_test_split(np.arange(len(dev)), test_size=.5, stratify=zd, random_state=1729)
    out = np.zeros((len(dev), 10))
    started = time.perf_counter()
    for fold, (fit_part, score_part) in enumerate([halves, halves[::-1]]):
        # repeat=0 is the control: identical recipe and epochs without validation images.
        extra = np.concatenate([fd[fit_part], fdf[fit_part]] * a.repeat or [fd[:0]])
        extra_targets = np.concatenate([zd[fit_part], zd[fit_part]] * a.repeat or [zd[:0]])
        features = np.empty((len(train) + len(af) + len(extra), ft.shape[1]), np.float32)
        features[:len(train)] = ft[train_rows]
        features[len(train):len(train) + len(af)] = af
        features[len(train) + len(af):] = extra
        scaler = StandardScaler(copy=False)
        features = scaler.fit_transform(features)
        targets = np.concatenate([y[train], y[chosen], extra_targets])
        model = MLPClassifier(hidden_layer_sizes=(384, 128), alpha=5.0, learning_rate_init=.0005,
                              batch_size=512, max_iter=1, random_state=42, shuffle=True)
        for _ in range(epochs):
            with threadpool_limits(limits=4):
                model.partial_fit(features, targets, classes=np.arange(10))
            flush_tiny_training_values(model)
        del features
        with threadpool_limits(limits=4):
            out[score_part] = (model.predict_proba(scaler.transform(fd[score_part].copy()))
                               + model.predict_proba(scaler.transform(fdf[score_part].copy()))) / 2
        print(f'{a.member} repeat={a.repeat} fold {fold} done {time.perf_counter() - started:.0f}s', flush=True)
    np.save(CACHE / f'valtrain_r{a.repeat}_{a.member}_oof.npy', out)
    record(dict(experiment='validation_training_oof', classifier=f'{a.member}_tta', repeat=a.repeat,
                epochs=epochs, dev_accuracy=float(np.mean(out.argmax(1) == zd)),
                elapsed_seconds=round(time.perf_counter() - started, 2)))


if __name__ == '__main__':
    main()
