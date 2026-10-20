"""Development-only 'robust3' augmentation; not part of the frozen submission.

Kept outside assignment_model.py so the frozen, notebook-embedded source and its
recorded hash stay unchanged. Random draws follow augment_images exactly
(flip, shift, then corruption), so cached experiment features are reproducible.
"""
import numpy as np
from scipy.ndimage import convolve, gaussian_filter


def _corrupt_v3(image, kind, rng):
    """One broad corruption of a float32 (32, 32, 3) image; kind 0 is identity.

    Wider than robust2: stronger noise, colored impulses, multi-box occlusion,
    fog/contrast/color casts, 4x pixelation, motion blur, and posterization.
    """
    x = image
    if kind == 1:
        x = x + rng.normal(0, rng.uniform(8, 75), size=x.shape)
    elif kind == 2:
        if rng.random() < .5:
            x = gaussian_filter(x, sigma=(rng.uniform(.5, 2.5),) * 2 + (0,))
        else:
            length = int(rng.integers(3, 8))
            kernel = np.zeros((length, length), np.float32)
            direction = rng.integers(0, 4)
            if direction == 0:
                kernel[length // 2] = 1
            elif direction == 1:
                kernel[:, length // 2] = 1
            else:
                kernel = np.eye(length, dtype=np.float32)
                if direction == 3:
                    kernel = kernel[::-1]
            kernel /= kernel.sum()
            x = np.stack([convolve(x[..., c], kernel, mode='reflect') for c in range(3)], axis=2)
    elif kind == 3:
        mean = x.mean(axis=(0, 1), keepdims=True)
        x = mean + rng.uniform(.3, .9) * (x - mean)
        if rng.random() < .5:
            fog = rng.uniform(.15, .6)
            x = x * (1 - fog) + fog * rng.uniform(150, 255)
    elif kind == 4:
        x = x * rng.uniform(.6, 1.4, size=(1, 1, 3)) + rng.uniform(-50, 50)
        if rng.random() < .5:
            gray = x.mean(axis=2, keepdims=True)
            x = gray + rng.uniform(0, .8) * (x - gray)
    elif kind == 5:
        mask = rng.random((32, 32)) < rng.uniform(.01, .12)
        if rng.random() < .5:
            x[mask] = rng.choice([0, 255], size=(mask.sum(), 3))
        else:
            x[mask] = rng.choice([0, 255], size=(mask.sum(), 1))
    elif kind == 6:
        factor = int(rng.choice([2, 4]))
        n = 32 // factor
        reduced = x.reshape(n, factor, n, factor, 3).mean(axis=(1, 3))
        x = np.repeat(np.repeat(reduced, factor, axis=0), factor, axis=1)
    elif kind == 7:
        for _ in range(int(rng.integers(1, 7))):
            h, w = rng.integers(3, 13, size=2)
            r, c = rng.integers(0, 33 - h), rng.integers(0, 33 - w)
            color = 0 if rng.random() < .5 else rng.integers(0, 256, size=3)
            x[r:r+h, c:c+w] = color
    elif kind == 8:
        levels = int(rng.integers(3, 9))
        step = 255 / (levels - 1)
        x = np.round(np.clip(x, 0, 255) / step) * step
    return np.clip(x, 0, 255).astype(np.float32)


def augment_robust3(images, seed):
    """Seeded flip/shift plus one corruption, compounded with probability 0.25."""
    rng = np.random.default_rng(seed)
    x = images.astype(np.float32).copy()
    flips = rng.random(len(x)) < .5
    x[flips] = x[flips, :, ::-1]
    for i in range(len(x)):
        dy, dx = rng.integers(-2, 3, size=2)
        padded = np.pad(x[i], ((2, 2), (2, 2), (0, 0)), mode='reflect')
        x[i] = padded[2+dy:34+dy, 2+dx:34+dx]
    for i in range(len(x)):
        kinds = [int(rng.integers(0, 9))]
        if kinds[0] and rng.random() < .25:
            kinds.append(int(rng.integers(1, 9)))
        for kind in kinds:
            x[i] = _corrupt_v3(x[i], kind, rng)
    return np.clip(x, 0, 255).astype(np.uint8)
