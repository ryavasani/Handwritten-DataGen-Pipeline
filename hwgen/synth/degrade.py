"""Ink-level imperfections applied to a rendered handwriting matte."""
from __future__ import annotations

import cv2
import numpy as np

from .style import Style


def _smooth_noise(shape: tuple[int, int], cell: float, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency noise in [0, 1] with features about `cell` pixels across."""
    gh, gw = max(2, int(shape[0] / cell) + 2), max(2, int(shape[1] / cell) + 2)
    n = cv2.resize(rng.random((gh, gw)).astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    n -= n.min()
    return n / max(float(n.max()), 1e-6)


def apply_ink(alpha: np.ndarray, style: Style, rng: np.random.Generator, n_words: int = 1) -> np.ndarray:
    a = alpha.copy()
    body = style.body_px
    # Pen pressure drifts slowly along the line.
    a *= 1.0 - style.pressure * _smooth_noise(a.shape, 3 * body, rng)
    # Ballpoints skip: occasional faded patches.
    patches = np.clip((_smooth_noise(a.shape, 1.5 * body, rng) - 0.7) / 0.3, 0, 1)
    a *= 1.0 - style.fade * patches
    # Ink blots where the pen rested.
    ys, xs = np.nonzero(alpha > 0.6)
    for _ in range(rng.poisson(style.blot_rate * n_words)):
        if len(xs) == 0:
            break
        i = rng.integers(len(xs))
        r = max(1, int(rng.uniform(0.08, 0.16) * body))
        blot = np.zeros_like(a)
        cv2.ellipse(blot, (int(xs[i]), int(ys[i])), (r, max(1, int(r * rng.uniform(0.6, 1.0)))),
                    rng.uniform(0, 180), 0, 360, 1.0, -1, cv2.LINE_AA)
        np.maximum(a, cv2.GaussianBlur(blot, (0, 0), 0.6) * 0.95, out=a)
    # Ink saturation varies by pen; then a slight blur as from scanning.
    a = np.clip(a, 0, 1) ** rng.uniform(0.85, 1.15)
    return cv2.GaussianBlur(a, (0, 0), rng.uniform(0.3, 0.6))
