"""Aged paper and the age/photo effects applied over finished handwriting:
yellowed stock, tide-mark stains, foxing, fibres, fold creases, show-through
from the reverse side, uneven lighting and a slight camera perspective."""
from __future__ import annotations

import cv2
import numpy as np

from .degrade import _smooth_noise

# Paper tones (BGR): off-white, cream, yellowed, tea-brown, grey.
TONES = [(222, 232, 236), (196, 220, 232), (170, 205, 225), (150, 185, 210), (200, 205, 205)]


def aged_paper(h: int, w: int, rng: np.random.Generator, age: float) -> np.ndarray:
    """A blank sheet; `age` in [0, 1] sets how yellowed and stained it is."""
    i = rng.uniform(0, age * (len(TONES) - 1))
    lo, hi = np.float32(TONES[int(i)]), np.float32(TONES[min(int(i) + 1, len(TONES) - 1)])
    base = lo + (hi - lo) * (i - int(i))
    page = np.empty((h, w, 3), np.float32)
    page[:] = base
    # Large blotchy discolouration and darker, browner edges.
    blotch = _smooth_noise((h, w), rng.uniform(120, 300), rng)
    page *= (1 - age * 0.12 * blotch)[..., None]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    edge = np.minimum.reduce([xx, yy, w - 1 - xx, h - 1 - yy]) / (0.08 * min(h, w))
    burn = np.clip(1 - edge, 0, 1) ** 2 * age * rng.uniform(0.1, 0.3)
    page *= (1 - burn[..., None] * np.float32([0.7, 0.45, 0.25]))  # blue fades first: edges go brown
    # Tide-mark stains: a pale interior with a darker rim.
    for _ in range(rng.poisson(age * 2.5)):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(0.05, 0.2) * w
        d = np.hypot((xx - cx) / r, (yy - cy) / (r * rng.uniform(0.6, 1.4)))
        d += 0.25 * _smooth_noise((h, w), r * 0.4, rng)  # irregular outline
        rim = np.exp(-((d - 1.0) / 0.05) ** 2) * rng.uniform(0.06, 0.15)
        inner = (d < 1.0) * rng.uniform(0.0, 0.05)
        page *= (1 - (rim + inner)[..., None] * np.float32([0.9, 0.6, 0.3]))
    # Foxing: small rust spots.
    for _ in range(rng.poisson(age * 40)):
        cx, cy, r = int(rng.uniform(0, w)), int(rng.uniform(0, h)), int(rng.uniform(1, 5))
        spot = np.zeros((h, w), np.float32)
        cv2.circle(spot, (cx, cy), r, rng.uniform(0.1, 0.3), -1, cv2.LINE_AA)
        page *= (1 - cv2.GaussianBlur(spot, (0, 0), r * 0.8 + 0.5)[..., None] * np.float32([0.9, 0.6, 0.35]))
    # Fibre grain.
    grain = rng.normal(0, 1, (h, w)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (0, 0), 0.8) + 0.5 * cv2.GaussianBlur(grain, (0, 0), 3)
    page += grain[..., None] * rng.uniform(2, 5)
    return np.clip(page, 0, 255).astype(np.uint8)


def ruled_lines(page: np.ndarray, ys, rng: np.random.Generator) -> None:
    """Faint printed notebook rules (light blue-grey) at the given heights."""
    h, w = page.shape[:2]
    rules = np.zeros((h, w), np.float32)
    for y in ys:
        cv2.line(rules, (0, int(round(y))), (w, int(round(y))), 1.0, 1, cv2.LINE_AA)
    rules = cv2.GaussianBlur(rules, (0, 0), 0.6) * rng.uniform(0.25, 0.5)
    tint = np.float32(rng.choice([(205, 175, 150), (190, 185, 180)], axis=0)) / 255
    page[:] = np.clip(page * (1 - rules[..., None] + rules[..., None] * tint), 0, 255).astype(np.uint8)


def show_through(page: np.ndarray, ink_alpha: np.ndarray, ink_bgr, rng: np.random.Generator, strength: float) -> None:
    """Faint mirrored writing from the back of the sheet, blurred by the paper."""
    back = cv2.GaussianBlur(np.ascontiguousarray(ink_alpha[:, ::-1]), (0, 0), rng.uniform(1.5, 3))
    shift = np.float32([[1, 0, rng.uniform(-40, 40)], [0, 1, rng.uniform(-30, 30)]])
    back = cv2.warpAffine(back, shift, back.shape[::-1]) * strength
    ink = np.asarray(ink_bgr, np.float32) / 255
    page[:] = np.clip(page * (1 - back[..., None] + back[..., None] * ink), 0, 255).astype(np.uint8)


def folds(page: np.ndarray, rng: np.random.Generator, n: int) -> None:
    """Fold creases: a thin dark line with a lit side and a shaded side."""
    h, w = page.shape[:2]
    out = page.astype(np.float32)
    for k in range(n):
        horizontal = k == 0 or rng.random() < 0.5
        L = h if horizontal else w
        pos = L * (rng.choice([0.5, 1 / 3, 2 / 3]) + rng.normal(0, 0.01))
        coord = np.arange(L, dtype=np.float32) - pos
        prof = -0.12 * np.exp(-(coord / 1.5) ** 2) + 0.05 * np.tanh(coord / 25) * np.exp(-np.abs(coord) / 60)
        prof *= rng.uniform(0.5, 1.0)
        along = 1 + 0.4 * (_smooth_noise((1, w if horizontal else h), 200, rng)[0] - 0.5)  # uneven along the fold
        f = prof[:, None] * along[None, :] if horizontal else (prof[:, None] * along[None, :]).T
        out *= (1 + f)[..., None]
    page[:] = np.clip(out, 0, 255).astype(np.uint8)


def photograph(page: np.ndarray, rng: np.random.Generator, strength: float) -> tuple[np.ndarray, np.ndarray]:
    """Uneven light, slight perspective, sensor noise and JPEG. Returns the
    image and the homography that maps page coordinates into it."""
    h, w = page.shape[:2]
    out = page.astype(np.float32)
    light = _smooth_noise((h, w), max(h, w) * 0.6, rng)
    out *= (1 - strength * 0.18 * (1 - light))[..., None]
    d = strength * 0.02 * min(h, w)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-d, d, (4, 2)).astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(np.clip(out, 0, 255).astype(np.uint8), H, (w, h), borderMode=cv2.BORDER_REPLICATE)
    out = np.clip(out + rng.normal(0, rng.uniform(2, 5), out.shape), 0, 255).astype(np.uint8)
    q = int(rng.integers(70, 92))
    return cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR), H
