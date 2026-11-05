"""Turn a scanned single-word image into a library crop."""
from __future__ import annotations

import cv2
import numpy as np

from .ink import binarize, ink_matte, normalize


def clean_word(image: np.ndarray, max_height: int = 160, min_area: int = 20, margin: int = 3):
    """Return (rgba, mask) with paper normalised to white, scanner specks
    removed, cropped to the ink and downscaled so height <= max_height.
    Returns None if the image holds no usable ink."""
    norm, dark = normalize(image)
    binary = binarize(dark)
    b = 2  # scans carry a dark 1-px border line; it would stretch the crop
    binary[:b], binary[-b:], binary[:, :b], binary[:, -b:] = False, False, False, False
    binary &= ~_ruling(binary)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    H, W = binary.shape
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        near = 8
        if (x <= near or y <= near or x + w >= W - near or y + h >= H - near) and \
                min(w, h) <= 5 and max(w, h) >= 10 * min(w, h):
            continue  # scan edge / ruling line along the border (a danda is never at the border)
        if h <= 6 and w >= 0.4 * W and (y + h < 0.25 * H or y > 0.75 * H):
            continue  # paper ruling above/below the word; a headline is joined to its letters
        touches_edge = x <= b or y <= b or x + w >= W - b or y + h >= H - b
        # Small blobs are specks unless they could be an anusvara/nukta dot;
        # dots never touch the image border, specks from scanning often do.
        if area >= min_area and not (touches_edge and area < 8 * min_area):
            keep[i] = True
    # Small components far from the main strokes are specks, not dots/danda.
    areas = stats[:, cv2.CC_STAT_AREA]
    big = keep & (areas >= 0.02 * areas[keep].sum()) if keep.any() else keep
    if big.any():
        bx0 = stats[big, cv2.CC_STAT_LEFT].min()
        by0 = stats[big, cv2.CC_STAT_TOP].min()
        bx1 = (stats[big, cv2.CC_STAT_LEFT] + stats[big, cv2.CC_STAT_WIDTH]).max()
        by1 = (stats[big, cv2.CC_STAT_TOP] + stats[big, cv2.CC_STAT_HEIGHT]).max()
        reach = 0.6 * (by1 - by0)
        for i in np.nonzero(keep & ~big)[0]:
            x, y, w, h = stats[i, :4]
            if x + w < bx0 - reach or x > bx1 + reach or y + h < by0 - reach or y > by1 + reach:
                keep[i] = False
    mask = keep[labels]
    if mask.sum() < 50:
        return None
    ys, xs = np.nonzero(mask)
    y0, y1 = max(0, ys.min() - margin), min(H, ys.max() + 1 + margin)
    x0, x1 = max(0, xs.min() - margin), min(W, xs.max() + 1 + margin)
    support = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    alpha = (ink_matte(dark) * support)[y0:y1, x0:x1]
    rgb = norm[y0:y1, x0:x1]

    scale = min(1.0, max_height / rgb.shape[0])
    if scale < 1.0:
        size = (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale)))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
    a8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    rgb = rgb.copy()
    rgb[a8 == 0] = 255  # paper texture is invisible under alpha=0; white compresses ~10x better
    rgba = np.dstack([rgb, a8])
    return rgba, alpha > 0.35


def _ruling(binary: np.ndarray, min_frac: float = 0.5, d: int = 4) -> np.ndarray:
    """Paper ruling the word was written on or under: horizontal runs at least
    `min_frac` of the image wide, in the bottom 45% or top 20% of the scan (the
    headline sits in between). Only thin parts are removed, so strokes that
    cross or touch the line survive."""
    H, W = binary.shape
    b = binary.astype(np.uint8)
    long = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(min_frac * W)), 1)))
    band = np.zeros_like(long)
    band[int(0.55 * H) :] = 1
    band[: int(0.2 * H)] = 1
    long = cv2.dilate(long & band, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))) & b
    pad = np.pad(b, ((d, d), (0, 0)))
    thin = (pad[: -2 * d] == 0) | (pad[2 * d :] == 0)
    return (long > 0) & thin
