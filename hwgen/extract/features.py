"""Style measurements for a handwriting sample. These describe the real
distribution that the generator's randomisation (task 2) should match."""
from __future__ import annotations

import cv2
import numpy as np


def x_height_band(mask: np.ndarray, frac: float = 0.35) -> tuple[int, int]:
    """Rows of the writing's core (body of lowercase letters / digits): the
    contiguous band around the densest row whose density is >= frac * peak."""
    prof = mask.sum(axis=1).astype(np.float32)
    if prof.max() == 0:
        return 0, mask.shape[0]
    prof = np.convolve(prof, np.ones(3) / 3, mode="same")
    peak = int(prof.argmax())
    thr = frac * prof[peak]
    top = peak
    while top > 0 and prof[top - 1] >= thr:
        top -= 1
    bot = peak
    while bot < len(prof) - 1 and prof[bot + 1] >= thr:
        bot += 1
    return top, bot + 1


def slant_angle(mask: np.ndarray, max_deg: int = 45, step: int = 2) -> float:
    """Shear angle (degrees, + = right-leaning) that makes strokes most vertical,
    i.e. maximises the energy of the vertical projection."""
    h, w = mask.shape
    m = mask.astype(np.float32)
    best, best_a = -1.0, 0.0
    for a in range(-max_deg, max_deg + 1, step):
        t = np.tan(np.radians(a))
        pad = int(abs(t) * h) + 1
        M = np.float32([[1, t, pad - t * h], [0, 1, 0]])  # shift so output stays in frame
        sheared = cv2.warpAffine(m, M, (w + 2 * pad, h), flags=cv2.INTER_NEAREST)
        e = float((sheared.sum(axis=0) ** 2).sum())
        if e > best:
            best, best_a = e, a
    return float(best_a)


def stroke_width(mask: np.ndarray) -> float:
    """Mean pen width ~= 2 * area / perimeter for thin strokes."""
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    perim = sum(cv2.arcLength(c, True) for c in contours)
    return float(2.0 * m.sum() / max(perim, 1.0))


def baseline_angle(mask: np.ndarray, x_height: float) -> float:
    """Degrees, + = rising to the right. Robust line through the bottoms of
    letter-sized components (descenders are excluded by the median filter)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    pts = [
        (s[0] + s[2] / 2.0, s[1] + s[3])
        for s in stats[1:]
        if s[cv2.CC_STAT_HEIGHT] >= 0.5 * x_height and s[cv2.CC_STAT_HEIGHT] <= 1.6 * x_height
    ]
    if len(pts) < 3:
        return 0.0
    xs, ys = np.array(pts).T
    k, b = np.polyfit(xs, ys, 1)
    resid = np.abs(ys - (k * xs + b))
    keep = resid <= max(np.median(resid) * 2.5, 1.0)
    if keep.sum() >= 3:
        k, b = np.polyfit(xs[keep], ys[keep], 1)
    return float(-np.degrees(np.arctan(k)))


def ink_stats(rgba: np.ndarray) -> dict:
    alpha = rgba[..., 3].astype(np.float32) / 255.0
    core = alpha > 0.6
    if core.sum() == 0:
        return {"ink_bgr": [0, 0, 0], "ink_darkness": 0.0, "ink_family": "black"}
    bgr = np.median(rgba[..., :3][core], axis=0)
    b, g, r = (float(v) for v in bgr)
    family = "blue" if b - r > 25 else "black"
    return {"ink_bgr": [round(b), round(g), round(r)], "ink_darkness": float(alpha[core].mean()), "ink_family": family}


def measure(rgba: np.ndarray, mask: np.ndarray) -> dict:
    top, bot = x_height_band(mask)
    xh = float(bot - top)
    ys = np.nonzero(mask.any(axis=1))[0]
    return {
        "width": int(mask.shape[1]),
        "ink_height": int(ys[-1] - ys[0] + 1) if len(ys) else 0,
        "x_height": xh,
        "baseline_y": int(bot),
        "slant_deg": slant_angle(mask),
        "stroke_width": round(stroke_width(mask), 2),
        "baseline_deg": round(baseline_angle(mask, xh), 2),
        **ink_stats(rgba),
    }


def headline_band(mask: np.ndarray, frac: float = 0.5) -> tuple[int, int]:
    """Rows of the Devanagari headline (shirorekha): the densest row in the
    upper 60% of the ink, extended while rows keep >= frac of its density."""
    rows = np.nonzero(mask.any(axis=1))[0]
    if len(rows) == 0:
        return 0, 0
    top, bot = rows[0], rows[-1] + 1
    prof = mask.sum(axis=1).astype(np.float32)
    search_end = top + max(1, int(0.6 * (bot - top)))
    peak = top + int(prof[top:search_end].argmax())
    thr = frac * prof[peak]
    a, b = peak, peak
    while a > top and prof[a - 1] >= thr:
        a -= 1
    while b < bot - 1 and prof[b + 1] >= thr:
        b += 1
    return int(a), int(b + 1)


def headline_angle(mask: np.ndarray, band: tuple[int, int], bins: int = 8) -> float:
    """Degrees, + = rising to the right: line through the densest headline row
    of each column bin (searched in a window around the global headline)."""
    a, b = band
    h = max(b - a, 1)
    lo, hi = max(0, a - 2 * h), min(mask.shape[0], b + 2 * h)
    cols = np.nonzero(mask[lo:hi].any(axis=0))[0]
    if len(cols) < bins * 2:
        return 0.0
    xs, ys = [], []
    for chunk in np.array_split(np.arange(cols[0], cols[-1] + 1), bins):
        prof = mask[lo:hi, chunk].sum(axis=1)
        if prof.max() >= 0.6 * len(chunk):  # a real horizontal run, not a gap
            xs.append(chunk.mean())
            ys.append(lo + prof.argmax())
    if len(xs) < 3:
        return 0.0
    k = np.polyfit(xs, ys, 1)[0]
    return float(-np.degrees(np.arctan(k)))


def measure_devanagari(rgba: np.ndarray, mask: np.ndarray) -> dict:
    """Style measurements for a Devanagari word. Heights are in pixels of the
    stored crop; `body_height` (headline to baseline) plays the role the
    x-height plays for Latin script."""
    rows = np.nonzero(mask.any(axis=1))[0]
    if len(rows) == 0:
        return {}
    top, bot = int(rows[0]), int(rows[-1] + 1)
    hl0, hl1 = headline_band(mask)
    below = mask[hl1:].sum(axis=1).astype(np.float64)
    # Baseline: where 92% of the ink under the headline lies above it, so
    # lower vowel signs (ु ू ृ) do not drag it down.
    if below.sum() > 0:
        baseline = hl1 + int(np.searchsorted(np.cumsum(below) / below.sum(), 0.92)) + 1
    else:
        baseline = bot
    body = max(baseline - hl1, 1)
    return {
        "width": int(mask.shape[1]),
        "height": int(mask.shape[0]),
        "ink_top": top,
        "headline_y": round((hl0 + hl1) / 2, 1),
        "headline_thickness": hl1 - hl0,
        "baseline_y": int(baseline),
        "body_height": int(body),
        "upper_extent": round((hl0 - top) / body, 3),      # matras above the headline
        "lower_extent": round((bot - baseline) / body, 3),  # matras below the baseline
        "slant_deg": slant_angle(mask),
        "stroke_width": round(stroke_width(mask), 2),
        "stroke_over_body": round(stroke_width(mask) / body, 3),
        "headline_deg": round(headline_angle(mask, (hl0, hl1)), 2),
        **ink_stats(rgba),
    }
