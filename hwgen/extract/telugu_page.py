"""Cut gold-labelled word crops out of a real handwritten Telugu page.

The page-level dataset gives one transcription per page and no boxes. Google
Vision finds the words and their boxes; its reading is aligned to the ground
truth token by token, and a box is labelled with the ground-truth word only
when Vision's reading of it is close to that word. The label is always the
ground truth, never Vision's reading.
"""
from __future__ import annotations

import re
import unicodedata

import cv2
import numpy as np

from ..verify.text import _edit_distance
from .ink import binarize, ink_matte, normalize

PUNCT = set(".,;:!?।|-–—'\"()[]")


def tokenize(text: str) -> list[str]:
    """Words and punctuation marks as separate tokens; ZWNJ/ZWJ dropped (invisible)."""
    s = unicodedata.normalize("NFC", text).replace("‌", "").replace("‍", "")
    return re.findall(r"[^\s.,;:!?।|\-–—'\"()\[\]]+|[.,;:!?।|\-–—'\"()\[\]]", s)


def _dist(a: str, b: str) -> float:
    return _edit_distance(list(a), list(b)) / max(len(b), 1)


def align_tokens(vis: list[str], gt: list[str], gap: float = 0.8) -> list[tuple[int, list[int], float]]:
    """Global alignment of Vision tokens to ground-truth tokens. A ground-truth
    word may match one Vision token, or two that Vision split it into (or one
    Vision token may cover two ground-truth words, which is not used). Returns
    (gt index, vision indices, distance) for every matched ground-truth token."""
    n, m = len(gt), len(vis)
    INF = float("inf")
    D = np.full((n + 1, m + 1), INF)
    back = np.zeros((n + 1, m + 1, 2), int)
    D[0, :] = np.arange(m + 1) * gap
    D[:, 0] = np.arange(n + 1) * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            opts = [(D[i - 1, j] + gap, (1, 0)), (D[i, j - 1] + gap, (0, 1)),
                    (D[i - 1, j - 1] + min(1.0, _dist(vis[j - 1], gt[i - 1])), (1, 1))]
            if j >= 2:  # Vision split the word in two
                opts.append((D[i - 1, j - 2] + min(1.0, _dist(vis[j - 2] + vis[j - 1], gt[i - 1])) + 0.1, (1, 2)))
            D[i, j], back[i, j] = min(opts, key=lambda o: o[0])
    out, i, j = [], n, m
    while i > 0 and j > 0:
        di, dj = back[i, j]
        if di == 1 and dj >= 1:
            js = list(range(j - dj, j))
            out.append((i - 1, js, _dist("".join(vis[k] for k in js), gt[i - 1])))
        i, j = i - di, j - dj
    return out[::-1]


def prepare_page(img: np.ndarray, word_h: float, stroke_px: float = 0.0):
    """(normalised BGR, darkness, ink mask) for a page whose words are about
    `word_h` px tall. Long thin ruled lines are removed from the mask."""
    k = int(max(31, 0.5 * word_h)) | 1
    norm, dark = normalize(img, k)
    ink = binarize(dark)
    b = ink.astype(np.uint8)
    L = int(1.5 * word_h)
    horiz = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    vert = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2 * L)))
    d = max(3, int(stroke_px or word_h / 14))
    pad = np.pad(b, ((d, d), (d, d)))
    thin_h = (pad[: -2 * d, d:-d] == 0) | (pad[2 * d :, d:-d] == 0)    # no ink d px above or below
    thin_v = (pad[d:-d, : -2 * d] == 0) | (pad[d:-d, 2 * d :] == 0)
    rules = ((cv2.dilate(horiz, np.ones((5, 1), np.uint8)) > 0) & thin_h) | \
            ((cv2.dilate(vert, np.ones((1, 5), np.uint8)) > 0) & thin_v)
    return norm, dark, ink & ~rules


def crop_word(norm: np.ndarray, dark: np.ndarray, ink: np.ndarray, box, min_area: int = 12,
              inside: float = 0.5, margin: int = 3):
    """RGBA crop of the ink belonging to one word box: connected components
    with at least `inside` of their pixels in the box (padded a little, since
    Vision's boxes can clip vowel marks and vattus). None if no ink."""
    x, y, w, h = box
    H, W = ink.shape
    px, py = int(0.15 * h), int(0.4 * h)
    X0, Y0, X1, Y1 = max(0, x - px), max(0, y - py), min(W, x + w + px), min(H, y + h + py)
    region = ink[Y0:Y1, X0:X1].astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
    core = np.zeros_like(region)
    cx0, cy0 = max(0, x - X0 - int(0.04 * h)), max(0, y - Y0 - int(0.12 * h))
    core[cy0 : y - Y0 + h + int(0.12 * h), cx0 : x - X0 + w + int(0.04 * h)] = 1
    inside_px = np.bincount(labels.ravel(), weights=core.ravel(), minlength=n)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area and inside_px[i] >= inside * area:
            keep[i] = True
    mask = keep[labels]
    if mask.sum() < 50:
        return None
    ys, xs = np.nonzero(mask)
    y0, y1 = max(0, ys.min() - margin), min(mask.shape[0], ys.max() + 1 + margin)
    x0, x1 = max(0, xs.min() - margin), min(mask.shape[1], xs.max() + 1 + margin)
    support = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    alpha = (ink_matte(dark[Y0:Y1, X0:X1]) * support)[y0:y1, x0:x1]
    rgb = norm[Y0:Y1, X0:X1][y0:y1, x0:x1].copy()
    a8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    rgb[a8 == 0] = 255
    return np.dstack([rgb, a8]), (X0 + x0, Y0 + y0)
