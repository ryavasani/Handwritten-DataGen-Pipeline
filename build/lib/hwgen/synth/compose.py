"""Composite handwriting onto a page."""
from __future__ import annotations

import numpy as np


def overlay(page: np.ndarray, alpha: np.ndarray, ink_bgr, x: int, y: int) -> tuple[int, int, int, int]:
    """Multiply-blend ink onto a BGR page in place (ink darkens paper and keeps
    its texture, like real ballpoint). Returns the written region (x, y, w, h)."""
    H, W = page.shape[:2]
    h, w = alpha.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return (x0, y0, 0, 0)
    a = alpha[y0 - y : y1 - y, x0 - x : x1 - x, None]
    ink = np.asarray(ink_bgr, np.float32) / 255.0
    region = page[y0:y1, x0:x1].astype(np.float32)
    page[y0:y1, x0:x1] = np.clip(region * (1 - a + a * ink), 0, 255).astype(np.uint8)
    return (x0, y0, x1 - x0, y1 - y0)
