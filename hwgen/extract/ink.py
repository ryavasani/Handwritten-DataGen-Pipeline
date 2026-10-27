"""Ink/paper separation: background normalisation, soft ink matte, binarisation."""
from __future__ import annotations

import cv2
import numpy as np


def estimate_background(img: np.ndarray, kernel: int = 31) -> np.ndarray:
    """Paper colour per pixel. Thin strokes vanish under a large dilate (max
    filter on a light background); the blur then smooths blocky artifacts."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    bg = cv2.dilate(img, k)
    bg = cv2.blur(bg, (kernel, kernel))
    return np.maximum(bg, img)


def normalize(img: np.ndarray, kernel: int = 31) -> tuple[np.ndarray, np.ndarray]:
    """Divide out the paper so it becomes pure white. Returns (normalised image
    as uint8 with paper=255, darkness in [0,1] from the grey channel)."""
    bg = estimate_background(img, kernel).astype(np.float32)
    norm = np.clip(img.astype(np.float32) / np.maximum(bg, 1.0), 0.0, 1.0)
    gray = norm if norm.ndim == 2 else norm.min(axis=2)  # min keeps coloured ink dark
    return (norm * 255).astype(np.uint8), 1.0 - gray


def ink_matte(darkness: np.ndarray, lo: float = 0.12, hi: float = 0.55) -> np.ndarray:
    """Soft alpha: 0 for paper and scanner noise, 1 for saturated ink, linear between."""
    return np.clip((darkness - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def binarize(darkness: np.ndarray, floor: float = 0.18) -> np.ndarray:
    """Otsu on darkness, but never below `floor` so that near-blank regions do
    not turn paper texture into ink."""
    d8 = (np.clip(darkness, 0, 1) * 255).astype(np.uint8)
    t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(t, int(floor * 255))
    return d8 > t
