"""Ink/paper separation: background normalisation, soft ink matte, rule-line removal."""
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


def rule_line_mask(binary: np.ndarray, min_len: int = 40, max_thickness: int = 3) -> np.ndarray:
    """Pixels belonging to thin horizontal rules or dotted leaders ("........")."""
    return _solid_rules(binary, min_len, max_thickness) | _dotted_leaders(binary)


def _solid_rules(binary: np.ndarray, min_len: int, max_thickness: int) -> np.ndarray:
    """Long unbroken horizontal runs. Only pixels with no ink `max_thickness` px
    above or below are reported, so strokes crossing the rule survive."""
    b = binary.astype(np.uint8)
    lines = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1)))
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))) & b
    d = max_thickness
    pad = np.pad(b, ((d, d), (0, 0)))
    thin = (pad[: -2 * d] == 0) | (pad[2 * d :] == 0)
    return (lines > 0) & thin


def _dotted_leaders(binary: np.ndarray, max_h: int = 3, min_run: int = 25) -> np.ndarray:
    """Flat components (dots, dashes, merged dot runs) that line up into a
    horizontal run. Built from those components alone, so handwriting is never
    bridged into a fake line."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    tiny = stats[:, cv2.CC_STAT_HEIGHT] <= max_h
    tiny[0] = False
    dots = tiny[labels].astype(np.uint8)
    run = cv2.morphologyEx(dots, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)))
    run = cv2.morphologyEx(run, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (min_run, 1)))
    leader = (dots > 0) & (cv2.dilate(run, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))) > 0)

    # Dots that touch a stroke are part of a bigger component and survive the
    # test above. Along each leader (extended past its ends), also remove ink
    # that is a thin horizontal sliver: no ink 2 px above and 2 px below.
    band = cv2.dilate(run, cv2.getStructuringElement(cv2.MORPH_RECT, (41, 3))) > 0
    b = binary.astype(np.uint8)
    pad = np.pad(b, ((2, 2), (0, 0)))
    sliver = (pad[:-4] == 0) & (pad[4:] == 0)
    return leader | (band & binary & sliver)
