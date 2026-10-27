"""Lay words out into a handwritten line."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .assemble import Glyph, render_word
from .bank import SampleBank
from .style import Style


@dataclass
class LineRender:
    alpha: np.ndarray                 # float32 ink matte of the whole line
    headline_y: float                 # y of the headline at x = 0 (line coordinates)
    baseline_y: float
    words: list[dict] = field(default_factory=list)  # text, box (x, y, w, h), parts


def _transform(g: Glyph, scale: float, slant_deg: float, rot_deg: float) -> tuple[np.ndarray, float, float]:
    """Scale, shear (slant about the baseline) and rotate a glyph.
    Returns (alpha, headline_y, baseline_y) in the transformed image."""
    h, w = g.alpha.shape
    t = np.tan(np.radians(slant_deg))
    # Scale, then shear x by -t*(y - baseline): + slant leans the tops right.
    A = np.array([[scale, -t * scale, t * scale * g.baseline_y], [0, scale, 0]], np.float64)
    R = cv2.getRotationMatrix2D((w * scale / 2, g.baseline_y * scale), rot_deg, 1.0)
    M = (np.vstack([R, [0, 0, 1]]) @ np.vstack([A, [0, 0, 1]]))[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64) @ M.T
    x0, y0 = np.floor(corners.min(axis=0)).astype(int)
    x1, y1 = np.ceil(corners.max(axis=0)).astype(int)
    M[:, 2] -= (x0, y0)
    out = cv2.warpAffine(g.alpha, M, (int(x1 - x0) + 1, int(y1 - y0) + 1), flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    cx = w / 2
    head = (M @ [cx, g.headline_y, 1])[1]
    base = (M @ [cx, g.baseline_y, 1])[1]
    return out, float(head), float(base)


@dataclass
class PreparedWord:
    text: str
    glyph: Glyph
    alpha: np.ndarray      # scaled, slanted and rotated
    head: float            # headline y within `alpha`

    @property
    def width(self) -> int:
        return self.alpha.shape[1]


def prepare_words(text: str, bank: SampleBank, style: Style, rng: np.random.Generator,
                  body_px: float | None = None) -> list[PreparedWord]:
    """Render and transform every word once; layout and wrapping reuse these."""
    body = body_px or style.body_px
    out = []
    for word in text.split():
        g = render_word(word, bank, style.writer, rng, style.akshara_gap)
        scale = body / g.body * (1 + rng.normal(0, style.size_jitter))
        a, head, _ = _transform(g, scale, style.extra_slant, rng.normal(0, style.rotation_jitter))
        out.append(PreparedWord(word, g, a, head))
    return out


def wrap(words: list[PreparedWord], max_width: float, style: Style, body_px: float | None = None) -> list[list[PreparedWord]]:
    """Greedy line breaking on the real rendered word widths."""
    gap = style.word_gap * (body_px or style.body_px)
    lines, cur, w = [], [], 0.0
    for pw in words:
        add = pw.width + (gap if cur else 0)
        if cur and w + add > max_width:
            lines.append(cur)
            cur, w = [], 0.0
            add = pw.width
        cur.append(pw)
        w += add
    if cur:
        lines.append(cur)
    return lines


def layout_line(words: list[PreparedWord], style: Style, rng: np.random.Generator,
                body_px: float | None = None) -> LineRender:
    body = body_px or style.body_px
    placed = []
    x = 0.0
    phase = rng.uniform(0, 2 * np.pi)
    for pw in words:
        # Headline height of the line at this x: slow wave plus overall tilt.
        drift = style.baseline_drift * body * np.sin(2 * np.pi * x / (style.drift_period * body) + phase)
        tilt = -np.tan(np.radians(style.line_angle)) * x
        placed.append((pw, x, drift + tilt - pw.head))
        x += pw.width + style.word_gap * body * rng.uniform(0.75, 1.3)
    if not placed:
        return LineRender(np.zeros((1, 1), np.float32), 0.0, body)
    top = min(y for *_, y in placed)
    W = int(np.ceil(max(px + pw.width for pw, px, _ in placed))) + 1
    H = int(np.ceil(max(py + pw.alpha.shape[0] for pw, _, py in placed) - top)) + 1
    canvas = np.zeros((H, W), np.float32)
    out_words = []
    for pw, px, py in placed:
        a = pw.alpha
        ix, iy = int(round(px)), int(round(py - top))
        region = canvas[iy : iy + a.shape[0], ix : ix + a.shape[1]]
        np.maximum(region, a[: region.shape[0], : region.shape[1]], out=region)
        ys, xs = np.nonzero(a > 0.2)
        box = (ix + int(xs.min()), iy + int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1) if len(xs) else (ix, iy, 0, 0)
        out_words.append({"text": pw.text, "box": box, "parts": pw.glyph.parts})
    return LineRender(canvas, -top, -top + body, out_words)
