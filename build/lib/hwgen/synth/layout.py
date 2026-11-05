"""Render, scale, slant and rotate the words of a handwritten text."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import assemble_te
from .assemble import Glyph, render_word
from .bank import SampleBank
from .style import Style


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
    render = assemble_te.render_word if bank.script == "telu" else render_word
    out = []
    extra = {"jitter": style.letter_jitter, "gap_jitter": style.spacing_jitter} if bank.script == "telu" else {}
    for word in text.split():
        g = render(word, bank, style.writer, rng, style.akshara_gap, **extra)
        scale = body / g.body * (1 + rng.normal(0, style.size_jitter))
        slant = style.extra_slant + (rng.normal(0, style.slant_jitter) if style.slant_jitter else 0.0)
        a, head, _ = _transform(g, scale, slant, rng.normal(0, style.rotation_jitter))
        out.append(PreparedWord(word, g, a, head))
    return out

