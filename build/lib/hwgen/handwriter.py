"""Public entry point: write any Hindi or Telugu text as handwriting, for use in
another document-generation pipeline.

    hw = Handwriter("data/library_te")
    hand = hw.write("text to write", height=20, max_width=900, seed=0)
    boxes = hand.paste(page, x=100, y=200)        # page: BGR uint8, changed in place
    page, boxes, layers = degrade(page, boxes, level=0.8, seed=0)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .library import Library
from .synth.assemble import MissingText
from .synth.bank import SampleBank
from .synth.compose import overlay
from .synth.degrade import apply_ink
from .synth.degrade_page import degrade_page
from .synth.layout import PreparedWord, prepare_words
from .synth.style import Style, sample_style


@dataclass
class Handwriting:
    ink: np.ndarray                 # float32 ink matte in [0, 1], 0 = no ink
    color: tuple[int, int, int]     # ink colour (BGR)
    words: list[dict]               # {"text", "line", "box": [x, y, w, h]} in `ink` coordinates
    skipped: list[str] = field(default_factory=list)  # words the library cannot write; left out of `ink`
    style: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        """What was actually written (the label for `ink`), one line per line."""
        lines: dict[int, list[str]] = {}
        for w in self.words:
            lines.setdefault(w["line"], []).append(w["text"])
        return "\n".join(" ".join(ws) for _, ws in sorted(lines.items()))

    def paste(self, page: np.ndarray, x: int, y: int) -> list[dict]:
        """Blend the ink onto a BGR page in place (ink darkens the paper and keeps
        its texture). Returns the words with boxes in page coordinates."""
        overlay(page, self.ink, self.color, int(x), int(y))
        return [{**w, "box": [w["box"][0] + int(x), w["box"][1] + int(y), *w["box"][2:]]} for w in self.words]

    def image(self, background: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """The handwriting alone on a plain BGR background."""
        page = np.empty((*self.ink.shape, 3), np.uint8)
        page[:] = background
        self.paste(page, 0, 0)
        return page


class Handwriter:
    """Loads a sample library once; `write` then renders any text in it."""

    def __init__(self, library: str | Path):
        self.bank = SampleBank(Library(Path(library)))
        self.script = "te" if self.bank.script == "telu" else "hi"

    @property
    def writers(self) -> list[str]:
        return list(self.bank.writers)

    def write(self, text: str, height: float = 20, max_width: float | None = None, seed=None,
              writer: str | None = None, irregular: float = 0.0, spacing: float = 0.0,
              line_spacing: float | None = None, color: tuple[int, int, int] | None = None) -> Handwriting:
        """Write `text` in one person's hand.

        height:       letter height (main body) in px; ink reaches about 1.5x above and below
        max_width:    wrap lines to this width in px; newlines in `text` always break
        seed:         int or numpy Generator; the same seed gives the same handwriting
        writer:       a writer id from `writers`; random if unset
        irregular:    0-1, slant changing from word to word (Telugu: also per-letter size, tilt, bounce)
        spacing:      0-1, uneven gaps between words and letters
        line_spacing: baseline-to-baseline distance in letter heights (default 2.6 Hindi, 3.2 Telugu)
        color:        ink colour (BGR); a blue or black ballpoint if unset
        """
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        st = sample_style(rng, self.bank.writers, body_px=height)
        st.word_gap *= 0.8  # the same closer word spacing as the document generator
        st.slant_jitter, st.letter_jitter, st.spacing_jitter = 8.0 * irregular, float(irregular), float(spacing)
        if writer is not None:
            st.writer = writer
        if color is not None:
            st.ink_bgr = tuple(int(c) for c in color)
        pitch = (line_spacing or (3.2 if self.script == "te" else 2.6)) * height

        lines, skipped = [], []
        for para in text.split("\n"):
            words = []
            for w in para.split():
                try:
                    words += prepare_words(w, self.bank, st, rng)
                except MissingText:
                    skipped.append(w)
            lines += self._wrap(words, st, max_width) if words else [[]]

        placed = []  # (word, x, y of its alpha's top, line number)
        for n, line in enumerate(lines):
            x, phase = 0.0, rng.uniform(0, 2 * np.pi)
            for pw in line:
                drift = st.baseline_drift * height * np.sin(2 * np.pi * x / (st.drift_period * height) + phase)
                placed.append((pw, x, n * pitch + drift - pw.head, n))
                x += pw.width + self._gap(st, rng) * height
        if not placed:
            return Handwriting(np.zeros((1, 1), np.float32), st.ink_bgr, [], skipped, st.to_dict())

        top = min(y for _, _, y, _ in placed)
        W = int(np.ceil(max(x + pw.width for pw, x, _, _ in placed))) + 2
        H = int(np.ceil(max(y + pw.alpha.shape[0] for pw, _, y, _ in placed) - top)) + 2
        ink = np.zeros((H, W), np.float32)
        words = []
        for pw, x, y, n in placed:
            ix, iy = int(round(x)), int(round(y - top))
            region = ink[iy : iy + pw.alpha.shape[0], ix : ix + pw.alpha.shape[1]]
            np.maximum(region, pw.alpha[: region.shape[0], : region.shape[1]], out=region)
            ys, xs = np.nonzero(pw.alpha > 0.2)
            box = [ix + int(xs.min()), iy + int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1] if len(xs) else [ix, iy, 0, 0]
            words.append({"text": pw.text, "line": n, "box": box})
        ink = apply_ink(ink, st, rng, len(words))
        return Handwriting(ink, st.ink_bgr, words, skipped, st.to_dict())

    @staticmethod
    def _gap(st: Style, rng) -> float:
        gap = st.word_gap * rng.uniform(0.6, 1.4)
        if st.spacing_jitter:
            gap *= float(np.exp(rng.normal(0, 0.55 * st.spacing_jitter)))
        return max(-0.1, gap)

    def _wrap(self, words: list[PreparedWord], st: Style, max_width: float | None) -> list[list[PreparedWord]]:
        if not max_width:
            return [words]
        lines, cur, used = [], [], 0.0
        for pw in words:
            add = pw.width + (st.word_gap * st.body_px if cur else 0)
            if cur and used + add > max_width:
                lines.append(cur)
                cur, used, add = [], 0.0, pw.width
            cur.append(pw)
            used += add
        return lines + [cur]


def degrade(page: np.ndarray, boxes: list[dict] | None = None, level: float = 0.8,
            capture: str | None = None, seed=None) -> tuple[np.ndarray, list[dict], list[str]]:
    """Age, damage and re-photograph a finished page (smudges, water stains, tears,
    then a phone / flatbed / photocopy / fax / archive capture and WhatsApp
    compression). The capture warps the page, so pass the word boxes to get them
    moved to match. Returns (page, boxes, names of the layers applied)."""
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    out, Hm, layers = degrade_page(page.copy(), rng, level, capture)
    moved = []
    for b in boxes or []:
        bx, by, bw, bh = b["box"]
        c = cv2.perspectiveTransform(np.float32([[[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]]]), Hm)[0]
        x0, y0 = np.clip(c.min(0), 0, [out.shape[1] - 1, out.shape[0] - 1])
        x1, y1 = np.clip(c.max(0), 0, [out.shape[1] - 1, out.shape[0] - 1])
        moved.append({**b, "box": [int(x0), int(y0), int(np.ceil(x1 - x0)), int(np.ceil(y1 - y0))]})
    return out, moved, layers
