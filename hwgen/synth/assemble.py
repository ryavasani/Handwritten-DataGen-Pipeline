"""Render one word: a real whole-word sample if the writer has it, otherwise an
assembly of real akshara samples joined by a continuous headline."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..text.devanagari import VIRAMA, aksharas, is_consonant, is_independent_vowel
from .bank import SampleBank

MODIFIER_MARKS = {"ं": "dot", "ँ": "chandrabindu"}  # ं ँ
STROKE_PUNCT = {"/", ".", ",", ":"}  # no samples in the source data; drawn as pen strokes


class MissingText(Exception):
    """The bank cannot render this text (an akshara with no sample and no backoff)."""


@dataclass
class Glyph:
    alpha: np.ndarray        # float32 ink matte
    headline_y: float
    baseline_y: float
    body: float              # headline-to-baseline height, px
    stroke: float            # pen width, px
    parts: list[dict] = field(default_factory=list)  # provenance of every piece


def is_letter(ak: str) -> bool:
    return any(is_consonant(c) or is_independent_vowel(c) for c in ak)


def render_word(text: str, bank: SampleBank, writer: str, rng: np.random.Generator,
                akshara_gap: float = 0.0) -> Glyph:
    s = bank.word(text, writer, rng)
    if s is not None:
        f = s.features
        target = bank.stroke_over_body[writer] * f["body_height"]
        a = normalize_stroke(bank.alpha(s), f["stroke_width"], target)
        return Glyph(a, f["headline_y"], f["baseline_y"], f["body_height"], target,
                     [{"text": text, "kind": "word", "sample": s.id, "writer": s.writer}])
    return _assemble(text, bank, writer, rng, akshara_gap)


def _assemble(text: str, bank: SampleBank, writer: str, rng: np.random.Generator, gap: float) -> Glyph:
    # 1. Find a sample (or a drawing rule) for every akshara.
    pieces = []
    for ak in aksharas(text):
        if ak.isspace():
            continue
        s = bank.akshara(ak, writer, rng)
        if s is not None:
            pieces.append({"ak": ak, "sample": s, "mark": None})
            continue
        if ak in STROKE_PUNCT:
            pieces.append({"ak": ak, "sample": None, "mark": None})
            continue
        base = "".join(c for c in ak if c not in MODIFIER_MARKS)
        marks = [MODIFIER_MARKS[c] for c in ak if c in MODIFIER_MARKS]
        s = bank.akshara(base, writer, rng) if base and marks else None
        if s is not None:
            pieces.append({"ak": ak, "sample": s, "mark": marks[0]})
            continue
        split = _split_conjunct(ak, bank)
        if split is None:
            raise MissingText(ak)
        # Written with an explicit halant: full first consonant + ् + the rest.
        first, rest = split
        pieces.append({"ak": first + VIRAMA, "sample": bank.akshara(first, writer, rng), "mark": "halant"})
        pieces.append({"ak": rest, "sample": bank.akshara(rest, writer, rng), "mark": None})

    # 2. Common scale: the median body height of the letter samples.
    letter_bodies = [p["sample"].features["body_height"] for p in pieces if p["sample"] and is_letter(p["ak"])]
    body = float(np.median(letter_bodies)) if letter_bodies else 100.0
    stroke = bank.stroke_over_body[writer] * body  # every piece is brought to this pen width
    top_pad = int(0.9 * body)  # room for matras above the headline
    head = float(top_pad)
    base_y = head + body

    # 3. Scale and place each piece left to right.
    placed = []
    x = 0.0
    for p in pieces:
        s = p["sample"]
        if s is None:
            a, y0 = _stroke_punct(p["ak"], body, stroke, head, base_y)
        else:
            a = bank.alpha(s)
            f = s.features
            if is_letter(p["ak"]):
                k = body / f["body_height"]
                a = normalize_stroke(_resize(a, k), f["stroke_width"] * k, stroke)
                y0 = head - f["headline_y"] * k  # align this sample's headline
            else:
                # Digits, danda, hyphen: no headline; size by ink height, sit on the baseline.
                rows = np.nonzero((a > 0.35).any(axis=1))[0]
                ink_h = max(rows[-1] - rows[0] + 1, 1) if len(rows) else a.shape[0]
                target = (0.45 if p["ak"] == "-" else 1.15) * body
                k = target / ink_h
                a = _resize(a, k)
                if p["ak"] == "-":  # centred on the middle of the body
                    y0 = head + 0.5 * body - (rows[0] + rows[-1] + 1) / 2 * k if len(rows) else head
                else:
                    y0 = base_y - ((rows[-1] + 1) * k if len(rows) else a.shape[0])
        placed.append((p, a, x, y0))
        x += a.shape[1] + (gap + rng.normal(0, 0.02)) * body

    # 4. Composite on one canvas (max keeps overlapping strokes at full ink).
    W = int(np.ceil(max(px + a.shape[1] for _, a, px, _ in placed))) + 2
    ys = [py for _, _, _, py in placed] + [0.0]
    y_shift = -min(ys)
    H = int(np.ceil(max(py + a.shape[0] for _, a, _, py in placed) + y_shift)) + 2
    canvas = np.zeros((H, W), np.float32)
    for _, a, px, py in placed:
        ix, iy = int(round(px)), int(round(py + y_shift))
        h, w = a.shape
        region = canvas[iy : iy + h, ix : ix + w]
        np.maximum(region, a[: region.shape[0], : region.shape[1]], out=region)
    head += y_shift
    base_y += y_shift

    # 5. One continuous headline across the letters, plus any modifier marks.
    letters = [(px, a.shape[1]) for p, a, px, _ in placed if is_letter(p["ak"])]
    if letters:
        x0 = letters[0][0] + 0.15 * letters[0][1]
        x1 = letters[-1][0] + letters[-1][1]
        _draw_headline(canvas, x0, x1, head, stroke, rng)
    for p, a, px, _ in placed:
        if p["mark"] == "halant":
            _draw_mark(canvas, "halant", px + 0.6 * a.shape[1], base_y + 0.1 * body, stroke)
        elif p["mark"]:
            _draw_mark(canvas, p["mark"], px + 0.65 * a.shape[1], head - 0.35 * body, stroke)

    parts = [{"text": p["ak"], "kind": "akshara" if p["sample"] else "stroke",
              "sample": p["sample"].id if p["sample"] else None,
              "writer": p["sample"].writer if p["sample"] else None,
              **({"mark": p["mark"]} if p["mark"] else {})} for p, _, _, _ in placed]
    return Glyph(canvas, head, base_y, body, stroke, parts)


def normalize_stroke(a: np.ndarray, current: float, target: float) -> np.ndarray:
    """Thicken or thin strokes toward `target` pen width with grayscale
    dilation/erosion. Each pass changes the width by ~2 px, so only clear
    mismatches (from different pages, pens or a fallback writer) are corrected."""
    steps = int(round((target - current) / 2.0))
    if steps == 0:
        return a
    k = np.ones((3, 3), np.uint8)
    op = cv2.dilate if steps > 0 else cv2.erode
    return op(a, k, iterations=min(abs(steps), 2))


def _split_conjunct(ak: str, bank: SampleBank) -> tuple[str, str] | None:
    """'फ्ता' -> ('फ', 'ता') when both halves have samples."""
    i = ak.find(VIRAMA)
    if i <= 0 or i + 1 >= len(ak):
        return None
    first, rest = ak[:i], ak[i + 1 :]
    if bank.has_akshara(first) and bank.has_akshara(rest):
        return first, rest
    return None


def _resize(a: np.ndarray, k: float) -> np.ndarray:
    size = (max(1, round(a.shape[1] * k)), max(1, round(a.shape[0] * k)))
    return cv2.resize(a, size, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)


def _draw_headline(canvas: np.ndarray, x0: float, x1: float, y: float, stroke: float, rng) -> None:
    """Bridge the headline across the gaps between pieces. Columns that already
    have ink at headline height keep the sample's own headline, so the line is
    never doubled; elsewhere a slightly wavering pen line is drawn."""
    band = canvas[max(0, int(y - 1.5 * stroke)) : int(y + 1.5 * stroke) + 1]
    covered = (band > 0.4).any(axis=0)
    n = max(2, int((x1 - x0) / 15))
    xs = np.linspace(x0, x1, n)
    ys = y + np.cumsum(rng.normal(0, 0.06 * stroke, n)) + rng.normal(0, 0.15 * stroke)
    S = 4  # draw supersampled for smooth anti-aliased edges
    pts = np.stack([xs * S, ys * S], axis=1).astype(np.int32)
    big = np.zeros((canvas.shape[0] * S, canvas.shape[1] * S), np.uint8)
    cv2.polylines(big, [pts], False, 255, thickness=max(1, int(round(stroke * 0.9 * S))), lineType=cv2.LINE_AA)
    line = cv2.resize(big, canvas.shape[::-1], interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    # Keep the drawn line only in uncovered columns, feathered over a few pixels.
    keep = cv2.blur((~covered).astype(np.float32)[None, :], (max(3, int(2 * stroke)) | 1, 1))[0]
    np.maximum(canvas, line * 0.9 * keep[None, :], out=canvas)


def _draw_mark(canvas: np.ndarray, kind: str, x: float, y: float, stroke: float) -> None:
    S = 4
    big = np.zeros((canvas.shape[0] * S, canvas.shape[1] * S), np.uint8)
    r = max(1, int(round(stroke * 0.8 * S)))
    if kind == "dot":
        cv2.circle(big, (int(x * S), int(y * S)), r, 255, -1, cv2.LINE_AA)
    elif kind == "halant":  # short slanted tick under the consonant
        cv2.line(big, (int(x * S), int(y * S)), (int((x - 4 * stroke) * S), int((y + 3 * stroke) * S)),
                 255, max(1, r), cv2.LINE_AA)
    else:  # chandrabindu: a small cup with a dot
        cv2.ellipse(big, (int(x * S), int(y * S)), (5 * r, 3 * r), 0, 0, 180, 255, max(1, r), cv2.LINE_AA)
        cv2.circle(big, (int(x * S), int((y - 1.5 * stroke) * S)), r, 255, -1, cv2.LINE_AA)
    small = cv2.resize(big, canvas.shape[::-1], interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    np.maximum(canvas, small * 0.9, out=canvas)


def _stroke_punct(ch: str, body: float, stroke: float, head: float, base_y: float) -> tuple[np.ndarray, float]:
    """Punctuation the source data lacks, drawn as simple pen strokes.
    Returns (alpha, top y in word coordinates)."""
    S = 4
    w = {"/": 0.55, ".": 0.3, ",": 0.3, ":": 0.3}[ch] * body
    top = head - 0.2 * body
    h = base_y + 0.35 * body - top
    big = np.zeros((int(h * S) + 1, int(w * S) + 1), np.uint8)
    t = max(1, int(round(stroke * S)))
    r = max(1, int(round(stroke * 0.75 * S)))

    def pt(xf, yy):  # x as a fraction of width, y in word coordinates
        return int(xf * w * S), int((yy - top) * S)

    if ch == "/":
        cv2.line(big, pt(0.1, base_y + 0.1 * body), pt(0.9, head), 255, t, cv2.LINE_AA)
    elif ch == ".":
        cv2.circle(big, pt(0.5, base_y - stroke), r, 255, -1, cv2.LINE_AA)
    elif ch == ",":
        cv2.line(big, pt(0.6, base_y - stroke), pt(0.35, base_y + 0.25 * body), 255, t, cv2.LINE_AA)
    elif ch == ":":
        cv2.circle(big, pt(0.5, base_y - stroke), r, 255, -1, cv2.LINE_AA)
        cv2.circle(big, pt(0.5, base_y - 0.5 * body), r, 255, -1, cv2.LINE_AA)
    a = cv2.resize(big, (max(1, int(w)), max(1, int(h))), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return a * 0.9, top
