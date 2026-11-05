"""Render one Telugu word: a real whole-word sample if the writer has it,
otherwise real akshara samples placed side by side on a shared main zone.

Telugu has no headline to bridge, so assembly is spacing and alignment only.
Backoffs when an akshara has no sample, in order:
- a trailing ం/ః with no sample of the whole akshara: the base akshara plus a
  drawn circle (ం) or pair of small circles (ః), as writers make them;
- a conjunct: its base letter (first consonant with the vowel sign) plus a
  real vattu sample for each following consonant, placed below where vattus
  sat in the source word;
- a consonant with a vowel sign nobody wrote on it (ఖా): the bare consonant
  (ఖ) with the sign cut from another letter that has it (కా, aligned on క);
- . , : / ? ! ( ) and quotes are drawn as pen strokes (the source pages have
  them, but as specks too small to cut reliably).

`jitter` (0-1) makes the hand less regular: every letter gets its own size,
tilt, lean and up/down bounce, and the strokes of the whole word a slight
wobble (whole-word samples get the wobble only). `gap_jitter` (0-1) makes the
space between letters uneven, with an occasional small break inside a word.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..text.telugu import ANUSVARA, VISARGA, ZW, aksharas, is_consonant, is_independent_vowel, split_cluster
from .assemble import STROKE_PUNCT, Glyph, MissingText, _resize, _stroke_punct, normalize_stroke
from .bank import SampleBank
from ..library import Sample
from .degrade import _smooth_noise

LETTER_GAP = 0.12  # natural space between handwritten Telugu aksharas, in body heights

# Where each vowel sign is written, for cutting it off one letter and putting it on
# another: "top" replaces the consonant's talakattu (the tick above the main zone),
# "right" extends past the consonant's body. ై and ృ are left out: their parts
# below the main zone cannot be told apart from the donor letter's own descenders.
SIGN_ZONES = {"ా": ("top", "right"), "ి": ("top",), "ీ": ("top",), "ు": ("right",), "ూ": ("right",),
              "ె": ("top",), "ే": ("top",), "ొ": ("top", "right"), "ో": ("top", "right"), "ౌ": ("top", "right")}


@dataclass
class Composed:
    """A consonant + vowel sign with no sample: `base` (the bare consonant) with
    the sign cut from `donor` (another consonant + the same sign), located by
    comparing the donor with `donor_bare` (the donor's consonant on its own)."""
    base: Sample
    donor: Sample
    donor_bare: Sample
    sign: str

    @property
    def writer(self) -> str:
        return self.base.writer

    @property
    def features(self) -> dict:
        return self.base.features

    @property
    def id(self) -> str:
        return f"{self.base.id}+{self.donor.id}"


def _donors(bank: SampleBank) -> dict[str, list[str]]:
    """Vowel sign -> consonants that have a sample both bare and with that sign."""
    d = getattr(bank, "_te_donors", None)
    if d is None:
        d = {}
        for t in bank.aksharas:
            if len(t) == 2 and is_consonant(t[0]) and t[1] in SIGN_ZONES and bank.has_akshara(t[0]) and bank.has_akshara(t):
                d.setdefault(t[1], []).append(t[0])
        bank._te_donors = d
    return d


def _letter(ak: str, bank: SampleBank, writer: str, rng) -> Sample | Composed | None:
    """A sample of `ak`, or failing that a consonant + vowel sign composition."""
    s = bank.akshara(ak, writer, rng)
    if s is not None or len(ak) != 2 or not is_consonant(ak[0]) or ak[1] not in SIGN_ZONES:
        return s
    base = bank.akshara(ak[0], writer, rng)
    cands = [c for c in _donors(bank).get(ak[1], []) if c != ak[0]]
    if base is None or not cands:
        return None
    for c in rng.permutation(cands)[:8]:  # skip donors whose sign sits where the cut cannot find it
        donor = bank.akshara(c + ak[1], writer, rng)
        cp = Composed(base, donor, bank.akshara(c, donor.writer, rng), ak[1])
        if _composed_alpha(cp, bank, 40.0, 40.0 * bank.stroke_over_body[writer], 48.0)[2] > 2.0 * 40.0:
            return cp
    return None


def _composed_alpha(cp: Composed, bank: SampleBank, body: float, stroke: float,
                    head: float) -> tuple[np.ndarray, float, float]:
    """Draw a Composed letter on the shared main zone [head, head + body].
    Returns (alpha, top y in word coordinates, ink of the cut-out sign)."""
    def scaled(smp):
        f = smp.features
        k = body / bank.body.get(smp.writer, f["body_height"])
        a = normalize_stroke(_resize(bank.alpha(smp), k), f["stroke_width"] * k, stroke)
        return a, head + body / 2 - (f["headline_y"] + f["baseline_y"]) / 2 * k

    def ink_w(a):
        cols = np.nonzero((a > 0.2).any(axis=0))[0]
        return float(cols[-1] + 1) if len(cols) else float(a.shape[1])

    zones = SIGN_ZONES[cp.sign]
    top_cut = head - 0.1 * body
    (b, yb), (d, yd), (c, _) = scaled(cp.base), scaled(cp.donor), scaled(cp.donor_bare)
    b = b.copy()
    if "top" in zones:  # the sign takes the talakattu's place
        b[: max(0, int(round(top_cut - yb)))] = 0
    rows = yd + np.arange(d.shape[0])[:, None]
    cols = np.arange(d.shape[1])[None, :]
    keep = np.zeros(d.shape, bool)
    if "top" in zones:
        keep |= rows < top_cut
    if "right" in zones:
        keep |= (cols >= ink_w(c) - 0.05 * body) & (rows >= top_cut)
    d = np.where(keep, d, 0)
    dx = ink_w(b) - ink_w(c)  # the donor's consonant ends where the base's does
    x0, y0 = min(0.0, dx), min(yb, yd)
    W = int(np.ceil(max(b.shape[1], dx + d.shape[1]) - x0)) + 1
    H = int(np.ceil(max(yb + b.shape[0], yd + d.shape[0]) - y0)) + 1
    out = np.zeros((H, W), np.float32)
    for a, ax, ay in ((b, 0.0, yb), (d, dx, yd)):
        ix, iy = int(round(ax - x0)), int(round(ay - y0))
        region = out[iy : iy + a.shape[0], ix : ix + a.shape[1]]
        np.maximum(region, a[: region.shape[0], : region.shape[1]], out=region)
    return out, y0, float(d.sum())


def is_letter(ak: str) -> bool:
    return any(is_consonant(c) or is_independent_vowel(c) for c in ak)


def _clean(text: str) -> str:
    return "".join(c for c in text if c not in ZW)


def render_word(text: str, bank: SampleBank, writer: str, rng: np.random.Generator,
                akshara_gap: float = 0.0, jitter: float = 0.0, gap_jitter: float = 0.0) -> Glyph:
    text = _clean(text)
    # Numbers are always assembled, so digits get the same size rule everywhere.
    s = bank.word(text, writer, rng) if any(is_letter(c) for c in text) else None
    if s is not None:
        f = s.features
        body = bank.body.get(s.writer, s.features["body_height"])
        mid = (f["headline_y"] + f["baseline_y"]) / 2
        target = bank.stroke_over_body[writer] * body
        a = normalize_stroke(bank.alpha(s), f["stroke_width"], target)
        if jitter:
            a = _wobble(a, 0.05 * jitter * body, 0.7 * body, rng)
        return Glyph(a, mid - body / 2, mid + body / 2, body, target,
                     [{"text": text, "kind": "word", "sample": s.id, "writer": s.writer}])
    g = _assemble(text, bank, writer, rng, akshara_gap, jitter, gap_jitter)
    if jitter:
        g.alpha = _wobble(g.alpha, 0.05 * jitter * g.body, 0.7 * g.body, rng)
    return g


def _wobble(a: np.ndarray, amp: float, cell: float, rng) -> np.ndarray:
    """Smooth random displacement of the strokes (an unsteady hand)."""
    if amp < 0.3:
        return a
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (_smooth_noise((h, w), cell, rng) - 0.5) * 2 * amp
    dy = (_smooth_noise((h, w), cell, rng) - 0.5) * 2 * amp
    return cv2.remap(a, xx + dx, yy + dy, cv2.INTER_LINEAR, borderValue=0)


def _letter_jitter(a: np.ndarray, y0: float, jitter: float, body: float, rng) -> tuple[np.ndarray, float]:
    """One letter written a little bigger or smaller, tilted, leaning and
    off the line. Returns (alpha trimmed to its ink columns, new top y)."""
    k = float(np.exp(rng.normal(0, 0.1 * jitter)))
    rot = rng.normal(0, 5 * jitter)
    lean = np.tan(np.radians(rng.normal(0, 7 * jitter)))
    h, w = a.shape
    cx, cy = w / 2, h / 2
    R = cv2.getRotationMatrix2D((cx, cy), rot, k)
    M = np.vstack([R, [0, 0, 1]]) @ np.array([[1, -lean, lean * cy], [0, 1, 0], [0, 0, 1]])
    pad = int(np.ceil(0.25 * max(h, w))) + 2
    M = M[:2] + [[0, 0, pad], [0, 0, pad]]
    out = cv2.warpAffine(a, M, (w + 2 * pad, h + 2 * pad), flags=cv2.INTER_LINEAR, borderValue=0)
    cols = np.nonzero((out > 0.05).any(axis=0))[0]
    x0 = int(cols[0]) if len(cols) else pad
    x1 = int(cols[-1]) + 1 if len(cols) else pad + w
    bounce = rng.normal(0, 0.08 * jitter) * body
    return out[:, x0:x1], y0 - pad + bounce


def _plan(ak: str, bank: SampleBank, writer: str, rng) -> dict:
    """How to write one akshara: its letter sample, drawn marks, vattu samples."""
    s = _letter(ak, bank, writer, rng)
    if s is not None:
        return {"ak": ak, "sample": s, "marks": [], "vattus": []}
    if ak in STROKE_PUNCT:
        return {"ak": ak, "sample": None, "marks": [], "vattus": []}
    marks = [c for c in ak if c in (ANUSVARA, VISARGA)]
    base = "".join(c for c in ak if c not in (ANUSVARA, VISARGA))
    if marks and base:
        s = _letter(base, bank, writer, rng)
        if s is not None:
            return {"ak": ak, "sample": s, "marks": marks, "vattus": []}
    split = split_cluster(base)
    if split:
        letter, rest = split
        vs = [bank.vattu(c, writer, rng) for c in rest]
        s = _letter(letter, bank, writer, rng)
        if s is not None and all(vs):
            return {"ak": ak, "sample": s, "marks": marks, "vattus": vs}
    raise MissingText(ak)


def can_write(text: str, bank: SampleBank) -> bool:
    """True if every word has a sample or every akshara a sample or backoff."""
    rng = np.random.default_rng(0)
    writer = bank.writers[0]
    for word in _clean(text).split():
        if bank.word(word, writer, rng) is not None:
            continue
        try:
            for ak in aksharas(word):
                if not ak.isspace():
                    _plan(ak, bank, writer, rng)
        except MissingText:
            return False
    return True


def _circle(body: float, stroke: float, d: float) -> np.ndarray:
    S = 4
    r = max(1.0, d * body / 2)
    size = int(2 * r + 2 * stroke + 2)
    big = np.zeros((size * S, size * S), np.uint8)
    cv2.circle(big, (size * S // 2, size * S // 2), int(r * S), 255, max(1, int(round(stroke * S))), cv2.LINE_AA)
    return cv2.resize(big, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0 * 0.9


def _assemble(text: str, bank: SampleBank, writer: str, rng: np.random.Generator, gap: float,
              jitter: float = 0.0, gap_jitter: float = 0.0) -> Glyph:
    plans = [_plan(ak, bank, writer, rng) for ak in aksharas(text) if not ak.isspace()]

    # Common scale: the median body height of the letter samples.
    bodies = [bank.body.get(p["sample"].writer, p["sample"].features["body_height"]) for p in plans if p["sample"] and is_letter(p["ak"])]
    body = float(np.median(bodies)) if bodies else 60.0
    stroke = bank.stroke_over_body[writer] * body
    head = 1.2 * body           # room for vowel marks above the main zone
    base_y = head + body

    placed = []                 # (alpha, x, y, part)
    x = 0.0
    for p in plans:
        s = p["sample"]
        part = {"text": p["ak"], "kind": "akshara" if s else "stroke",
                "sample": s.id if s else None, "writer": s.writer if s else None}
        if s is None:
            a, y0 = _stroke_punct(p["ak"], body, stroke, head, base_y)
        elif isinstance(s, Composed):
            a, y0, _ = _composed_alpha(s, bank, body, stroke, head)
            part.update(kind="composed", sample=[s.base.id, s.donor.id])
            if jitter and not p["vattus"]:
                a, y0 = _letter_jitter(a, y0, jitter, body, rng)
        else:
            a = bank.alpha(s)
            f = s.features
            if is_letter(p["ak"]):
                # Scale by the writer's letter size; centre the sample's main zone.
                k = body / bank.body.get(s.writer, s.features["body_height"])
                a = normalize_stroke(_resize(a, k), f["stroke_width"] * k, stroke)
                y0 = head + body / 2 - (f["headline_y"] + f["baseline_y"]) / 2 * k
                if jitter and not p["vattus"]:  # a vattu is placed against the letter's original shape
                    a, y0 = _letter_jitter(a, y0, jitter, body, rng)
            else:  # digits: no main zone of their own; a little taller than the letters
                rows = np.nonzero((a > 0.35).any(axis=1))[0]
                ink_h = max(rows[-1] - rows[0] + 1, 1) if len(rows) else a.shape[0]
                k = 1.1 * body / ink_h
                a = _resize(a, k)
                y0 = base_y - ((rows[-1] + 1) * k if len(rows) else a.shape[0])
        placed.append((a, x, y0, part))
        right = x + a.shape[1]
        for v in p["vattus"]:
            vf = v.features
            k = body / bank.body.get(v.writer, vf["body_height"])
            va = normalize_stroke(_resize(bank.alpha(v), k), vf["stroke_width"] * k, stroke)
            # Same offset from the letter as in the source word, scaled to this letter's width.
            sx = a.shape[1] / max(vf["letter_w_body"] * body, 1.0)
            placed.append((va, x + vf["dx_body"] * body * sx, base_y + vf["dy_body"] * body,
                           {"text": "్" + v.text, "kind": "vattu", "sample": v.id, "writer": v.writer}))
            right = max(right, x + vf["dx_body"] * body * sx + va.shape[1])
            part["kind"] = "akshara+vattu"
        for m in p["marks"]:
            c = _circle(body, stroke, 0.5 if m == ANUSVARA else 0.22)
            if m == ANUSVARA:
                cx = right + 0.08 * body
                placed.append((c, cx, head + 0.5 * body - c.shape[0] / 2, {"text": m, "kind": "stroke"}))
                right = cx + c.shape[1]
            else:
                cx = right + 0.06 * body
                placed.append((c, cx, head + 0.05 * body, {"text": m, "kind": "stroke"}))
                placed.append((c, cx, head + 0.55 * body, {"text": "", "kind": "stroke"}))
                right = cx + c.shape[1]
            part["mark"] = m
        step = LETTER_GAP + gap + rng.normal(0, 0.03)
        if gap_jitter:  # uneven spacing; now and then the pen lifts for a small break mid-word
            step += rng.normal(0, 0.1 * gap_jitter) + (0.18 if rng.random() < 0.08 * gap_jitter else 0.0)
            step = max(step, -0.12)
        x = right + step * body

    W = int(np.ceil(max(px + a.shape[1] for a, px, _, _ in placed))) + 2
    y_shift = -min([py for _, _, py, _ in placed] + [0.0])
    H = int(np.ceil(max(py + a.shape[0] for a, _, py, _ in placed) + y_shift)) + 2
    canvas = np.zeros((H, W), np.float32)
    for a, px, py, _ in placed:
        ix, iy = int(round(px)), int(round(py + y_shift))
        region = canvas[iy : iy + a.shape[0], ix : ix + a.shape[1]]
        np.maximum(region, a[: region.shape[0], : region.shape[1]], out=region)
    parts = [pt for *_, pt in placed if pt.get("text")]
    return Glyph(canvas, head + y_shift, base_y + y_shift, body, stroke, parts)
