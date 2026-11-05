"""Generate a plain handwritten Hindi or Telugu legal document (petition,
application, affidavit) in the cramped, hurried style of real Indian court
papers. The script follows the sample bank (`bank.script`).

What makes it messy, all scaled by one `messiness` knob:
- cramped: small word gaps (Devanagari headlines often run on from word to word),
  tight line pitch so matras of neighbouring lines touch, words squeezed at the
  right margin instead of wrapped;
- unsteady: a slight per-line tilt, baseline wave, size and rotation jitter, and a
  smooth elastic wobble of the strokes;
- pen: a broad nib (angled stroke widening), ink that fades between dips and
  goes dark again after a refill, feathering into the paper;
- corrections: started-and-struck words, a vertical note in the margin;
- paper: either aged court paper (yellowed, stained, folded, show-through)
  written with a dip/fountain pen, or modern notebook paper (plain or ruled,
  writing sitting on the rules) written with a ballpoint; then photographed.

Ground truth stays exact: every written word is in the annotation with its box;
struck words are marked `struck` and excluded from the line text.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..text import devanagari, telugu
from . import legal_text, legal_text_te
from .assemble import MissingText
from .assemble_te import can_write
from .bank import SampleBank
from .compose import overlay
from .degrade import _smooth_noise, apply_ink
from .layout import PreparedWord, prepare_words
from .degrade_page import degrade_page
from .paper import aged_paper, folds, photograph, ruled_lines, show_through
from .style import Style, sample_style

PAGE = (1754, 1240)  # A4 at 150 dpi (h, w)

# Dip-pen and fountain-pen inks (BGR): blue, blue-black, iron-gall brown, black.
DOC_INKS = [(150, 75, 40), (125, 60, 35), (95, 55, 40), (45, 65, 100), (35, 55, 85), (40, 35, 35)]


@dataclass
class Messy:
    """Document-level parameters on top of the per-writer Style."""
    m: float               # overall messiness 0..1
    pitch: float           # headline-to-headline line spacing, in body heights
    page_tilt: float       # deg; lines drift up or down the same way
    tilt_jitter: float     # deg, per line
    join_prob: float       # chance a small gap is bridged by the headline
    elastic: float         # stroke wobble amplitude, in body heights
    strike_rate: float     # struck false starts per word
    squeeze: float         # max horizontal squeeze of a word at the margin
    nib: int               # nib length in px (0 = ballpoint-like)
    nib_angle: float       # deg
    dip_words: float       # words written per dip
    dip_fade: float        # how far the ink fades before the next dip
    paper: str             # "aged" or "notebook"
    ruled: bool            # notebook rules to write on


def sample_messy(rng: np.random.Generator, writers: list[str], m: float, paper: str,
                 script: str = "deva") -> tuple[Style, Messy]:
    st = sample_style(rng, writers, body_px=rng.uniform(15, 21))
    te = script == "telu"
    lerp = lambda a, b: a + (b - a) * m  # noqa: E731  (tidy -> messy)
    ballpoint = paper == "notebook"
    if not ballpoint:  # sample_style already picked a ballpoint ink
        st.ink_bgr = tuple(int(c) for c in DOC_INKS[rng.integers(len(DOC_INKS))])
    st.word_gap = float(lerp(rng.uniform(0.48, 0.8), rng.uniform(0.06, 0.24)))
    st.akshara_gap = float(lerp(rng.uniform(-0.04, 0.03), rng.uniform(-0.12, -0.03)))
    if te:  # Telugu letters stand apart; a hurried hand just packs them closer
        st.akshara_gap = float(lerp(rng.uniform(0.0, 0.08), rng.uniform(-0.1, -0.02)))
        st.word_gap = float(lerp(rng.uniform(0.56, 0.88), rng.uniform(0.2, 0.36)))
    st.size_jitter += 0.06 * m
    st.rotation_jitter += 1.0 * m
    st.extra_slant = float(rng.normal(0, 3 + 6 * m))
    st.baseline_drift += 0.1 * m
    st.pressure = float(rng.uniform(0.1, 0.35))
    st.fade = float(rng.uniform(0.0, 0.3) * (0.4 if ballpoint else 1))
    st.blot_rate = float((rng.uniform(0.0, 0.03) + 0.03 * m) * (0.3 if ballpoint else 1))
    ruled = ballpoint and rng.random() < 0.6
    mx = Messy(
        m=m,
        # Telugu needs more room: vowel marks above and vattus below the main zone.
        pitch=float(lerp(rng.uniform(2.6, 3.2), rng.uniform(2.0, 2.45)) * (1.25 if te else 1)),
        page_tilt=float(rng.normal(0, 0.15 + 0.35 * m) * (0.4 if ruled else 1)),
        tilt_jitter=float((0.08 + 0.25 * m) * (0.5 if ruled else 1)),
        join_prob=0.0 if te else float(m * rng.uniform(0.3, 0.9)),
        elastic=float(m * rng.uniform(0.05, 0.12)),
        strike_rate=float(m * rng.uniform(0.0, 0.02)),
        squeeze=float(lerp(0.9, 0.55)),
        nib=0 if ballpoint else int(round(rng.uniform(0, 1) * (1 + 1.5 * m) * st.body_px / 20)),
        nib_angle=float(rng.uniform(20, 60)),
        dip_words=float(rng.uniform(40, 80) if ballpoint else rng.uniform(8, 30)),
        dip_fade=float(rng.uniform(0.03, 0.1) if ballpoint else rng.uniform(0.15, 0.3 + 0.3 * m)),
        paper=paper,
        ruled=ruled,
    )
    return st, mx


# ---------------------------------------------------------------- pen drawing

def _pen(canvas: np.ndarray, pts, thickness: float, value: float = 0.9) -> None:
    """Anti-aliased polyline drawn into a float matte (supersampled)."""
    S = 4
    big = np.zeros((canvas.shape[0] * S, canvas.shape[1] * S), np.uint8)
    p = (np.asarray(pts, np.float64) * S).astype(np.int32)
    cv2.polylines(big, [p], False, 255, max(1, int(round(thickness * S))), cv2.LINE_AA)
    small = cv2.resize(big, canvas.shape[::-1], interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    np.maximum(canvas, small * value, out=canvas)


def _wavy(x0, x1, y, amp, rng, n=None):
    n = n or max(3, int((x1 - x0) / 6))
    xs = np.linspace(x0, x1, n)
    return np.stack([xs, y + np.cumsum(rng.normal(0, amp, n)) * 0.5 + rng.normal(0, amp, n) * 0.3], 1)


# ---------------------------------------------------------------- words and lines

@dataclass
class Word:
    pw: PreparedWord
    struck: bool = False


def _prepare_sentences(sentences: list[str], bank: SampleBank, st: Style, mx: Messy, rng,
                       body: float) -> list[list[Word]]:
    """Render each sentence's words; a sentence the bank cannot write is
    dropped. Before some words the writer starts, stops and strikes the false start."""
    aksharas = telugu.aksharas if bank.script == "telu" else devanagari.aksharas
    out = []
    for sent in sentences:
        try:
            pws = prepare_words(sent, bank, st, rng, body)
        except MissingText:
            continue
        words: list[Word] = []
        for pw in pws:
            aks = [a for a in aksharas(pw.text) if not a.isspace()]
            if len(aks) >= 3 and rng.random() < mx.strike_rate:
                try:
                    start = "".join(aks[: int(rng.integers(1, len(aks) - 1))])
                    words.append(Word(prepare_words(start, bank, st, rng, body)[0], struck=True))
                except MissingText:
                    pass
            words.append(Word(pw))
        out.append(words)
    return out


def _squeeze(w: Word, width: float) -> Word:
    a = w.pw.alpha
    a = cv2.resize(a, (max(1, int(width)), max(1, int(a.shape[0] * 0.92))), interpolation=cv2.INTER_AREA)
    return Word(PreparedWord(w.pw.text, w.pw.glyph, a, w.pw.head * 0.92), w.struck)


def _wrap(words: list[Word], first_width: float, width: float, gap: float, mx: Messy, rng) -> list[list[Word]]:
    """Greedy breaking; a word that only just overflows is squeezed into the
    end of the line, as hurried writers do, instead of moving down."""
    lines, cur, used, limit = [], [], 0.0, first_width
    for w in words:
        add = w.pw.width + (gap if cur else 0)
        if cur and used + add > limit:
            room = limit - used - gap
            if room >= mx.squeeze * w.pw.width and rng.random() < 0.3 + 0.6 * mx.m:
                cur.append(_squeeze(w, room))
                w = None
            lines.append(cur)
            cur, used, limit = [], 0.0, width
            if w is None:
                continue
            add = w.pw.width
        cur.append(w)
        used += add
    if cur:
        lines.append(cur)
    return lines


@dataclass
class Line:
    alpha: np.ndarray
    head: float            # headline y at the left end
    words: list[dict]      # text, box (x, y, w, h) in line coords, parts, struck


def _render_line(words: list[Word], st: Style, mx: Messy, rng, body: float,
                 ink_state: dict, join_prob: float | None = None) -> Line:
    join_prob = mx.join_prob if join_prob is None else join_prob
    angle = mx.page_tilt + rng.normal(0, mx.tilt_jitter)
    # Cap the rise over the whole line so a tilted line touches, but never sits on, its neighbour.
    total = sum(w.pw.width for w in words) * (1 + st.word_gap * 0.3)
    max_deg = np.degrees(np.arctan(0.4 * body / max(total, 1.0)))
    angle = float(np.clip(angle, -max_deg, max_deg))
    phase = rng.uniform(0, 2 * np.pi)
    placed, x = [], 0.0
    for w in words:
        drift = st.baseline_drift * body * np.sin(2 * np.pi * x / (st.drift_period * body) + phase)
        y = drift - np.tan(np.radians(angle)) * x - w.pw.head
        placed.append([w, x, y])
        gap = st.word_gap * rng.uniform(0.6, 1.4)
        if st.spacing_jitter:  # uneven hand: some words nearly touch, some stand far apart
            gap *= float(np.exp(rng.normal(0, 0.55 * st.spacing_jitter)))
        x += w.pw.width + max(-0.1, gap) * body
    pad = int(np.ceil(mx.elastic * body)) + 4
    top = min(y for *_, y in placed)
    W = int(np.ceil(max(px + w.pw.width for w, px, _ in placed))) + 2 * pad
    H = int(np.ceil(max(py + w.pw.alpha.shape[0] for w, _, py in placed) - top)) + 2 * pad
    canvas = np.zeros((H, W), np.float32)
    idx = np.zeros((H, W), np.int32)
    heads = []
    for i, (w, px, py) in enumerate(placed):
        a = w.pw.alpha
        ix, iy = int(round(px)) + pad, int(round(py - top)) + pad
        region = canvas[iy : iy + a.shape[0], ix : ix + a.shape[1]]
        sub = a[: region.shape[0], : region.shape[1]]
        np.maximum(region, sub, out=region)
        idx[iy : iy + sub.shape[0], ix : ix + sub.shape[1]][sub > 0.2] = i + 1
        heads.append((ix, iy + w.pw.head, w))

    # Headline run-ons between close words.
    for (ax, ay, aw), (bx, by, bw) in zip(heads, heads[1:]):
        stroke = aw.pw.glyph.stroke * body / aw.pw.glyph.body
        if aw.struck or bw.struck or rng.random() > join_prob:
            continue
        ends = []
        for ox, oy, ow, side in ((ax, ay, aw, -1), (bx, by, bw, 0)):
            band = ow.pw.alpha[max(0, int(ow.pw.head - 1.5 * stroke)) : int(ow.pw.head + 1.5 * stroke) + 1]
            cols = np.nonzero((band > 0.4).any(axis=0))[0]
            if len(cols) == 0:
                break
            ends.append((ox + cols[side], oy))
        if len(ends) == 2 and 0 < ends[1][0] - ends[0][0] < 0.6 * body:
            _pen(canvas, [ends[0], ends[1]], 0.9 * stroke)

    # Elastic wobble of the whole line (strokes and word index move together).
    if mx.elastic > 0:
        amp, cell = mx.elastic * body, 0.8 * body
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        mx_ = xx + (_smooth_noise((H, W), cell, rng) - 0.5) * 2 * amp
        my_ = yy + (_smooth_noise((H, W), cell, rng) - 0.5) * 2 * amp
        canvas = cv2.remap(canvas, mx_, my_, cv2.INTER_LINEAR, borderValue=0)
        idx = cv2.remap(idx.astype(np.float32), mx_, my_, cv2.INTER_NEAREST, borderValue=0).astype(np.int32)

    # Ink flow: fades word by word since the last dip, dark again after a refill.
    centers, levels = [], []
    for i, (w, px, _) in enumerate(placed):
        if ink_state["since"] >= ink_state["next"]:
            ink_state["since"], ink_state["next"] = 0, mx.dip_words * rng.uniform(0.7, 1.3)
        levels.append(1 - mx.dip_fade * ink_state["since"] / ink_state["next"])
        centers.append(px + pad + w.pw.width / 2)
        ink_state["since"] += 1
    canvas *= np.interp(np.arange(W), centers, levels).astype(np.float32)[None, :]

    out = []
    for i, (w, _, _) in enumerate(placed):
        ys, xs = np.nonzero(idx == i + 1)
        box = (int(xs.min()), int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1) if len(xs) else (0, 0, 0, 0)
        if w.struck and len(xs):  # one or two hurried strokes through the body
            bx, by, bw, bh = box
            stroke = w.pw.glyph.stroke * body / w.pw.glyph.body
            for k in range(int(rng.integers(1, 3))):
                yc = by + bh * rng.uniform(0.45, 0.7) + k * 0.25 * body
                _pen(canvas, _wavy(bx - 2, bx + bw + 3, yc, 0.08 * body, rng), stroke)
        out.append({"text": w.pw.text, "box": box, "parts": w.pw.glyph.parts, **({"struck": True} if w.struck else {})})
    head = float(placed[0][2] + placed[0][0].pw.head - top + pad)
    return Line(canvas, head, out)


# ---------------------------------------------------------------- page

def _put(page_a: np.ndarray, line: Line, x: int, head_y: float) -> list[dict]:
    """Max-composite a line so its headline sits at head_y; return its words in page coords."""
    top = int(round(head_y - line.head))
    H, W = page_a.shape
    a = line.alpha
    y0, x0 = max(0, top), max(0, x)
    y1, x1 = min(H, top + a.shape[0]), min(W, x + a.shape[1])
    if y1 > y0 and x1 > x0:
        np.maximum(page_a[y0:y1, x0:x1], a[y0 - top : y1 - top, x0 - x : x1 - x], out=page_a[y0:y1, x0:x1])
    return [{**w, "box": [x + w["box"][0], top + w["box"][1], w["box"][2], w["box"][3]]} for w in line.words]


def _line_record(words: list[dict]) -> dict:
    return {"text": " ".join(w["text"] for w in words if not w.get("struck")), "words": words}


def _warp_boxes(fields: list[dict], H: np.ndarray, shape) -> None:
    h, w = shape
    for f in fields:
        for line in f["lines"]:
            for wd in line["words"]:
                bx, by, bw, bh = wd["box"]
                c = cv2.perspectiveTransform(np.float32([[[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]]]), H)[0]
                x0, y0 = np.clip(c.min(0), 0, [w - 1, h - 1])
                x1, y1 = np.clip(c.max(0), 0, [w - 1, h - 1])
                wd["box"] = [int(x0), int(y0), int(np.ceil(x1 - x0)), int(np.ceil(y1 - y0))]


def generate_document(bank: SampleBank, rng: np.random.Generator, messiness: float | None = None,
                      age: float | None = None, photo: bool = True, paper: str | None = None,
                      degrade: float = 0.0, capture: str | None = None, size: float = 1.0,
                      spacing: float = 0.0, irregular: float = 0.0, tight: float = 0.0):
    """`degrade` > 0 adds the damage/capture/sharing layers of degrade_page in
    place of the plain photo; `capture` pins how the page was digitised.
    `size` scales the letter height (< 1: small writing, more text on the page
    instead of letters grown to fill it); `spacing` (0-1) makes word and letter
    gaps uneven; `irregular` (0-1) adds per-word slant changes and per-letter
    size, tilt, lean and bounce. `tight` (0-1) packs letters, words and lines
    closer together. capture="archive" defaults to old court paper."""
    m = float(rng.uniform(0.4, 1.0) if messiness is None else messiness)
    if capture == "archive":  # a century-old sheet: court paper, well aged
        paper = paper or "aged"
        age = float(rng.uniform(0.6, 1.0)) if age is None else age
    paper = paper or ("notebook" if rng.random() < 0.5 else "aged")
    if age is None:
        age = float(rng.uniform(0.0, 0.12) if paper == "notebook" else rng.uniform(0.2, 1.0))
    te = bank.script == "telu"
    st, mx = sample_messy(rng, bank.writers, m, paper, bank.script)
    st.spacing_jitter = float(spacing)
    st.slant_jitter = 8.0 * irregular
    st.letter_jitter = float(irregular)
    if tight:
        st.word_gap *= 1 - 0.55 * tight
        st.akshara_gap -= 0.08 * tight
        mx.pitch *= 1 - 0.18 * tight
    if size != 1.0:
        mx.nib = int(mx.nib * size)  # a broad nib would fill in small letters
    doc = (legal_text_te.sample_document(rng, lambda t: can_write(t, bank), length=(1 + 0.6 * tight) / size ** 2) if te
           else legal_text.sample_document(rng))
    Hp, Wp = PAGE
    body = st.body_px
    left = int(rng.uniform(0.07, 0.12) * Wp)
    right = Wp - int(rng.uniform(0.03, 0.07) * Wp)
    y = rng.uniform(0.06, 0.09) * Hp
    bottom = Hp - 0.05 * Hp

    def block_style(b, body):
        if b.get("signature"):
            return Style(**{**st.to_dict(), "extra_slant": st.extra_slant + 15, "word_gap": 0.15}), body * 1.35
        return st, body

    def prepare(body):
        out = []
        for b in doc["blocks"]:
            bst, bbody = block_style(b, body)
            sents = _prepare_sentences(b.get("sentences") or [b["text"]], bank, bst, mx, rng, bbody)
            if sents:
                out.append((b, bst, bbody, sents))
        return out

    def plan(blocks, body):
        """Lines per block and the total height they need."""
        width, groups, height = right - left, [], 0.0
        for b, _, bbody, sents in blocks:
            words = [w for ws in sents for w in ws]
            if b["align"] == "left":
                indent = int(0.07 * Wp) if b.get("indent") else 0
                g = _wrap(words, width - indent, width, st.word_gap * bbody, mx, np.random.default_rng(0))
            else:
                g = [words]
            groups.append(g)
            height += len(g) * mx.pitch * bbody + 0.3 * body
        return groups, height

    # Size the writing so the text fills the sheet, then drop sentences until it fits.
    avail = (bottom - y) * rng.uniform(0.85, 1.0)
    if size != 1.0:  # pinned size: no growing to fill the sheet
        body *= size
        st.body_px = body
    blocks = prepare(body)
    # Pinned size: only grow (by at most 25% in all) when the text runs out before the sheet does.
    lo, hi, body_lim = (0.8, 1.6, (13, 32)) if size == 1.0 else (1.0, 1.25, (0, body * 1.25))
    for _ in range(2):
        _, need = plan(blocks, body)
        k = float(np.clip(np.sqrt(avail / need), lo, hi))
        if abs(k - 1) <= 0.06 or not body_lim[0] <= body * k <= body_lim[1]:
            break
        body *= k
        st.body_px = body
        blocks = prepare(body)
    while True:
        _, need = plan(blocks, body)
        droppable = [(i, j) for i, (b, _, _, sents) in enumerate(blocks) if b.get("sentences") and len(sents) > 1
                     for j in range(1, len(sents))]
        if need <= avail or not droppable:
            break
        i, j = droppable[rng.integers(len(droppable))]
        del blocks[i][3][j]

    rule = mx.pitch * body               # ruled paper: every line sits on the next rule
    rule_ys = []
    page_a = np.zeros((Hp, Wp), np.float32)
    ink_state = {"since": 0, "next": mx.dip_words}
    fields, n_words = [], 0

    for b, bst, bbody, sents in blocks:
        sig = b.get("signature", False)
        words = [w for ws in sents for w in ws]
        indent = int(rng.uniform(0.04, 0.1) * Wp) if b.get("indent") else 0
        width = right - left
        groups = _wrap(words, width - indent, width, st.word_gap * bbody, mx, rng) if b["align"] == "left" else [words]
        recs = []
        for k, g in enumerate(groups):
            if y + 1.3 * bbody > bottom:
                break
            line = _render_line(g, bst, mx, rng, bbody, ink_state, join_prob=1.0 if sig and not te else None)
            lw = line.alpha.shape[1]
            if b["align"] == "center":
                x = int((left + right - lw) / 2 + rng.normal(0, 0.02 * Wp))
            elif b["align"] == "right":
                x = int(right - lw - rng.uniform(0.02, 0.08) * Wp)
            else:
                x = left + (indent if k == 0 else 0) + int(rng.normal(0, 0.15 * body))
            placed = _put(page_a, line, x, y)
            if mx.ruled and not sig and len(placed) >= 3:
                # Where the ink actually sits: most words end at the baseline (a few
                # dip lower with ु/ू), so a low percentile of word bottoms finds it.
                rule_ys.append((y, np.percentile([w["box"][1] + w["box"][3] for w in placed], 30) - y))
            recs.append(_line_record(placed))
            n_words += len(placed)
            ink_boxes = [w["box"] for w in placed if w["box"][2]]
            if (b.get("underline") or sig) and ink_boxes:  # wavy rule under the title / flourish under a signature
                x0 = min(bx for bx, _, _, _ in ink_boxes)
                x1 = max(bx + bw for bx, _, bw, _ in ink_boxes)
                yb = max(by + bh for _, by, _, bh in ink_boxes) + 0.25 * body
                pts = _wavy(x0 - 0.3 * body, x1 + (0.8 if sig else 0.3) * body, yb, 0.06 * body, rng)
                if sig:  # rises to the right like a quick flourish
                    pts[:, 1] -= np.linspace(0, 0.6 * body, len(pts))
                _pen(page_a, pts, max(1.2, 0.08 * body))
            y += rule if mx.ruled else mx.pitch * bbody * (1 + rng.normal(0, 0.04))
        if recs:
            fields.append({"name": b["name"], "text": " ".join(r["text"] for r in recs), "lines": recs})
        if mx.ruled:  # skip a rule after the title and before most paragraphs
            y += rule if b["name"] == "title" or (b.get("indent") and rng.random() < 0.6) else 0
        else:
            y += rng.uniform(0.0, 0.6) * body * (1.5 - mx.m)  # a little extra space between blocks
            if b["name"] == "title":
                y += 0.5 * body
        if y > bottom:
            break

    # A note written up the left margin (court papers only).
    if mx.paper == "aged" and rng.random() < 0.3 + 0.5 * m:
        sents = _prepare_sentences([doc["margin_note"]], bank, st, mx, rng, body * 0.9)
        if sents:
            words = sents[0]
            line = _render_line(words, st, mx, rng, body * 0.9, ink_state)
            a = np.ascontiguousarray(np.rot90(line.alpha))   # reads bottom to top
            h_, w_ = line.alpha.shape
            x = int(rng.uniform(0.15, 0.8) * (left - h_)) if left > h_ else 0
            yb = int(rng.uniform(0.3, 0.8) * Hp)            # bottom end of the note
            top = yb - a.shape[0]
            if 0 <= top and x + a.shape[1] <= Wp:
                np.maximum(page_a[top:yb, x : x + a.shape[1]], a, out=page_a[top:yb, x : x + a.shape[1]])
                ws = []
                for wd in line.words:  # rot90 (ccw): (x, y) -> (y, w_ - 1 - x)
                    bx, by, bw, bh = wd["box"]
                    ws.append({**wd, "box": [x + by, top + w_ - (bx + bw), bh, bw]})
                fields.append({"name": "margin_note", "rotation": 90, "text": _line_record(ws)["text"],
                               "lines": [_line_record(ws)]})

    # Pen: broad nib widens strokes along its angle; pressure, fades and blots; feathering.
    if mx.nib > 0:
        k = np.zeros((2 * mx.nib + 1, 2 * mx.nib + 1), np.uint8)
        dx, dy = mx.nib * np.cos(np.radians(mx.nib_angle)), mx.nib * np.sin(np.radians(mx.nib_angle))
        cv2.line(k, (int(round(mx.nib - dx)), int(round(mx.nib + dy))), (int(round(mx.nib + dx)), int(round(mx.nib - dy))), 1, 1)
        page_a = cv2.dilate(page_a, k)
    page_a = apply_ink(page_a, st, rng, n_words)
    bleed = rng.uniform(0.05, 0.2) if mx.paper == "notebook" else rng.uniform(0.2, 0.5)
    page_a = np.maximum(page_a, cv2.GaussianBlur(page_a, (0, 0), 1.2) * bleed)

    page = aged_paper(Hp, Wp, rng, age)
    if mx.ruled:  # extend the rule grid over the whole sheet
        r0 = rule_ys[0][0] + np.median([off for _, off in rule_ys]) + 0.06 * body if rule_ys else y + body
        rule_ys = np.arange(r0 - rule * int(r0 / rule), Hp, rule)
        ruled_lines(page, rule_ys, rng)
    if mx.paper == "aged":
        show_through(page, page_a, st.ink_bgr, rng, strength=rng.uniform(0.02, 0.05) * (0.5 + age))
    overlay(page, page_a, st.ink_bgr, 0, 0)
    n_folds = int(rng.integers(0, 3)) if mx.paper == "aged" and age > 0.3 else 0
    if n_folds:
        folds(page, rng, n_folds)
    layers = []
    if degrade > 0:
        page, Hm, layers = degrade_page(page, rng, degrade, capture)
        _warp_boxes(fields, Hm, (Hp, Wp))
    elif photo:
        page, Hm = photograph(page, rng, strength=rng.uniform(0.3, 1.0))
        _warp_boxes(fields, Hm, (Hp, Wp))
    ann = {"kind": "document", "script": bank.script, "paper": mx.paper, "degrade": round(float(degrade), 3), "layers": layers, "messiness": round(m, 3), "age": round(age, 3),
           "size": size, "spacing": spacing, "irregular": irregular, "tight": tight, "style": st.to_dict(),
           "messy": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in vars(mx).items()},
           "fields": fields, **({"rules_y": [round(float(r), 1) for r in rule_ys]} if mx.ruled else {})}
    return page, ann
