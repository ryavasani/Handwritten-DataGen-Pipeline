"""The three verification checks:

1. OCR      - the VLM transcribes each handwritten line (without seeing the
              ground truth); we score it against the text we meant to write.
2. Realism  - the VLM rates a page against real held-out handwriting on a
              fixed rubric and lists concrete artifacts.
3. Real vs fake - a real held-out word and a synthetic assembly of the same
              word, rendered identically, side by side: can the VLM tell which
              one was assembled? ~50% accuracy means indistinguishable.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..synth.assemble import _assemble
from ..synth.bank import SampleBank
from ..synth.compose import overlay
from ..synth.degrade import apply_ink
from ..synth.layout import _transform, layout_line, prepare_words
from ..synth.style import Style, sample_style
from ..text.devanagari import aksharas
from .text import aer, cer
from .vlm import VLMClient

# ---------------------------------------------------------------- 1. OCR

TRANSCRIBE_SYSTEM = (
    "You are an expert reader of handwritten Hindi (Devanagari script). Transcribe exactly what is "
    "written in each numbered image, character by character. Keep Devanagari digits as Devanagari "
    "digits and keep punctuation as written (/ : . , । -). Do not correct spelling, do not complete "
    "words from context, and do not translate. Write ? for any character you cannot read."
)
TRANSCRIBE_SCHEMA = {
    "type": "object",
    "properties": {"lines": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "text": {"type": "string"}},
        "required": ["id", "text"], "additionalProperties": False}}},
    "required": ["lines"], "additionalProperties": False,
}


def line_crops(page: np.ndarray, ann: dict, pad: int = 6) -> list[dict]:
    """One crop per written line, from the word boxes in the ground truth."""
    out = []
    H, W = page.shape[:2]
    for f in ann["fields"]:
        for k, line in enumerate(f["lines"]):
            boxes = np.array([w["box"] for w in line["words"]])
            if len(boxes) == 0:
                continue
            x0, y0 = boxes[:, 0].min() - pad, boxes[:, 1].min() - pad
            x1, y1 = (boxes[:, 0] + boxes[:, 2]).max() + pad, (boxes[:, 1] + boxes[:, 3]).max() + pad
            crop = page[max(0, y0) : min(H, y1), max(0, x0) : min(W, x1)]
            out.append({"field": f["name"], "line": k, "text": line["text"], "image": crop})
    return out


def check_ocr(vlm: VLMClient, page: np.ndarray, ann: dict) -> list[dict]:
    crops = line_crops(page, ann)
    parts = ["Transcribe each numbered handwritten line image."]
    for i, c in enumerate(crops):
        parts += [f"Line {i}:", c["image"]]
    res = vlm.ask(parts, TRANSCRIBE_SCHEMA, TRANSCRIBE_SYSTEM)
    got = {r["id"]: r["text"] for r in res.get("lines", [])}
    out = []
    for i, c in enumerate(crops):
        hyp = got.get(i)
        out.append({"field": c["field"], "line": c["line"], "ref": c["text"], "hyp": hyp,
                    "cer": cer(hyp, c["text"]) if hyp is not None else None,
                    "aer": aer(hyp, c["text"]) if hyp is not None else None})
    return out


# ---------------------------------------------------------------- 2. Realism

RUBRIC = {
    "letterforms": "Do individual letters look naturally handwritten (no cut-off strokes, stray fragments, broken shapes)?",
    "joins": "Do letters within a word connect naturally, with a continuous, believable headline (shirorekha)?",
    "size_spacing": "Are letter sizes, word spacing and baselines consistent the way one person's writing is?",
    "ink": "Does the ink look like real ballpoint on paper (pressure, edges, colour, fading)?",
    "writer_consistency": "Does the whole page look like it was written by a single person?",
    "document_plausibility": "Would this pass as a real handwritten Indian legal form (e.g. an FIR)?",
}
JUDGE_SYSTEM = (
    "You are a forensic document examiner specialising in handwritten Hindi (Devanagari) documents. "
    "You will see reference images of REAL handwriting, then a document page whose handwriting may "
    "be real or computer-generated. Judge the handwriting on the page only (ignore the printed form). "
    "Score each criterion from 1 (clearly artificial) to 5 (indistinguishable from real), comparing "
    "against the references. List every concrete artifact you notice, naming the field it is in."
)
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {"type": "object", "properties": {k: {"type": "integer", "minimum": 1, "maximum": 5} for k in RUBRIC},
                   "required": list(RUBRIC), "additionalProperties": False},
        "overall": {"type": "integer", "minimum": 1, "maximum": 5},
        "artifacts": {"type": "array", "items": {
            "type": "object",
            "properties": {"field": {"type": "string"}, "description": {"type": "string"},
                           "severity": {"type": "string", "enum": ["minor", "moderate", "severe"]}},
            "required": ["field", "description", "severity"], "additionalProperties": False}},
        "summary": {"type": "string"},
    },
    "required": ["scores", "overall", "artifacts", "summary"], "additionalProperties": False,
}


def reference_panel(ref_bank: SampleBank, rng: np.random.Generator, n_lines: int = 4,
                    words_per_line: int = 6) -> np.ndarray:
    """Lines of REAL handwriting: whole-word samples from held-out writers,
    laid out with the same line renderer and ink as generated pages, so the
    judge compares letterforms like for like."""
    rows = []
    writers = ref_bank.writers
    texts = [t for t, by in ref_bank.words.items() if any(by.values()) and len(aksharas(t)) >= 2]
    for _ in range(n_lines):
        st = sample_style(rng, writers, body_px=22)
        st.extra_slant = 0.0
        words = [str(t) for t in rng.choice(texts, words_per_line, replace=False)]
        # Only whole words exist in this text set, so nothing is assembled.
        pw = prepare_words(" ".join(words), ref_bank, st, rng)
        line = layout_line(pw, st, rng)
        a = apply_ink(line.alpha, st, rng, len(words))
        canvas = np.full((a.shape[0] + 16, a.shape[1] + 20, 3), 242, np.uint8)
        overlay(canvas, a, st.ink_bgr, 10, 8)
        rows.append(canvas)
    W = max(r.shape[1] for r in rows)
    return np.vstack([np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0)), constant_values=242) for r in rows])


def check_realism(vlm: VLMClient, page: np.ndarray, references: np.ndarray) -> dict:
    criteria = "\n".join(f"- {k}: {v}" for k, v in RUBRIC.items())
    parts = ["Reference: real handwriting by several different writers.", references,
             "Document page to examine:", page,
             f"Score the page's handwriting on these criteria:\n{criteria}"]
    return vlm.ask(parts, JUDGE_SCHEMA, JUDGE_SYSTEM)


# ---------------------------------------------------------------- 3. Real vs fake

PAIR_SYSTEM = (
    "You are a forensic document examiner specialising in handwritten Hindi (Devanagari). "
    "Each image shows the same word twice, labelled A and B. One was written by a person in a single "
    "pass. The other was assembled by a computer from pieces of real handwriting by possibly different "
    "people. Identify the assembled one."
)
PAIR_SCHEMA = {
    "type": "object",
    "properties": {"assembled": {"type": "string", "enum": ["A", "B"]},
                   "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                   "cues": {"type": "string"}},
    "required": ["assembled", "confidence", "cues"], "additionalProperties": False,
}


def pair_candidates(ref_bank: SampleBank, synth_bank: SampleBank, min_aksharas: int = 3) -> list[str]:
    """Words with a real held-out sample whose aksharas all exist in the synth
    bank, so the same word can be assembled from other writers' pieces."""
    out = []
    for t, by in ref_bank.words.items():
        aks = [a for a in aksharas(t) if not a.isspace()]
        if any(by.values()) and len(aks) >= min_aksharas and all(synth_bank.has_akshara(a) for a in aks):
            out.append(t)
    return sorted(out)


def _render_word(alpha: np.ndarray, head: float, base: float, body_px: float, st: Style, rng) -> np.ndarray:
    from ..synth.assemble import Glyph
    g = Glyph(alpha, head, base, base - head, 0.0)
    a, _, _ = _transform(g, body_px / g.body, 0.0, 0.0)
    a = apply_ink(a, st, rng)
    canvas = np.full((a.shape[0] + 20, a.shape[1] + 24, 3), 242, np.uint8)
    overlay(canvas, a, st.ink_bgr, 12, 10)
    return canvas


def make_pair(text: str, ref_bank: SampleBank, synth_bank: SampleBank, rng: np.random.Generator,
              body_px: float = 40) -> tuple[np.ndarray, str, dict]:
    """Return (side-by-side image, which label is synthetic, metadata)."""
    real = ref_bank.word(text, str(rng.choice(ref_bank.writers)), rng)
    f = real.features
    writer = str(rng.choice(synth_bank.writers))
    fake = _assemble(text, synth_bank, writer, rng, gap=float(rng.uniform(-0.05, 0.02)))
    # Identical ink, paper and degradation for both, so only the writing differs.
    st = sample_style(rng, synth_bank.writers, body_px=body_px)
    seed = int(rng.integers(1 << 30))
    real_img = _render_word(ref_bank.alpha(real), f["headline_y"], f["baseline_y"], body_px, st, np.random.default_rng(seed))
    fake_img = _render_word(fake.alpha, fake.headline_y, fake.baseline_y, body_px, st, np.random.default_rng(seed))
    fake_first = bool(rng.random() < 0.5)
    a_img, b_img = (fake_img, real_img) if fake_first else (real_img, fake_img)
    h = max(a_img.shape[0], b_img.shape[0])

    def labelled(img, lab):
        img = np.pad(img, ((0, h - img.shape[0]), (0, 0), (0, 0)), constant_values=242)
        bar = np.full((28, img.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, lab, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        return np.vstack([bar, img])

    gap = np.full((h + 28, 30, 3), 255, np.uint8)
    pair = np.hstack([labelled(a_img, "A"), gap, labelled(b_img, "B")])
    meta = {"text": text, "real_sample": real.id, "real_writer": real.writer, "fake_writer": writer,
            "fake_parts": fake.parts}
    return pair, ("A" if fake_first else "B"), meta


def check_pair(vlm: VLMClient, pair: np.ndarray, text: str) -> dict:
    return vlm.ask([f"The word is: {text}", pair, "Which one, A or B, was assembled by a computer?"],
                   PAIR_SCHEMA, PAIR_SYSTEM, max_tokens=4000)
