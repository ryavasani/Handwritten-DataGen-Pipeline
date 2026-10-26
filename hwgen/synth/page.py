"""Generate a filled, handwritten FIR page with ground truth."""
from __future__ import annotations

import cv2
import numpy as np

from .assemble import MissingText
from .bank import SampleBank
from .compose import overlay
from .content import sample_values
from .degrade import apply_ink
from .forms import Field, make_form
from .layout import LineRender, layout_line, prepare_words, wrap
from .style import Style, sample_style


def _fit_words(text: str, bank: SampleBank, style: Style, rng, body: float):
    """Prepare words, re-drawing the field value if the bank cannot render it."""
    try:
        return text, prepare_words(text, bank, style, rng, body)
    except MissingText as e:
        return None, str(e)


def _place(page: np.ndarray, line: LineRender, style: Style, rng, x: int, baseline: int, max_w: int) -> dict:
    a = line.alpha
    if a.shape[1] > max_w:  # squeeze slightly to stay on the dotted line
        k = max(0.7, max_w / a.shape[1])
        a = cv2.resize(a, (int(a.shape[1] * k), a.shape[0]), interpolation=cv2.INTER_AREA)
        for w in line.words:
            bx, by, bw, bh = w["box"]
            w["box"] = (int(bx * k), by, max(1, int(bw * k)), bh)
    a = apply_ink(a, style, rng, len(line.words))
    # Writers sit their baseline a little above the printed dots.
    top = int(round(baseline - line.baseline_y - rng.uniform(1, 4)))
    overlay(page, a, style.ink_bgr, x, top)
    words = [{"text": w["text"], "box": [x + w["box"][0], top + w["box"][1], w["box"][2], w["box"][3]],
              "parts": w["parts"]} for w in line.words]
    return {"text": " ".join(w["text"] for w in line.words), "words": words}


def fill_field(page, f: Field, text: str, bank, style, rng) -> list[dict]:
    body = style.body_px
    if f.name.endswith("_signature"):
        body *= 1.4
        style = Style(**{**style.to_dict(), "extra_slant": style.extra_slant + 12, "word_gap": 0.4})
    width = f.x1 - f.x0 - 10
    if f.lines > 1:
        body = min(body, 0.45 * f.line_gap)  # keep matras clear of the neighbouring ruled lines
        text, words = _fit_words(text, bank, style, rng, body)
        if text is None:
            return []
        groups = wrap(words, width, style, body)[: f.lines]
        rendered = [layout_line(ws, style, rng, body) for ws in groups]
    else:
        # A short field: write smaller (down to 65%) rather than run past the dots.
        for k in (1.0, 0.85, 0.75, 0.65):
            text, words = _fit_words(text, bank, style, rng, body * k)
            if text is None:
                return []
            rendered = [layout_line(words, style, rng, body * k)]
            if rendered[0].alpha.shape[1] <= width - 10:
                break
    out = []
    for k, line in enumerate(rendered):
        x = f.x0 + (int(rng.uniform(4, 24)) if k == 0 else int(rng.uniform(0, 10)))
        out.append(_place(page, line, style, rng, x, f.baseline + k * f.line_gap, f.x1 - x))
    return out


def scan_effects(page: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light scanner/photocopy look; heavier degradation belongs to the
    downstream pipeline this will be integrated with."""
    h, w = page.shape[:2]
    out = page.astype(np.float32)
    shade = cv2.resize(rng.random((6, 4)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    out *= (0.94 + 0.06 * shade)[..., None]
    out += rng.normal(0, rng.uniform(2, 5), out.shape)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.normal(0, 0.4), 1.0)
    out = cv2.warpAffine(np.clip(out, 0, 255).astype(np.uint8), M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    q = int(rng.integers(70, 92))
    return cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def generate_page(bank: SampleBank, rng: np.random.Generator, scan: bool = True, max_retries: int = 5):
    page, fields = make_form(rng)
    style = sample_style(rng, bank.writers, body_px=rng.uniform(19, 26))
    values = sample_values(rng)
    ann = {"style": style.to_dict(), "fields": [], "skipped": []}
    for f in fields:
        lines = []
        for attempt in range(max_retries):
            text = values[f.name] if attempt == 0 else sample_values(rng)[f.name]
            lines = fill_field(page, f, text, bank, style, rng)
            if lines:
                break
        if lines:
            ann["fields"].append({"name": f.name, "label": f.label,
                                  "text": " ".join(l["text"] for l in lines), "lines": lines})
        else:
            ann["skipped"].append(f.name)
    if scan:
        page = scan_effects(page, rng)
    return page, ann
