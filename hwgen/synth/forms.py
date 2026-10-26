"""Synthetic clean pages: a printed Hindi FIR form (CrPC §154 layout) with the
positions of every field a police officer fills in by hand."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/System/Library/Fonts/Kohinoor.ttc"  # Devanagari + Latin; Pillow shapes it via raqm
PAGE_W, PAGE_H = 1240, 1754                   # A4 at 150 DPI
MARGIN = 80


@dataclass
class Field:
    name: str
    label: str
    x0: int
    x1: int
    baseline: int          # y of the dotted leader the handwriting sits on
    lines: int = 1
    line_gap: int = 0      # distance between ruled lines for multi-line fields


# Each row: list of (printed label, field name or None, leader width in px or "rest").
FORM = [
    ("title", "प्रथम सूचना रिपोर्ट"),
    ("subtitle", "(धारा 154 दंड प्रक्रिया संहिता के अंतर्गत)"),
    ("row", [("1. जिला", "district", 200), ("थाना", "station", 190), ("वर्ष", "year", 110), ("प्र.सू.रि. सं.", "fir_no", "rest")]),
    ("row", [("दिनांक", "fir_date", 220)]),
    ("row", [("2. अधिनियम", "act", 330), ("धाराएँ", "sections", "rest")]),
    ("row", [("3. (क) घटना का दिन", "incident_day", 190), ("दिनांक", "incident_date", 210), ("समय", "incident_time", "rest")]),
    ("row", [("(ख) थाने पर सूचना प्राप्त होने का दिनांक", "received_date", 210), ("समय", "received_time", "rest")]),
    ("row", [("4. सूचना का प्रकार : लिखित / मौखिक", None, 0)]),
    ("row", [("5. घटनास्थल : (क) थाने से दिशा और दूरी", "direction", "rest")]),
    ("row", [("(ख) पता", "place", "rest")]),
    ("row", [("6. शिकायतकर्ता / सूचनाकर्ता :", None, 0)]),
    ("row", [("(क) नाम", "complainant", 330), ("(ख) पिता / पति का नाम", "father", "rest")]),
    ("row", [("(ग) जन्म वर्ष", "birth_year", 160), ("(घ) राष्ट्रीयता", "nationality", 200), ("(ङ) मोबाइल नं.", "phone", "rest")]),
    ("row", [("(च) पता", "address", "rest")]),
    ("block", "7. ज्ञात / संदिग्ध / अज्ञात अभियुक्त का पूर्ण विवरण :", "accused", 2),
    ("block", "8. सूचना देने में विलंब के कारण :", "delay", 1),
    ("block", "9. चोरी गई / संलिप्त संपत्ति का विवरण :", "property", 1),
    ("block", "10. प्रथम सूचना रिपोर्ट की विषय-वस्तु :", "narrative", 6),
    ("block", "11. की गई कार्यवाही :", "action", 2),
    ("signatures", None),
]


def _dotted(draw: ImageDraw.ImageDraw, x0: int, x1: int, y: int, color=(70, 70, 70)) -> None:
    for x in range(x0, x1, 6):
        draw.ellipse((x, y, x + 1.6, y + 1.6), fill=color)


def make_form(rng: np.random.Generator) -> tuple[np.ndarray, list[Field]]:
    """Render a blank form; returns (BGR page, fields)."""
    paper = tuple(int(v) for v in rng.integers(236, 250, 3))
    img = Image.new("RGB", (PAGE_W, PAGE_H), paper)
    d = ImageDraw.Draw(img)
    ink = (25, 25, 25)
    f_title = ImageFont.truetype(FONT, 34, index=2)  # bold face of the collection
    f_sub = ImageFont.truetype(FONT, 20)
    f = ImageFont.truetype(FONT, 21)
    row_h = 47
    fields: list[Field] = []
    y = MARGIN
    for item in FORM:
        kind = item[0]
        if kind == "title":
            w = d.textlength(item[1], font=f_title)
            d.text(((PAGE_W - w) / 2, y), item[1], font=f_title, fill=ink)
            y += 50
        elif kind == "subtitle":
            w = d.textlength(item[1], font=f_sub)
            d.text(((PAGE_W - w) / 2, y), item[1], font=f_sub, fill=ink)
            y += 46
        elif kind == "row":
            x = MARGIN
            base = y + 30  # baseline of printed text at size 21 is ~24 px below its top
            for label, name, width in item[1]:
                d.text((x, y), label, font=f, fill=ink)
                x += int(d.textlength(label, font=f)) + 8
                if name:
                    x1 = PAGE_W - MARGIN if width == "rest" else x + width
                    _dotted(d, x, x1, base)
                    fields.append(Field(name, label, x, x1, base))
                    x = x1 + 14
            y += row_h
        elif kind == "block":
            _, label, name, n = item
            d.text((MARGIN, y), label, font=f, fill=ink)
            first = y + row_h + 22
            for k in range(n):
                _dotted(d, MARGIN + 20, PAGE_W - MARGIN, first + k * row_h)
            fields.append(Field(name, label, MARGIN + 20, PAGE_W - MARGIN, first, n, row_h))
            y = first + (n - 1) * row_h + 24
        elif kind == "signatures":
            y += 10
            d.text((MARGIN, y + 60), "शिकायतकर्ता / सूचनाकर्ता के हस्ताक्षर", font=f, fill=ink)
            fields.append(Field("complainant_signature", "हस्ताक्षर", MARGIN, MARGIN + 380, y + 45))
            rx = PAGE_W // 2 + 80
            d.text((rx, y + 60), "थाना प्रभारी के हस्ताक्षर", font=f, fill=ink)
            fields.append(Field("officer_signature", "हस्ताक्षर", rx, PAGE_W - MARGIN, y + 45))
            d.text((rx, y + 110), "नाम", font=f, fill=ink)
            nx = rx + int(d.textlength("नाम", font=f)) + 8
            _dotted(d, nx, PAGE_W - MARGIN, y + 140)
            fields.append(Field("officer_name", "नाम", nx, PAGE_W - MARGIN, y + 140))
            d.text((rx, y + 160), "पद", font=f, fill=ink)
            px = rx + int(d.textlength("पद", font=f)) + 8
            _dotted(d, px, PAGE_W - MARGIN, y + 190)
            fields.append(Field("officer_rank", "पद", px, PAGE_W - MARGIN, y + 190))
    page = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return page, fields
