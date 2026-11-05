"""Compare a recognised transcription with the ground truth."""
from __future__ import annotations

import unicodedata


def normalize(s: str) -> str:
    """NFC, unify danda variants, collapse whitespace. Case and spacing around
    punctuation are irrelevant for Devanagari handwriting verification."""
    s = unicodedata.normalize("NFC", s).replace("|", "।").replace("॥", "।।")
    return " ".join(s.split())


def _edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


_DEVA_TO_ASCII = str.maketrans("०१२३४५६७८९", "0123456789")


def normalize_ocr(s: str) -> str:
    """Normalisation for OCR output: digit script is ignored
    (OCR engines often answer ४ as 4), and runs of dots/dashes (a dotted line
    read as text) are dropped; single dots, as in भा.दं.सं., are kept.
    Tokenisation conventions are unified so they do not count as word errors:
    punctuation attaches to the preceding word ("है ।" = "है।"), and a visarga
    and a colon after a word are the same mark."""
    import re

    s = normalize(s).translate(_DEVA_TO_ASCII).replace("ः", ":")
    s = re.sub(r"[-–—_.…·•]{2,}", " ", s)
    s = re.sub(r"\s+([।॥,:!?])", r"\1", s)
    return " ".join(s.split()).strip(" .")


def char_errors(hyp: str, ref: str) -> tuple[int, int]:
    """(edit distance, reference length) on code points, spaces ignored; sum
    these over a set of lines for the pooled CER."""
    h, r = normalize_ocr(hyp).replace(" ", ""), normalize_ocr(ref).replace(" ", "")
    return _edit_distance(list(h), list(r)), len(r)


def word_errors(hyp: str, ref: str) -> tuple[int, int]:
    h, r = normalize_ocr(hyp).split(), normalize_ocr(ref).split()
    return _edit_distance(h, r), len(r)
