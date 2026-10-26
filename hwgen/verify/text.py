"""Compare a recognised transcription with the ground truth."""
from __future__ import annotations

import unicodedata

from ..text.devanagari import aksharas


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


def cer(hyp: str, ref: str) -> float:
    """Character error rate on Unicode code points (spaces ignored)."""
    h, r = normalize(hyp).replace(" ", ""), normalize(ref).replace(" ", "")
    return _edit_distance(list(h), list(r)) / max(len(r), 1)


def aer(hyp: str, ref: str) -> float:
    """Akshara error rate: the unit the generator assembles, so one wrong
    conjunct counts once rather than as three code points."""
    h = [a for a in aksharas(normalize(hyp)) if not a.isspace()]
    r = [a for a in aksharas(normalize(ref)) if not a.isspace()]
    return _edit_distance(h, r) / max(len(r), 1)


def wer(hyp: str, ref: str) -> float:
    h, r = normalize(hyp).split(), normalize(ref).split()
    return _edit_distance(h, r) / max(len(r), 1)
