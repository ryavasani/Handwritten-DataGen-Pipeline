"""Telugu text units for handwriting assembly.

As in Devanagari, handwritten Telugu is built from *aksharas*: a consonant
cluster with its vowel sign, written as one glyph. In a conjunct the first
consonant keeps its full form and each following consonant is written as a
small *vattu* below it (క్ష = క with a ష-vattu), so a cluster still occupies
one letter slot. There is no headline; letters sit in a main zone with vowel
marks above (ి ీ ె ే, the talakattu) and vattus below.

Akshara grammar used here:
    C N? (H ZW? C N?)* (H ZW? | M*) D*    consonant cluster + vowel sign(s) + modifiers
    V D*                                  independent vowel + modifiers
    any other single character            digits, punctuation, Latin
where C consonant, N nukta, H virama (pollu), M dependent vowel sign, V
independent vowel, D candrabindu/anusvara/visarga.
"""
from __future__ import annotations

import unicodedata

NUKTA = "఼"
VIRAMA = "్"
ANUSVARA = "ం"
VISARGA = "ః"
ZW = {"‌", "‍"}


def _in(ch: str, *ranges: tuple[int, int]) -> bool:
    o = ord(ch)
    return any(a <= o <= b for a, b in ranges)


def is_consonant(ch: str) -> bool:
    return _in(ch, (0x0C15, 0x0C39), (0x0C58, 0x0C5A))


def is_independent_vowel(ch: str) -> bool:
    return _in(ch, (0x0C05, 0x0C14), (0x0C60, 0x0C61))


def is_vowel_sign(ch: str) -> bool:
    return _in(ch, (0x0C3E, 0x0C4C), (0x0C55, 0x0C56), (0x0C62, 0x0C63))


def is_modifier(ch: str) -> bool:
    return _in(ch, (0x0C00, 0x0C04))


def aksharas(text: str) -> list[str]:
    """Split text into aksharas. Whitespace becomes its own unit."""
    s = unicodedata.normalize("NFC", text)
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        j = i + 1
        if is_consonant(ch):
            if j < n and s[j] == NUKTA:
                j += 1
            while j < n and s[j] == VIRAMA:
                k = j + 1
                while k < n and s[k] in ZW:
                    k += 1
                if k < n and is_consonant(s[k]):
                    j = k + 1
                    if j < n and s[j] == NUKTA:
                        j += 1
                else:
                    j = k  # word-final pollu (and any ZWNJ) stays on the consonant
                    break
            while j < n and is_vowel_sign(s[j]):
                j += 1
        while j < n and (is_modifier(s[j]) or s[j] == NUKTA):
            j += 1
        out.append(s[i:j])
        i = j
    return out


def split_cluster(ak: str) -> tuple[str, list[str]] | None:
    """'క్షా' -> ('కా', ['ష']): the first consonant with the akshara's vowel signs
    and modifiers (the glyph that is written in the letter slot), and the
    consonants written as vattus below it. None if the akshara is not a conjunct."""
    cons = [c for c in ak if is_consonant(c)]
    if len(cons) < 2 or VIRAMA not in ak:
        return None
    tail = "".join(c for c in ak if is_vowel_sign(c) or is_modifier(c))
    return cons[0] + tail, cons[1:]
