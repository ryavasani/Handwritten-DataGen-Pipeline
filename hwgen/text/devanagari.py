"""Devanagari text units for handwriting assembly.

Handwritten Devanagari is assembled from *aksharas* (orthographic syllables),
not Unicode code points: a conjunct like क्ष or a pre-base vowel sign like the
ि in कि is one visual unit even though it spans several code points (and the
ि is drawn to the left of a consonant that comes before it in memory).

Akshara grammar used here:
    C (N)? (H C (N)?)* (H | M* ) D*      consonant cluster + vowel sign(s) + modifiers
    V D*                                  independent vowel + modifiers
    any other single character            digits, danda, punctuation, Latin
where C consonant, N nukta, H virama, M dependent vowel sign, V independent
vowel, D chandrabindu/anusvara/visarga. ZWJ/ZWNJ stay attached to the virama.
"""
from __future__ import annotations

import unicodedata

NUKTA = "़"
VIRAMA = "्"
ZW = {"‌", "‍"}


def _in(ch: str, *ranges: tuple[int, int]) -> bool:
    o = ord(ch)
    return any(a <= o <= b for a, b in ranges)


def is_consonant(ch: str) -> bool:
    return _in(ch, (0x0915, 0x0939), (0x0958, 0x095F), (0x0978, 0x097F))


def is_independent_vowel(ch: str) -> bool:
    return _in(ch, (0x0904, 0x0914), (0x0960, 0x0961), (0x0972, 0x0977))


def is_vowel_sign(ch: str) -> bool:
    return _in(ch, (0x093A, 0x093B), (0x093E, 0x094C), (0x094E, 0x094F), (0x0955, 0x0957), (0x0962, 0x0963))


def is_modifier(ch: str) -> bool:
    return _in(ch, (0x0900, 0x0903))


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
            # Conjuncts: virama (optionally ZWJ/ZWNJ) followed by another consonant.
            while j < n and s[j] == VIRAMA:
                k = j + 1
                while k < n and s[k] in ZW:
                    k += 1
                if k < n and is_consonant(s[k]):
                    j = k + 1
                    if j < n and s[j] == NUKTA:
                        j += 1
                else:
                    j = k  # word-final virama (halant) stays on the consonant
                    break
            while j < n and is_vowel_sign(s[j]):
                j += 1
        elif is_independent_vowel(ch):
            pass
        while j < n and (is_modifier(s[j]) or s[j] == NUKTA):
            j += 1
        out.append(s[i:j])
        i = j
    return out
