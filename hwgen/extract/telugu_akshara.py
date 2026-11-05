"""Cut a gold-labelled Telugu word image into aksharas.

Telugu has no headline, and most aksharas are written as separate glyphs, so
the word is cut along connected components rather than columns. Components
that fill the main zone are letter *units*; smaller ones above it (vowel marks,
the talakattu when lifted) or below it (vattus, ు/ూ tails) are *marks* that
belong to the letter they sit over. As in the Devanagari cutter, units and
aksharas do not map one-to-one: ం and ః are written as separate circles, ు/ూ
may stand apart from the consonant, a vattu may be written beside the letter, and some letters break into two strokes.
Each akshara has a few allowed unit patterns (expected relative widths with a
prior penalty); a dynamic programme assigns consecutive units to aksharas and
the cut is accepted only when the best assignment fits well.

A conjunct akshara's marks below the main zone are its vattus: they are also
returned on their own, so a missing conjunct can be built from its base
letter plus a vattu written by someone else.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..text.telugu import ANUSVARA, VISARGA, aksharas, is_consonant, is_independent_vowel, split_cluster

U_SIGNS = set("ుూ")


def patterns(ak: str) -> list[tuple[list[float], float]]:
    """Allowed unit patterns: (expected width of each unit, in letter widths; prior penalty)."""
    if not any(is_consonant(c) or is_independent_vowel(c) for c in ak):
        return [([0.6], 0.0)] if ak.isdigit() else [([0.3], 0.0)]
    w = 1.0
    opts = [([w], 0.0), ([0.55 * w, 0.55 * w], 0.45)]  # one glyph, or broken into two strokes
    if U_SIGNS & set(ak):
        opts = [([1.35 * w], 0.0), ([w, 0.45 * w], 0.1)]
    if ANUSVARA in ak:
        opts = [(p + [0.5], c) for p, c in opts] + [([sum(p) + 0.5], c + 0.4) for p, c in opts[:1]]
    if VISARGA in ak:
        opts = [(p + [0.3], c) for p, c in opts]
    # Many writers put a vattu (్న ్య ్ర ...) beside the letter rather than under it.
    n_vattu = sum(is_consonant(c) for c in ak) - 1
    if n_vattu > 0:
        opts += [(p[:1] + [0.55] * n_vattu + p[1:], c + 0.05) for p, c in opts if len(p) <= 2]
    return opts


@dataclass
class Unit:
    x0: int
    x1: int
    comps: list[int] = field(default_factory=list)


def units_and_marks(mask: np.ndarray, zone: tuple[int, int], min_frac: float = 0.01):
    """(labels, letter units left to right, mark component ids). Components
    smaller than min_frac of the ink are dropped as specks."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    z0, z1 = zone
    zh = max(z1 - z0, 1)
    total = stats[1:, cv2.CC_STAT_AREA].sum() if n > 1 else 0
    spans, marks = [], []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_frac * total:
            continue
        overlap = min(y + h, z1) - max(y, z0)
        if overlap >= 0.5 * zh or (overlap >= 0.6 * h and h >= 0.35 * zh):
            spans.append((x, x + w, i))
        else:
            marks.append(i)
    units: list[Unit] = []
    for a, b, i in sorted(spans):
        if units and min(units[-1].x1, b) - max(units[-1].x0, a) > 0.5 * min(units[-1].x1 - units[-1].x0, b - a):
            units[-1].x1 = max(units[-1].x1, b)
            units[-1].comps.append(i)
        else:
            units.append(Unit(a, b, [i]))
    return labels, stats, units, marks


def align(units: list[Unit], aks: list[str], zh: float, narrow_cost: float = 0.4):
    """Assign consecutive units to aksharas: ([(first, end) unit index per
    akshara], mean cost per akshara) or None if nothing fits."""
    m, n = len(units), len(aks)
    pats = [patterns(a) for a in aks]
    if not (sum(min(len(p) for p, _ in ps) for ps in pats) <= m <= sum(max(len(p) for p, _ in ps) for ps in pats)):
        return None
    # Pixel width of one letter, from the units' own widths.
    u = sum(x.x1 - x.x0 for x in units) / max(sum(sum(ps[0][0]) for ps in pats), 1e-6)
    INF = float("inf")
    dp = np.full((n + 1, m + 1), INF)
    back = np.zeros((n + 1, m + 1), int)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            for widths, prior in pats[i - 1]:
                k = len(widths)
                if k > j or dp[i - 1, j - k] == INF:
                    continue
                us = units[j - k : j]
                c = prior
                for ew, x in zip(widths, us):
                    if (ew < 0.5) != ((x.x1 - x.x0) < 0.45 * zh):
                        c += narrow_cost
                exp_w = sum(widths) * u
                c += abs(sum(x.x1 - x.x0 for x in us) - exp_w) / exp_w
                if dp[i - 1, j - k] + c < dp[i, j]:
                    dp[i, j], back[i, j] = dp[i - 1, j - k] + c, k
    if dp[n, m] == INF:
        return None
    ranges, j = [], m
    for i in range(n, 0, -1):
        k = back[i, j]
        ranges.append((j - k, j))
        j -= k
    return ranges[::-1], float(dp[n, m] / n)


@dataclass
class AksharaPiece:
    text: str
    mask: np.ndarray               # this akshara's ink (full word size)
    vattus: list[tuple[str, np.ndarray]] = field(default_factory=list)  # (consonant, ink below the main zone)


def cut_word(mask: np.ndarray, text: str, zone: tuple[int, int], max_cost: float = 0.35):
    """(akshara pieces, mean alignment cost), or None if the cut is not trustworthy."""
    aks = [a for a in aksharas(text) if not a.isspace()]
    if len(aks) < 2:
        return None
    labels, stats, units, marks = units_and_marks(mask, zone)
    if not units:
        return None
    zh = zone[1] - zone[0]
    res = align(units, aks, zh)
    if res is None or res[1] > max_cost:
        return None
    ranges, cost = res
    owner = {}
    spans = []
    for t, (a, b) in enumerate(ranges):
        spans.append((units[a].x0, units[b - 1].x1))
        for un in units[a:b]:
            for c in un.comps:
                owner[c] = t
    below = set()
    for c in marks:
        x, y, w, h, _ = stats[c]
        ov = [min(x + w, s1) - max(x, s0) for s0, s1 in spans]
        t = int(np.argmax(ov))
        if ov[t] <= 0:  # beside every letter: assign to the nearest one
            t = int(np.argmin([min(abs(x - s1), abs(x + w - s0)) for s0, s1 in spans]))
        owner[c] = t
        if y >= zone[1] - 0.15 * zh:
            below.add(c)
    lut = np.full(labels.max() + 1, -1, int)
    for c, t in owner.items():
        lut[c] = t
    own = lut[labels]
    pieces = []
    for t, ak in enumerate(aks):
        piece = AksharaPiece(ak, own == t)
        split = split_cluster(ak)
        low = [c for c in below if owner[c] == t]
        if split and len(split[1]) == 1 and low and not (U_SIGNS & set(ak)):
            piece.vattus.append((split[1][0], np.isin(labels, low)))
        pieces.append(piece)
    return pieces, cost
