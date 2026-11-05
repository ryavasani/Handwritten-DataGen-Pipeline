"""Cut a gold-labelled Devanagari word image into aksharas.

Below the headline, Devanagari letters separate into vertical *units*, which
are either letter bodies or narrow bars. Units and aksharas do not map
one-to-one: ा/ी/ो/ौ add a bar to the right (unless the writer joined it to the
consonant), ि adds a stem to the left, a half consonant in a conjunct may or
may not stand apart, and some consonants (ग ण श ख) often split into a body and
a bar. Each akshara therefore has several allowed unit *patterns*; a dynamic
programme assigns consecutive units to aksharas, scoring each choice by how
well unit types and widths match the pattern. The cut is accepted only when
the best assignment fits well.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import cv2
import numpy as np

from ..text.devanagari import aksharas, is_consonant, is_independent_vowel
from .features import headline_band

# Unit types: C letter body, H half consonant, B bar (right of a body), S ि stem.
WIDTH = {"C": 1.0, "H": 0.6, "B": 0.3, "S": 0.3}
NARROW = {"B", "S"}

TWO_PART = set("गणशखऋ")
BAR_SIGNS = set("ाीोौ")  # ा ी ो ौ
I_SIGN = "ि"                             # ि
VOWEL_PATTERNS = {
    "अ": [("CB", 0.0), ("C", 0.2)],
    "आ": [("CBB", 0.0), ("CB", 0.2)],
    "ओ": [("CBB", 0.0), ("CB", 0.2)],
    "औ": [("CBB", 0.0), ("CB", 0.2)],
    "ई": [("C", 0.0), ("CB", 0.1)],
    "ऊ": [("C", 0.0), ("CC", 0.1)],
}
JOINED_BAR_PENALTY = 0.3  # writer joined ा/ी to the consonant: possible, less common


def patterns(ak: str) -> list[tuple[str, float]]:
    """Allowed unit patterns for an akshara, each with a prior penalty."""
    if not ak:
        return []
    if is_independent_vowel(ak[0]):
        base = VOWEL_PATTERNS.get(ak[0], [("C", 0.0)])
    else:
        consonants = [c for c in ak if is_consonant(c)]
        if not consonants:
            # Danda and the Latin digit 1 are bars; other digits/punctuation are
            # bodies (the Devanagari one, १, is a hook, not a bar).
            return [("B", 0.0)] if ak in ("।", "॥", "1", "|") else [("C", 0.0)]
        opts = []
        for k, c in enumerate(consonants):
            if k < len(consonants) - 1:
                opts.append([("H", 0.0), ("", 0.0)])           # half form: apart or fused
            elif c in TWO_PART:
                opts.append([("C", 0.0), ("CB", 0.0)])
            else:
                opts.append([("C", 0.0)])
        base = [("".join(p for p, _ in combo), sum(c for _, c in combo)) for combo in product(*opts)]
    out = []
    for pat, cost in base:
        if I_SIGN in ak:
            pat = "S" + pat
        if any(ch in BAR_SIGNS for ch in ak):
            out.append((pat + "B", cost))
            out.append((pat, cost + JOINED_BAR_PENALTY))
        else:
            out.append((pat, cost))
    # Distinct patterns, cheapest first.
    best: dict[str, float] = {}
    for p, c in out:
        if p and c < best.get(p, 9.0):
            best[p] = c
    return sorted(best.items(), key=lambda pc: pc[1])


@dataclass
class Unit:
    x0: int
    x1: int
    narrow: bool


@dataclass
class AksharaCut:
    text: str
    x0: int
    x1: int  # exclusive


def body_units(mask: np.ndarray, min_frac: float = 0.03, narrow_frac: float = 0.3) -> list[Unit]:
    """Column spans of ink below the headline, typed as narrow bars or bodies.
    Small blobs (nukta, stray dots) are not units; overlapping spans merge."""
    hl0, hl1 = headline_band(mask)
    body = mask[hl1:].astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(body, connectivity=8)
    if n <= 1:
        return []
    total = stats[1:, cv2.CC_STAT_AREA].sum()
    spans = sorted(
        (int(s[cv2.CC_STAT_LEFT]), int(s[cv2.CC_STAT_LEFT] + s[cv2.CC_STAT_WIDTH]))
        for s in stats[1:]
        if s[cv2.CC_STAT_AREA] >= min_frac * total
    )
    merged: list[list[int]] = []
    for a, b in spans:
        if merged and min(merged[-1][1], b) - max(merged[-1][0], a) > 0.5 * min(merged[-1][1] - merged[-1][0], b - a):
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    # Body height: headline to the row above which 92% of the body ink lies.
    prof = body.sum(axis=1).astype(np.float64)
    body_h = max(int(np.searchsorted(np.cumsum(prof) / prof.sum(), 0.92)) + 1, 1)
    return [Unit(a, b, (b - a) <= narrow_frac * body_h) for a, b in merged]


def column_units(mask: np.ndarray, min_frac: float = 0.03) -> list[Unit]:
    """Connected components over the full height, merged when they overlap
    horizontally: the units of a headline-less string such as ४०००."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return []
    total = stats[1:, cv2.CC_STAT_AREA].sum()
    spans = sorted((int(s[0]), int(s[0] + s[2])) for s in stats[1:] if s[4] >= min_frac * total)
    merged: list[list[int]] = []
    for a, b in spans:
        if merged and min(merged[-1][1], b) - max(merged[-1][0], a) > 0.5 * min(merged[-1][1] - merged[-1][0], b - a):
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    h = mask.shape[0]
    return [Unit(a, b, (b - a) <= 0.15 * h) for a, b in merged]


def align(units: list[Unit], aks: list[str], type_cost: float = 0.5):
    """Assign consecutive units to aksharas. Returns ([(first, end) unit index per
    akshara], mean cost per akshara) or None if nothing fits."""
    m, n = len(units), len(aks)
    pats = [patterns(a) for a in aks]
    if any(not p for p in pats) or not (sum(min(len(p) for p, _ in ps) for ps in pats) <= m
                                         <= sum(max(len(p) for p, _ in ps) for ps in pats)):
        return None
    # Pixel width of one letter body, estimated from the units' own widths.
    u = sum(x.x1 - x.x0 for x in units) / max(sum(WIDTH[t] for t in pats_avg(pats)), 1e-6)
    INF = float("inf")
    dp = np.full((n + 1, m + 1), INF)
    back = np.zeros((n + 1, m + 1), int)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            for pat, prior in pats[i - 1]:
                k = len(pat)
                if k > j or dp[i - 1, j - k] == INF:
                    continue
                us = units[j - k : j]
                c = prior
                for t, x in zip(pat, us):
                    if (t in NARROW) != x.narrow:
                        c += type_cost
                exp_w = sum(WIDTH[t] for t in pat) * u
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


def pats_avg(pats: list[list[tuple[str, float]]]) -> str:
    """Unit types of each akshara's cheapest pattern, concatenated; used only to
    estimate the pixel width of one letter body."""
    return "".join(ps[0][0] for ps in pats)


def cut_word(mask: np.ndarray, text: str, max_cost: float = 0.3) -> tuple[list[AksharaCut], float] | None:
    """(akshara column ranges, mean alignment cost) for a word, or None if the
    alignment is not trustworthy."""
    aks = [a for a in aksharas(text) if not a.isspace()]
    # Digits and punctuation have no headline: split on whole components instead.
    letters = any(is_consonant(c) or is_independent_vowel(c) for c in text)
    units = body_units(mask) if letters else column_units(mask)
    if len(aks) < 2 or not units:
        return None
    res = align(units, aks)
    if res is None or res[1] > max_cost:
        return None
    ranges, cost = res
    cuts = []
    for t, (a, b) in enumerate(ranges):
        # Cut halfway through the gap between neighbouring aksharas.
        x0 = 0 if t == 0 else (units[a - 1].x1 + units[a].x0) // 2
        x1 = mask.shape[1] if t == len(ranges) - 1 else (units[b - 1].x1 + units[b].x0) // 2
        cuts.append(AksharaCut(aks[t], int(x0), int(x1)))
    return cuts, cost
