"""Degradation layers applied to a finished page image, in the order a real
document collects them:

1. physical damage   smudged ink, water damage, washed-out patches,
                     punch holes, a torn corner;
2. capture           phone photo (shadow, blur, perspective), flatbed scan
                     (slight rotation, dark bed edge, dust, a hair),
                     photocopy (grey toner, streaks, speckle) or fax
                     (low resolution, black and white), or archive (a
                     century-old sheet on a copy stand: tan crumbling paper
                     split along its folds, faded iron-gall ink, a rubber
                     stamp, photographed on a backdrop);
3. sharing           WhatsApp-style downscale and heavy JPEG.

`level` in [0, 1] sets how many layers are drawn and how strong they are.
Nothing moves the page except the capture geometry, which is returned as a
homography so word boxes can follow it.
"""
from __future__ import annotations

import cv2
import numpy as np

from .degrade import _smooth_noise
from .paper import photograph

BACKGROUNDS = [(40, 50, 70), (60, 60, 60), (25, 35, 55), (200, 200, 200)]  # table / scanner lid behind the sheet


def _paper_colour(page: np.ndarray) -> np.ndarray:
    return np.percentile(page.reshape(-1, 3), 90, axis=0).astype(np.float32)


def _blob(shape, rng, r_frac=(0.08, 0.2)) -> np.ndarray:
    """A soft irregular region in [0, 1]."""
    h, w = shape
    cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h
    r = rng.uniform(*r_frac) * w
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.hypot((xx - cx) / r, (yy - cy) / (r * rng.uniform(0.5, 1.5)))
    d += 0.4 * (_smooth_noise((h, w), r * 0.5, rng) - 0.5)
    return np.clip((1.15 - d) / 0.3, 0, 1)


def _motion_kernel(length: int, angle: float) -> np.ndarray:
    k = np.zeros((length, length), np.float32)
    c = length // 2
    dx, dy = np.cos(np.radians(angle)) * c, np.sin(np.radians(angle)) * c
    cv2.line(k, (int(c - dx), int(c - dy)), (int(c + dx), int(c + dy)), 1.0, 1, cv2.LINE_AA)
    return k / k.sum()


# ---------------------------------------------------------------- 1. physical

def smudge(page, rng, s):
    """A hand dragged through wet ink: ink smeared along one direction."""
    out = page.astype(np.float32)
    for _ in range(int(rng.integers(1, 3))):
        m = _blob(page.shape[:2], rng, (0.05, 0.12))[..., None] * rng.uniform(0.5, 0.9) * s
        smear = cv2.filter2D(out, -1, _motion_kernel(int(rng.uniform(9, 31)) | 1, rng.uniform(0, 180)))
        out = out * (1 - m) + np.minimum(out, smear * rng.uniform(0.93, 1.0)) * m
    return np.clip(out, 0, 255).astype(np.uint8)


def water_damage(page, rng, s):
    """Ink runs and blurs inside a wetted patch; the paper browns with a tide line."""
    out = page.astype(np.float32)
    m = _blob(page.shape[:2], rng, (0.1, 0.25))
    wet = cv2.GaussianBlur(out, (0, 0), rng.uniform(1.5, 3.5))
    mm = (m * rng.uniform(0.6, 1.0) * s)[..., None]
    out = out * (1 - mm) + wet * mm
    rim = np.clip(1 - np.abs(m - 0.15) / 0.1, 0, 1) * 0.12 * s
    out *= 1 - (rim[..., None] + 0.04 * s * m[..., None]) * np.float32([0.8, 0.5, 0.25])
    return np.clip(out, 0, 255).astype(np.uint8)


def washed_out(page, rng, s):
    """Ink faded towards the paper colour in a patch (light, damp, rubbing)."""
    out = page.astype(np.float32)
    m = (_blob(page.shape[:2], rng, (0.12, 0.3)) * rng.uniform(0.4, 0.8) * s)[..., None]
    out = out + (_paper_colour(page) - out) * m
    return np.clip(out, 0, 255).astype(np.uint8)


def holes_and_tear(page, rng, s, bg):
    """Two punched holes in the left margin and/or a torn-off corner."""
    out = page.copy()
    h, w = page.shape[:2]
    if rng.random() < 0.6:
        x, r = int(rng.uniform(0.02, 0.035) * w), int(0.012 * w)
        for fy in (0.33, 0.67):
            y = int(fy * h + rng.normal(0, 5))
            cv2.circle(out, (x + 2, y + 2), r + 2, tuple(int(c * 0.7) for c in page[y, x]), -1, cv2.LINE_AA)
            cv2.circle(out, (x, y), r, bg, -1, cv2.LINE_AA)
    if rng.random() < 0.5 * s + 0.2:
        # Jagged tear across one corner.
        cx, cy = int(rng.choice([0, w])), int(rng.choice([0, h]))
        a, b = rng.uniform(0.04, 0.1) * w, rng.uniform(0.04, 0.1) * h
        t = np.linspace(0, 1, 24)
        xs = cx + (1 - t) * a * (1 if cx == 0 else -1) + rng.normal(0, 3, 24)
        ys = cy + t * b * (1 if cy == 0 else -1) + rng.normal(0, 3, 24)
        pts = np.concatenate([np.stack([xs, ys], 1), [[cx, cy]]]).astype(np.int32)
        cv2.fillPoly(out, [pts], bg, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------- 2. capture

def phone(page, rng, s, bg):
    """Phone photo: a soft shadow (hand or phone), slight defocus, perspective."""
    h, w = page.shape[:2]
    out = page.astype(np.float32)
    if rng.random() < 0.7:
        shade = np.zeros((h, w), np.float32)
        x0 = rng.uniform(0.3, 0.9) * w
        poly = np.int32([[x0, 0], [x0 + rng.uniform(-0.3, 0.3) * w, h], [w, h], [w, 0]])
        if rng.random() < 0.5:
            poly[:, 0] = w - poly[:, 0]
        cv2.fillPoly(shade, [poly], 1.0)
        shade = cv2.GaussianBlur(shade, (0, 0), rng.uniform(30, 90))
        out *= 1 - shade[..., None] * rng.uniform(0.15, 0.4) * s
    out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 0.6 + 1.0 * s))
    page, H = photograph(np.clip(out, 0, 255).astype(np.uint8), rng, strength=0.3 + 0.7 * s)
    return page, H


def flatbed(page, rng, s, bg):
    """Flatbed scan: small rotation, a dark strip along one edge, dust and a hair."""
    h, w = page.shape[:2]
    A = cv2.getRotationMatrix2D((w / 2, h / 2), rng.normal(0, 0.8), 1.0)
    out = cv2.warpAffine(page, A, (w, h), borderValue=(25, 25, 25)).astype(np.float32)
    edge = int(rng.uniform(0, 0.015) * w)
    if edge:
        side = rng.integers(4)
        ramp = np.linspace(0.15, 1, edge, dtype=np.float32)
        if side == 0:
            out[:, :edge] *= ramp[None, :, None]
        elif side == 1:
            out[:, w - edge :] *= ramp[::-1][None, :, None]
        elif side == 2:
            out[:edge] *= ramp[:, None, None]
        else:
            out[h - edge :] *= ramp[::-1][:, None, None]
    for _ in range(rng.poisson(30 * s)):
        cv2.circle(out, (int(rng.uniform(0, w)), int(rng.uniform(0, h))), int(rng.integers(1, 3)),
                   (40, 40, 40), -1, cv2.LINE_AA)
    if rng.random() < 0.4 * s:
        t = np.linspace(0, 1, 30)
        x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
        pts = np.stack([x0 + t * rng.uniform(-200, 200) + 15 * np.sin(t * 6),
                        y0 + t * rng.uniform(-200, 200)], 1).astype(np.int32)
        cv2.polylines(out, [pts], False, (50, 50, 50), 1, cv2.LINE_AA)
    out += rng.normal(0, 2, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8), np.vstack([A, [0, 0, 1]])


def photocopy(page, rng, s, bg):
    """Photocopy: grey, high contrast (faint ink drops out or goes darker),
    toner speckle and drum streaks."""
    h, w = page.shape[:2]
    g = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    paper = np.percentile(g, 90)
    g = np.clip(g / max(paper, 1e-3), 0, 1)
    k = rng.uniform(6, 14) + 8 * s  # contrast of the copier's threshold curve
    g = 1 / (1 + np.exp(-k * (g - rng.uniform(0.55, 0.75))))
    g = g * rng.uniform(0.85, 0.97) + 0.02
    speck = rng.random((h, w)) < 0.0015 * s
    g[speck] *= 0.2
    for _ in range(int(rng.integers(0, 4))):
        x = int(rng.uniform(0, w))
        g[:, x : x + int(rng.integers(1, 4))] *= rng.uniform(0.75, 0.95)
    g = cv2.GaussianBlur(g, (0, 0), 0.6)
    out = cv2.cvtColor((np.clip(g, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return out, np.eye(3)


def fax(page, rng, s, bg):
    """Fax: ~100 dpi, hard black/white, a few dropped scanlines."""
    h, w = page.shape[:2]
    g = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    k = rng.uniform(0.45, 0.65)
    small = cv2.resize(g, (int(w * k), int(h * k * rng.uniform(0.5, 1.0))), interpolation=cv2.INTER_AREA)
    t, _ = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = np.where(small > min(255, t * rng.uniform(1.0, 1.12)), 255, 0).astype(np.uint8)
    for _ in range(int(rng.integers(0, 4))):
        bw[int(rng.uniform(0, bw.shape[0]))] = 255
    out = cv2.resize(bw, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR), np.eye(3)


def _ragged(n: int, rng, cells=(70, 20, 6)) -> np.ndarray:
    """1-D multi-scale noise in [0, 1] (a torn or crumbled outline)."""
    out = sum(wt * _smooth_noise((1, n), c, rng)[0] for wt, c in zip((0.65, 0.28, 0.07), cells))
    return out.astype(np.float32)


def _crease(out, pos, horizontal, depth, rng):
    """A deep fold: dark valley with a lit and a shaded side, uneven along its length."""
    h, w = out.shape[:2]
    L, n = (h, w) if horizontal else (w, h)
    c = np.arange(L, dtype=np.float32) - pos
    prof = (-0.18 * np.exp(-(c / 2.5) ** 2) + 0.07 * np.tanh(c / 30) * np.exp(-np.abs(c) / 80)) * depth
    along = 0.6 + 0.8 * _smooth_noise((1, n), 150, rng)[0]
    f = prof[:, None] * along[None, :]
    out *= (1 + (f if horizontal else f.T))[..., None]


def _stamp(h, w, rng) -> np.ndarray:
    """Matte of a worn oval rubber stamp: double rim, a ring of capitals, a line inside."""
    S = 2
    a, b = int(rng.uniform(0.11, 0.15) * w), 0
    b = int(a * rng.uniform(0.45, 0.6))
    cx, cy = int(rng.uniform(0.68, 0.86) * w), int(rng.uniform(0.04, 0.1) * h + b)
    ang = rng.normal(0, 8)
    m = np.zeros((h * S, w * S), np.uint8)
    for k, t in ((1.0, 3), (0.93, 2), (0.62, 2)):
        cv2.ellipse(m, (cx * S, cy * S), (int(a * k * S), int(b * k * S)), ang, 0, 360, 255, t * S, cv2.LINE_AA)
    letters = "ABCDEFGHIKLMNOPRSTUV"
    ring = " ".join("".join(rng.choice(list(letters), int(rng.integers(3, 8)))) for _ in range(4))
    fs = b * S * 0.009
    for i, ch in enumerate(ring):
        th = np.radians(200 + 150 * i / max(len(ring) - 1, 1))  # along the upper arc
        px, py = a * 0.78 * np.cos(th), b * 0.76 * np.sin(th)
        rot = np.radians(ang)
        x = cx + px * np.cos(rot) - py * np.sin(rot)
        y = cy + px * np.sin(rot) + py * np.cos(rot)
        tile = np.zeros((int(60 * fs) + 4, int(60 * fs) + 4), np.uint8)
        cv2.putText(tile, ch, (2, int(45 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs, 255, max(1, int(2 * fs)), cv2.LINE_AA)
        R = cv2.getRotationMatrix2D((tile.shape[1] / 2, tile.shape[0] / 2), -(np.degrees(th) + 90 + ang), 1.0)
        tile = cv2.warpAffine(tile, R, tile.shape[::-1])
        x0, y0 = int(x * S - tile.shape[1] / 2), int(y * S - tile.shape[0] / 2)
        if 0 <= x0 and 0 <= y0 and x0 + tile.shape[1] <= w * S and y0 + tile.shape[0] <= h * S:
            np.maximum(m[y0 : y0 + tile.shape[0], x0 : x0 + tile.shape[1]], tile, out=m[y0 : y0 + tile.shape[0], x0 : x0 + tile.shape[1]])
    mid = "".join(rng.choice(list(letters), int(rng.integers(4, 7))))
    fs2 = b * S * 0.012
    (tw, th_), _ = cv2.getTextSize(mid, cv2.FONT_HERSHEY_SIMPLEX, fs2, max(1, int(2 * fs2)))
    cv2.putText(m, mid, (cx * S - tw // 2, cy * S + th_ // 2), cv2.FONT_HERSHEY_SIMPLEX, fs2, 255,
                max(1, int(2 * fs2)), cv2.LINE_AA)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    # Uneven inking and wear: patches print faint or not at all.
    wear = np.clip((_smooth_noise((h, w), 0.25 * a, rng) - 0.15) / 0.6, 0, 1)
    return cv2.GaussianBlur(m * wear, (0, 0), 0.8)


def archive(page, rng, s, bg):
    """A century-old sheet photographed on a copy stand: tan, mottled paper with
    crumbled edges, split along its folds, faded brown iron-gall ink, a worn
    rubber stamp; the sheet lies on a blue-grey backdrop."""
    h, w = page.shape[:2]
    f = page.astype(np.float32)
    # Separate ink from paper: the local paper colour is the brightest nearby.
    k = max(15, int(0.02 * w)) | 1
    paper_est = cv2.GaussianBlur(cv2.dilate(f, np.ones((k, k), np.uint8)), (0, 0), k / 2)
    lum_p = paper_est.mean(2)
    ink = np.clip((1 - f.mean(2) / np.maximum(lum_p, 1)) / 0.6, 0, 1)

    # Paper: tan stock keeping the old sheet's stains, then mottling, grime and tide marks.
    r = rng.uniform(185, 212)
    tan = np.float32([r * rng.uniform(0.45, 0.6), r * rng.uniform(0.76, 0.82), r])
    out = tan * (lum_p / np.median(lum_p))[..., None]
    brown = np.float32([0.9, 0.7, 0.45])  # blue fades first: darker means browner
    dark = 0.2 * s * _smooth_noise((h, w), rng.uniform(150, 300), rng) + 0.08 * _smooth_noise((h, w), 30, rng)
    for _ in range(int(rng.integers(2, 5))):
        m = _blob((h, w), rng, (0.06, 0.2))
        dark += (np.clip(1 - np.abs(m - 0.2) / 0.12, 0, 1) * 0.12 + 0.06 * m) * s
    out *= 1 - dark[..., None] * brown
    out += rng.normal(0, 1, (h, w, 1)).astype(np.float32) * 4  # fibre grain

    # Ink: faded iron-gall brown, unevenly.
    ink_col = np.float32([rng.uniform(25, 50), rng.uniform(35, 55), rng.uniform(45, 75)])
    a = ink * rng.uniform(0.75, 0.95) * (1 - 0.4 * s * _smooth_noise((h, w), 200, rng))
    out = out * (1 - a[..., None]) + ink_col * a[..., None]

    # Rubber stamp in faded violet, struck over the top corner.
    if rng.random() < 0.85:
        st = _stamp(h, w, rng)[..., None] * rng.uniform(0.45, 0.75)
        out = out * (1 - st) + out * np.float32([0.62, 0.42, 0.5]) * st

    # Folds: across the middle and down the centre, plus a few wrinkles.
    fy = h * (0.5 + rng.normal(0, 0.01))
    fx = w * (0.5 + rng.normal(0, 0.01))
    _crease(out, fy, True, rng.uniform(0.8, 1.3), rng)
    _crease(out, fx, False, rng.uniform(0.5, 1.0), rng)
    if rng.random() < 0.5:
        _crease(out, h * rng.choice([0.25, 0.75]), True, rng.uniform(0.3, 0.6), rng)
    for _ in range(rng.poisson(8 * s)):
        t = np.linspace(0, 1, 40)
        x0, y0, L, th = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.05, 0.25) * w, rng.uniform(0, np.pi)
        pts = np.stack([x0 + t * L * np.cos(th), y0 + t * L * np.sin(th)], 1) + np.cumsum(rng.normal(0, 1.2, (40, 2)), 0)
        line = np.zeros((h, w), np.float32)
        cv2.polylines(line, [pts.astype(np.int32)], False, 1.0, 1, cv2.LINE_AA)
        lit = np.roll(line, 2, axis=0)
        out *= (1 - cv2.GaussianBlur(line, (0, 0), 1.0) * 0.18 * s + cv2.GaussianBlur(lit, (0, 0), 1.2) * 0.06)[..., None]

    # What is left of the sheet: crumbled edges, chunks bitten out, splits along the folds, holes.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    depth = (0.008 + 0.022 * s) * w
    lost = np.zeros((h, w), bool)
    for side in range(4):  # each edge has its own crumbled outline
        n = w if side < 2 else h
        r = depth * (0.15 + 2.4 * _ragged(n, rng) ** 2)  # mostly shallow, now and then a deep chip
        if side == 0:
            lost |= yy < r[None, :]
        elif side == 1:
            lost |= (h - 1 - yy) < r[None, :]
        elif side == 2:
            lost |= xx < r[:, None]
        else:
            lost |= (w - 1 - xx) < r[:, None]
    for _ in range(rng.poisson(4 * s) + 1):  # bigger bites from the edge
        side = rng.integers(4)
        cx = rng.uniform(0, w) if side < 2 else (0 if side == 2 else w)
        cy = rng.uniform(0, h) if side >= 2 else (0 if side == 0 else h)
        r = rng.uniform(0.015, 0.045) * w
        d = np.hypot(xx - cx, yy - cy) / r + 0.5 * (_smooth_noise((h, w), r * 0.4, rng) - 0.5)
        lost |= d < 1
    # The horizontal fold has split from both edges, leaving a few bridges.
    yc = fy + np.cumsum(rng.normal(0, 0.6, w)).astype(np.float32)
    edge = np.clip(1 - np.minimum(np.arange(w), w - 1 - np.arange(w)) / (0.35 * w), 0, 1)
    open_ = np.clip((_ragged(w, rng, (90, 25, 6)) - 0.25 + 0.5 * edge) / 0.5, 0, 1)
    hw = (1.5 + 5 * s) * open_ * (rng.random() < 0.8 + 0.2 * s)
    lost |= np.abs(yy - yc[None, :]) < hw[None, :]
    # The vertical fold has split down from the top and out from the middle split.
    for y0, y1 in ((0, rng.uniform(0.04, 0.14) * h), (fy - rng.uniform(0, 0.05) * h, fy + rng.uniform(0, 0.05) * h)):
        ys = np.arange(int(max(y0, 0)), int(min(y1, h)))
        if len(ys) < 2:
            continue
        xc = fx + np.cumsum(rng.normal(0, 0.5, len(ys)))
        # Widest where the split starts (the top edge, or the horizontal split).
        start, span = (0, y1) if y0 == 0 else (fy, (y1 - y0) / 2)
        taper = np.clip(1 - np.abs(ys - start) / max(span, 1), 0, 1)
        half = (1 + 4 * s) * taper
        lost[ys] |= np.abs(xx[ys] - xc[:, None]) < half[:, None]
    for _ in range(rng.poisson(3 * s)):  # brittle holes, mostly near the folds
        cx = fx + rng.normal(0, 0.15 * w) if rng.random() < 0.5 else rng.uniform(0.05, 0.95) * w
        cy = fy + rng.normal(0, 0.03 * h) if rng.random() < 0.6 else rng.uniform(0.05, 0.95) * h
        r = rng.uniform(0.003, 0.012) * w
        d = np.hypot(xx - cx, yy - cy) / r + 0.6 * (_smooth_noise((h, w), max(r * 0.5, 2), rng) - 0.5)
        lost |= d < 1
    keep = (~lost).astype(np.uint8)
    dist = cv2.distanceTransform(keep, cv2.DIST_L2, 3)
    out *= 1 - (0.35 * np.exp(-dist / (0.012 * w)) * s)[..., None] * brown   # brown, brittle rim
    pale = tan * 1.12
    fringe = (np.exp(-dist / 1.5) * 0.5)[..., None]                              # light torn fibres
    out = out + (pale - out) * fringe
    alpha = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 0.7)

    # Lay the sheet on the backdrop, with a soft shadow beneath it.
    sc = rng.uniform(0.86, 0.92)
    M = np.float32([[sc, 0, (1 - sc) * w / 2 + rng.normal(0, 0.01 * w)],
                    [0, sc, (1 - sc) * h / 2 + rng.normal(0, 0.01 * h)]])
    sheet = cv2.warpAffine(np.clip(out, 0, 255), M, (w, h))
    al = cv2.warpAffine(alpha, M, (w, h))[..., None]
    shadow = cv2.GaussianBlur(cv2.warpAffine(alpha, M + np.float32([[0, 0, 4], [0, 0, 6]]), (w, h)), (0, 0), 8)[..., None]
    back = np.float32([rng.uniform(140, 165), rng.uniform(122, 140), rng.uniform(100, 118)])
    back = back * (0.92 + 0.12 * _smooth_noise((h, w), 0.7 * w, rng))[..., None]
    back *= 1 - 0.35 * shadow
    res = back * (1 - al) + sheet * al
    res, Hp = photograph(np.clip(res, 0, 255).astype(np.uint8), rng, strength=0.3 + 0.3 * s)
    return res, Hp @ np.vstack([M, [0, 0, 1]])


CAPTURES = {"phone": phone, "flatbed": flatbed, "photocopy": photocopy, "fax": fax, "archive": archive}


# ---------------------------------------------------------------- 3. sharing

def whatsapp(page, rng, s):
    """Forwarded on a messaging app: downscaled, heavy JPEG, scaled back up
    (so coordinates are unchanged)."""
    h, w = page.shape[:2]
    k = rng.uniform(0.55, 0.85)
    small = cv2.resize(page, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    q = int(rng.uniform(30, 60))
    small = cv2.imdecode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def degrade_page(page: np.ndarray, rng: np.random.Generator, level: float,
                 capture: str | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (degraded page, homography page -> output, names of the layers applied)."""
    s = float(np.clip(level, 0, 1))
    bg = tuple(int(c) for c in BACKGROUNDS[rng.integers(len(BACKGROUNDS))])
    applied = []
    for name, fn, p in (("smudge", smudge, 0.5), ("water_damage", water_damage, 0.35), ("washed_out", washed_out, 0.4)):
        if rng.random() < p * s:
            page = fn(page, rng, s)
            applied.append(name)
    if capture != "archive" and rng.random() < 0.5 * s:  # the archive sheet has its own losses
        page = holes_and_tear(page, rng, s, bg)
        applied.append("holes_or_tear")
    if capture is None:
        names, p = ["phone", "flatbed", "photocopy", "fax"], np.array([0.5, 0.25, 0.17, 0.08 * s])
        capture = str(rng.choice(names, p=p / p.sum()))
    page, H = CAPTURES[capture](page, rng, s, bg)
    applied.append(capture)
    if capture != "archive" and rng.random() < 0.6 * s:
        page = whatsapp(page, rng, s)
        applied.append("whatsapp")
    return page, H, applied
