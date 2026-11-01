"""Fetch real handwritten Telugu pages with ground truth, spread over difficulty.

Source: ICDAR 2024 handwritten Indic competition, page-level task (PLHWTR-1.0),
Telugu training set (IIIT Hyderabad): phone photos of handwritten pages, each
with a human transcription. The zip is 3 GB, so only the chosen pages are read
from it with HTTP range requests.

    python scripts/fetch_real_telugu.py [--n 50] [--pool 200] [--out data/raw/telugu_real]

Difficulty is scored from the image alone (see `difficulty`), and the n pages
are drawn evenly from five tiers of a random pool. Writes images/, gt/ (the
transcription as released), manifest.jsonl (sentences, metrics, tier) and
contact sheets per tier.
"""
import argparse
import io
import json
import re
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hwgen.extract.ink import binarize, normalize  # noqa: E402

URL = ("https://ilocr.iiit.ac.in/icdar_2024_hwd/assets/dataset/"
       "PLHWTR-1.0/Training-Set/Page_Level_Telugu_Training_Set.zip")
TIERS = ["very_easy", "easy", "medium", "hard", "very_hard"]


# ---------------------------------------------------------------- remote zip

def _get(url: str, start: int, end: int, tries: int = 6) -> bytes:
    """Range read; the server drops connections under load, so retry with backoff."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    for t in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) == end - start + 1:
                return data
        except OSError:
            if t == tries - 1:
                raise
        time.sleep(2 ** t)
    raise OSError(f"short read {start}-{end}")


class _RangeFile(io.RawIOBase):
    """Just enough of a seekable file for zipfile to read the central directory."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or not len(b):
            return 0
        data = _get(self.url, self.pos, min(self.size, self.pos + len(b)) - 1)
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def list_members(url: str) -> list[zipfile.ZipInfo]:
    return zipfile.ZipFile(io.BufferedReader(_RangeFile(url), buffer_size=1 << 20)).infolist()


def read_member(url: str, info: zipfile.ZipInfo) -> bytes:
    """One member: its local header gives the data offset, then one range read."""
    head = _get(url, info.header_offset, info.header_offset + 29)
    name_len, extra_len = struct.unpack("<HH", head[26:30])
    start = info.header_offset + 30 + name_len + extra_len
    data = _get(url, start, start + info.compress_size - 1)
    if info.compress_type == zipfile.ZIP_DEFLATED:
        return zlib.decompress(data, -15)
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"unsupported compression {info.compress_type}")
    return data


# ---------------------------------------------------------------- ground truth

def parse_sentences(gt: str) -> list[dict]:
    """Split '17235 text. 17236 text.' into numbered sentences. Sentence ids
    are consecutive, so only the next expected id starts a new sentence;
    numbers inside a sentence ('10 రోజుల') are left alone."""
    gt = " ".join(gt.split())
    m = re.match(r"(\d+)\s", gt)
    if not m:
        return [{"id": None, "text": gt}]
    out, cur, pos = [], int(m.group(1)), m.end()
    while True:
        nxt = re.search(rf"(?:^|\s){cur + 1}\s", gt[pos:])
        end = pos + nxt.start() if nxt else len(gt)
        out.append({"id": cur, "text": gt[pos:end].strip()})
        if not nxt:
            return out
        cur, pos = cur + 1, pos + nxt.end()


# ---------------------------------------------------------------- difficulty

def _skew(binary: np.ndarray, max_deg: float = 8, step: float = 0.5) -> float:
    """Angle (deg) whose rotation makes text rows sharpest (projection-profile energy)."""
    h, w = binary.shape
    b = binary.astype(np.float32)
    best, best_a = -1.0, 0.0
    for a in np.arange(-max_deg, max_deg + step, step):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        e = float(np.var(cv2.warpAffine(b, M, (w, h)).sum(axis=1)))
        if e > best:
            best, best_a = e, a
    return float(best_a)


def difficulty(img: np.ndarray, gt_chars: int) -> dict:
    """Image measurements that make a page harder to read, at a fixed working
    size (long side 1360 px). Each is oriented so that larger = harder."""
    k = 1360 / max(img.shape[:2])
    small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.blur(cv2.dilate(gray, np.ones((41, 41), np.uint8)), (41, 41))
    norm, dark = normalize(small, 41)
    ink = binarize(dark)
    ink_dark = float(np.median(dark[ink])) if ink.any() else 0.0
    # Writing size: median height of letter-sized components.
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    hs = stats[1:, cv2.CC_STAT_HEIGHT][stats[1:, cv2.CC_STAT_AREA] >= 15]
    letter_h = float(np.median(hs)) if len(hs) else 0.0
    # Sharpness of the strokes: Laplacian energy on ink edges.
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    edges = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    sharp = float(lap[edges].var()) if edges.any() else 0.0
    # Clutter: dark non-paper regions (fingers, table, shadow wedges) as a fraction of the frame.
    clutter = float((bg < 0.55 * np.percentile(bg, 95)).mean())
    return {
        "shadow": round(float(np.std(bg) / max(np.mean(bg), 1)), 4),     # uneven lighting
        "faint_ink": round(1 - ink_dark, 4),                              # low ink contrast
        "blur": round(1 / (1 + sharp / 100), 4),                          # soft strokes
        "skew_deg": round(abs(_skew(ink)), 2),
        "small_writing": round(1 / max(letter_h, 1), 4),
        "density": round(gt_chars / max(ink.shape[0] * ink.shape[1] / 1e5, 1), 2),  # chars per 100k px
        "clutter": round(clutter, 4),
        "ink_px": int(ink.sum()),
        "letter_h_px": round(letter_h, 1),
    }


SCORED = ("shadow", "faint_ink", "blur", "skew_deg", "small_writing", "density", "clutter")


def add_scores(rows: list[dict]) -> None:
    """Robust z-score of each measurement over the pool, averaged."""
    for key in SCORED:
        v = np.array([r["metrics"][key] for r in rows], np.float64)
        med, mad = np.median(v), np.median(np.abs(v - np.median(v))) * 1.4826 or 1.0
        for r, z in zip(rows, (v - med) / mad):
            r.setdefault("z", {})[key] = round(float(np.clip(z, -3, 3)), 2)
    for r in rows:
        r["difficulty"] = round(float(np.mean(list(r["z"].values()))), 3)


def contact_sheet(paths: list[Path], labels: list[str], cols: int = 5, w: int = 300) -> np.ndarray:
    tiles = []
    for p, lab in zip(paths, labels):
        im = cv2.imread(str(p))
        im = cv2.resize(im, (w, int(im.shape[0] * w / im.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.rectangle(im, (0, 0), (w, 26), (255, 255, 255), -1)
        cv2.putText(im, lab, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 2)
        tiles.append(im)
    h = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, h - t.shape[0]), (2, 2), (0, 0)), constant_values=255) for t in tiles]
    tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % cols)
    return np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--pool", type=int, default=200, help="random candidates scored before picking")
    ap.add_argument("--out", type=Path, default=Path("data/raw/telugu_real"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    (a.out / "images").mkdir(parents=True, exist_ok=True)
    (a.out / "gt").mkdir(exist_ok=True)
    pool_dir = a.out / "pool"
    pool_dir.mkdir(exist_ok=True)

    infos = {i.filename: i for i in list_members(URL)}
    stems = sorted(f[:-4] for f in infos if f.endswith(".jpg") and f[:-4] + ".txt" in infos)
    print(f"{len(stems)} page/transcription pairs in the zip")
    rng = np.random.default_rng(a.seed)
    pool = [str(s) for s in rng.choice(stems, size=min(a.pool, len(stems)), replace=False)]

    def fetch(stem: str) -> dict | None:
        name = Path(stem).name
        img_p, gt_p = pool_dir / f"{name}.jpg", pool_dir / f"{name}.txt"
        if not (img_p.exists() and gt_p.exists()):  # write both only once both arrived
            img_b, gt_b = read_member(URL, infos[stem + ".jpg"]), read_member(URL, infos[stem + ".txt"])
            gt_p.write_bytes(gt_b)
            img_p.write_bytes(img_b)
        gt = gt_p.read_text(encoding="utf-8").strip()
        img = cv2.imread(str(img_p))
        if img is None or not gt:
            return None
        cv2.setNumThreads(1)
        return {"id": name, "source": stem, "width": img.shape[1], "height": img.shape[0],
                "gt": gt, "metrics": difficulty(img, len(gt.replace(" ", "")))}

    with ThreadPoolExecutor(a.workers) as ex:
        rows = [r for r in ex.map(fetch, pool) if r]
    print(f"{len(rows)} candidates scored")
    add_scores(rows)

    # Five equal-population tiers over the pool; draw n/5 from each, spread across the tier.
    rows.sort(key=lambda r: r["difficulty"])
    per = a.n // len(TIERS)
    chosen = []
    for t, tier_rows in enumerate(np.array_split(np.array(rows, dtype=object), len(TIERS))):
        idx = np.linspace(0, len(tier_rows) - 1, per + (1 if t < a.n % len(TIERS) else 0)).round().astype(int)
        for r in tier_rows[idx]:
            chosen.append({**r, "tier": TIERS[t]})

    with open(a.out / "manifest.jsonl", "w", encoding="utf-8") as fh:
        for r in chosen:
            (a.out / "images" / f"{r['id']}.jpg").write_bytes((pool_dir / f"{r['id']}.jpg").read_bytes())
            (a.out / "gt" / f"{r['id']}.txt").write_text(r["gt"] + "\n", encoding="utf-8")
            rec = {"id": r["id"], "tier": r["tier"], "difficulty": r["difficulty"],
                   "image": f"images/{r['id']}.jpg", "gt": f"gt/{r['id']}.txt", "source_zip_path": r["source"],
                   "width": r["width"], "height": r["height"], "text": r["gt"],
                   "sentences": parse_sentences(r["gt"]), "metrics": r["metrics"], "z": r["z"]}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    for t in TIERS:
        sel = [r for r in chosen if r["tier"] == t]
        sheet = contact_sheet([a.out / "images" / f"{r['id']}.jpg" for r in sel],
                              [f"{r['id']} d={r['difficulty']:+.2f}" for r in sel])
        cv2.imwrite(str(a.out / f"sheet_{t}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{len(chosen)} pages -> {a.out}")
    for t in TIERS:
        d = [r["difficulty"] for r in chosen if r["tier"] == t]
        print(f"  {t:10s} {len(d):3d} pages, difficulty {min(d):+.2f} .. {max(d):+.2f}")


if __name__ == "__main__":
    main()
