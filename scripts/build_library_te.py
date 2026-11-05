"""Build the Telugu handwriting sample library from real page images.

    python scripts/fetch_real_telugu.py                   # the 50 pages first
    python scripts/build_library_te.py [--data data/raw/telugu_real] [--out data/library_te]
    python scripts/build_library_te.py --pool             # also use the 200-page candidate pool

Each page is one writer. Google Vision finds word boxes (one call per page,
cached in data/cache/gocr); a box gets the ground-truth word it aligns to
when Vision read it within --max-dist (character edit distance / length).
Words are then cut into aksharas, and a conjunct's vattu is also stored on
its own (level "vattu") so the synthesiser can build conjuncts it has no
sample of. Every --ref-every'th page is held out as the "reference" split.
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hwgen.extract.features import measure_telugu  # noqa: E402
from hwgen.extract.telugu_akshara import cut_word  # noqa: E402
from hwgen.extract.telugu_page import PUNCT, align_tokens, crop_word, prepare_page, tokenize  # noqa: E402
from hwgen.library import Sample, write_sample  # noqa: E402
from hwgen.text.telugu import aksharas  # noqa: E402
from hwgen.verify.gocr import GoogleOCR  # noqa: E402

WORD_H = 110  # pages are rescaled so a typical Vision word box is this tall
INHERITED = ("body_height", "stroke_width", "stroke_over_body", "slant_deg", "ink_bgr", "ink_family", "ink_darkness")


def load_pages(data: Path, pool: bool) -> list[dict]:
    pages = {}
    for line in (data / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        pages[r["id"]] = {"id": r["id"], "image": data / r["image"], "text": r["text"]}
    if pool:
        for img in sorted((data / "pool").glob("*.jpg")):
            if img.stem not in pages and img.with_suffix(".txt").exists():
                pages[img.stem] = {"id": img.stem, "image": img,
                                   "text": img.with_suffix(".txt").read_text(encoding="utf-8").strip()}
    return list(pages.values())


def _drop_trailing_punct(rgba: np.ndarray) -> np.ndarray:
    """Vision boxed a word together with its full stop or comma: remove small
    components at the right end of the crop."""
    mask = (rgba[..., 3] > 90).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 2:
        return rgba
    total = stats[1:, cv2.CC_STAT_AREA].sum()
    right = max(stats[i, 0] + stats[i, 2] for i in range(1, n))
    out = rgba.copy()
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 0.05 * total and x + w >= right - 2 and x > 0.7 * rgba.shape[1]:
            out[labels == i, 3] = 0
    cols = np.nonzero((out[..., 3] > 90).any(axis=0))[0]
    return out[:, : cols[-1] + 4] if len(cols) else out


def _drop_foreign(rgba: np.ndarray, zone: tuple[int, int]) -> np.ndarray:
    """Remove ink from neighbouring lines: components that do not touch the
    main zone and are either far from it or letter-sized."""
    z0, z1 = zone
    zh = max(z1 - z0, 1)
    mask = (rgba[..., 3] > 90).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = rgba.copy()
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if y + h > z0 and y < z1:
            continue
        far = y + h < z0 - 0.9 * zh or y > z1 + 1.3 * zh
        if far or h > 0.9 * zh:
            out[labels == i, 3] = 0
    ys, xs = np.nonzero(out[..., 3] > 90)
    if len(xs) == 0:
        return out
    return out[max(0, ys.min() - 3) : ys.max() + 4, max(0, xs.min() - 3) : xs.max() + 4]


def _faint(rgba: np.ndarray) -> bool:
    """Pencil, a dry pen or a blurred photo: little of the ink is solid."""
    a = rgba[..., 3]
    ink = a > 90
    return ink.sum() == 0 or (a[ink] > 200).mean() < 0.35


def _shape(a: np.ndarray, n: int = 24) -> np.ndarray:
    """Ink shape descriptor: the crop padded to a square, shrunk and blurred."""
    h, w = a.shape
    s = max(h, w)
    sq = np.zeros((s, s), np.float32)
    sq[(s - h) // 2 : (s - h) // 2 + h, (s - w) // 2 : (s - w) // 2 + w] = a / 255.0
    d = cv2.GaussianBlur(cv2.resize(sq, (n, n), interpolation=cv2.INTER_AREA), (0, 0), 1.0).ravel()
    return d / max(np.linalg.norm(d), 1e-6)


def drop_shape_outliers(root: Path, records: list[dict], min_n: int = 4, k: float = 2.0) -> list[dict]:
    """A miscut akshara (a neighbour's stroke, half a letter) looks unlike the
    other samples of its label. Drop samples whose shape is far from the
    label's medoid: more than k times the label's median distance."""
    by = {}
    for r in records:
        if r["level"] == "akshara":
            by.setdefault(r["text"], []).append(r)
    bad = set()
    for rs in by.values():
        if len(rs) < min_n:
            continue
        D = np.stack([_shape(cv2.imread(str(root / f"akshara/{r['id']}.png"), cv2.IMREAD_GRAYSCALE)) for r in rs])
        dist = 1 - D @ D.T
        med = dist[int(dist.sum(1).argmin())]
        lim = k * max(float(np.median(med)), 0.02)
        bad |= {r["id"] for r, d in zip(rs, med) if d > lim}
    for i in bad:
        (root / f"akshara/{i}.png").unlink()
    return [r for r in records if r["id"] not in bad]


def _fragmented(mask: np.ndarray, n_aks: int) -> bool:
    """Faint pencil or a weak binarisation breaks strokes into many pieces."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    big = (stats[1:, cv2.CC_STAT_AREA] >= 0.02 * stats[1:, cv2.CC_STAT_AREA].sum()).sum()
    return big > 2.2 * n_aks + 2


def _sub_sample(rgba, piece_mask, f, sid, level, text, parent: Sample, extra: dict):
    m = cv2.dilate(piece_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ys, xs = np.nonzero(piece_mask)
    if len(xs) == 0:
        return None
    y0, y1 = max(0, ys.min() - 1), min(m.shape[0], ys.max() + 2)
    x0, x1 = max(0, xs.min() - 1), min(m.shape[1], xs.max() + 2)
    sub = rgba[y0:y1, x0:x1].copy()
    sub[..., 3] = (sub[..., 3] * m[y0:y1, x0:x1]).astype(np.uint8)
    feats = {"width": int(x1 - x0), "height": int(y1 - y0),
             "headline_y": round(f["headline_y"] - y0, 1), "baseline_y": int(f["baseline_y"] - y0),
             "x0": int(x0), **{k: f[k] for k in INHERITED if k in f}, **extra}
    s = Sample(sid, level, text, "derived", parent.source, parent.writer, parent.split,
               script="telu", parent_id=parent.id, features=feats)
    return s, sub


def _process(job):
    page, words, split, out, max_dist = job
    cv2.setNumThreads(1)
    img = cv2.imread(str(page["image"]))
    hmed = float(np.median([w["box"][3] for w in words if len(w["text"]) > 2] or [WORD_H]))
    k = WORD_H / hmed
    img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    norm, dark, ink = prepare_page(img, WORD_H)

    # Vision tokens, remembering which Vision word (box) each came from.
    vis, src = [], []
    for wi, w in enumerate(words):
        for t in tokenize(w["text"]):
            vis.append(t)
            src.append(wi)
    gt = tokenize(page["text"])
    stats = Counter(gt_words=sum(t not in PUNCT for t in gt))
    records = []
    for gi, js, dist in align_tokens(vis, gt):
        word = gt[gi]
        if word in PUNCT:
            continue
        stats["aligned"] += 1
        if dist > max_dist:
            continue
        wis = sorted({src[j] for j in js})
        # The Vision word(s) must hold nothing but this token (plus trailing punctuation).
        own = [vis[j] for j in range(len(vis)) if src[j] in wis]
        letters = [t for t in own if t not in PUNCT]
        if len(letters) != len(js) or (own and own[0] in PUNCT):
            stats["shared_box"] += 1
            continue
        boxes = np.array([[v * k for v in words[wi]["box"]] for wi in wis])
        x0, y0 = boxes[:, 0].min(), boxes[:, 1].min()
        x1, y1 = (boxes[:, 0] + boxes[:, 2]).max(), (boxes[:, 1] + boxes[:, 3]).max()
        if y1 - y0 > 1.8 * WORD_H or len({words[wi]["line"] for wi in wis}) > 1:
            stats["two_lines"] += 1
            continue
        res = crop_word(norm, dark, ink, [int(x0), int(y0), int(np.ceil(x1 - x0)), int(np.ceil(y1 - y0))])
        if res is None:
            continue
        rgba, _ = res
        if len(own) > len(letters):
            rgba = _drop_trailing_punct(rgba)
        mask = rgba[..., 3] > 90
        aks = [a for a in aksharas(word) if not a.isspace()]
        if _faint(rgba):
            stats["faint"] += 1
            continue
        if _fragmented(mask, len(aks)):
            stats["fragmented"] += 1
            continue
        f = measure_telugu(rgba, mask)
        if not f:
            continue
        rgba = _drop_foreign(rgba, (int(f["headline_y"]), f["baseline_y"]))
        mask = rgba[..., 3] > 90
        f = measure_telugu(rgba, mask)
        if not f:
            continue
        s = Sample(f"te_{page['id']}_{gi:03d}", "word", word, "gold", str(page["image"].name), page["id"], split,
                   script="telu", features={**f, "ocr_dist": round(dist, 3)})
        records.append(write_sample(out, s, rgba))
        stats["words"] += 1

        # Aksharas: a one-akshara word is itself a sample; longer words are cut.
        if len(aks) == 1:
            sub = _sub_sample(rgba, mask, f, f"{s.id}_a0", "akshara", aks[0], s, {"position": 0, "n_aksharas": 1})
            if sub:
                records.append(write_sample(out, *sub))
                stats["aksharas"] += 1
            continue
        cut = cut_word(mask, word, (int(f["headline_y"]), f["baseline_y"]))
        if cut is None:
            continue
        pieces, cost = cut
        stats["words_cut"] += 1
        for t, p in enumerate(pieces):
            sub = _sub_sample(rgba, p.mask, f, f"{s.id}_a{t}", "akshara", p.text, s,
                              {"position": t, "n_aksharas": len(pieces), "align_cost": round(cost, 3)})
            if sub is None:
                continue
            records.append(write_sample(out, *sub))
            stats["aksharas"] += 1
            ax0 = sub[0].features["x0"]
            for cons, vm in p.vattus:
                v = _sub_sample(rgba, vm, f, f"{s.id}_a{t}_v", "vattu", cons, s, {})
                if v is None:
                    continue
                vf = v[0].features
                body = f["body_height"]
                # Where the vattu sits relative to its letter: x from the letter's left
                # edge and y from the baseline, both in body heights.
                vf["dx_body"] = round((vf["x0"] - ax0) / body, 3)
                vf["dy_body"] = round(-vf["baseline_y"] / body, 3)  # top of the vattu crop
                vf["letter_w_body"] = round(sub[0].features["width"] / body, 3)
                records.append(write_sample(out, *v))
                stats["vattus"] += 1
    return records, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/raw/telugu_real"))
    ap.add_argument("--out", type=Path, default=Path("data/library_te"))
    ap.add_argument("--pool", action="store_true", help="also use the pages in <data>/pool")
    ap.add_argument("--max-dist", type=float, default=0.25)
    ap.add_argument("--ref-every", type=int, default=10, help="hold out every Nth page as reference (0: none)")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    pages = load_pages(a.data, a.pool)
    ocr = GoogleOCR(language="te")
    with ThreadPoolExecutor(4) as ex:
        words = list(ex.map(lambda p: ocr.page_words(cv2.imread(str(p["image"]))), pages))
    print(f"{len(pages)} pages; Vision calls {ocr.calls}, cached {ocr.cached}")

    if a.out.exists():
        shutil.rmtree(a.out)
    a.out.mkdir(parents=True)
    jobs = [(p, w, "reference" if a.ref_every and i % a.ref_every == a.ref_every - 1 else "synth", a.out, a.max_dist)
            for i, (p, w) in enumerate(zip(pages, words))]
    records, total = [], Counter()
    with ProcessPoolExecutor(a.workers) as ex:
        for recs, st in ex.map(_process, jobs):
            records += recs
            total += st
    n0 = len(records)
    records = drop_shape_outliers(a.out, records)
    total["shape_outliers"] = n0 - len(records)
    with open(a.out / "index.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(dict(total))
    synth = Counter(r["text"] for r in records if r["level"] == "akshara" and r["split"] == "synth")
    print(f"distinct aksharas (synth): {len(synth)}; with >=3 samples: {sum(v >= 3 for v in synth.values())}; "
          f"vattu consonants: {len({r['text'] for r in records if r['level'] == 'vattu'})}")


if __name__ == "__main__":
    main()
