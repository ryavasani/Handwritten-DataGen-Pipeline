"""Cut gold word samples into akshara samples and add them to the library.

    python scripts/build_library.py     # words first
    python scripts/build_aksharas.py [--max-cost 0.3]

Rerunning replaces previously added aksharas.
"""
import argparse
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hwgen.extract.akshara import cut_word  # noqa: E402
from hwgen.library import Library, Sample, write_sample  # noqa: E402

INHERITED = ("body_height", "stroke_width", "slant_deg", "ink_bgr", "ink_family", "ink_darkness")


def _drop_edge_fragments(sub: np.ndarray, headline_y: float, max_frac: float = 0.12) -> np.ndarray:
    """A column cut can catch slivers of the neighbouring akshara. Remove small
    components touching the left/right cut that do not reach the headline
    (the akshara's own strokes hang from the headline or are large)."""
    mask = (sub[..., 3] > 90).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    total = stats[1:, cv2.CC_STAT_AREA].sum()
    W = sub.shape[1]
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        at_edge = x == 0 or x + w >= W
        reaches_headline = y <= headline_y + 2 <= y + h
        if at_edge and not reaches_headline and area < max_frac * total:
            region = cv2.dilate((labels == i).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            sub[region, 3] = 0
    return sub


def _process(job):
    root, words, max_cost = job
    cv2.setNumThreads(1)
    lib = Library.__new__(Library)  # image loading only; skip reading the index
    lib.root = Path(root)
    out, tried, accepted = [], 0, 0
    for w in words:
        rgba = lib.image(w)
        mask = rgba[..., 3] > 90
        tried += 1
        res = cut_word(mask, w.text, max_cost)
        if res is None:
            continue
        accepted += 1
        cuts, cost = res
        for k, c in enumerate(cuts):
            sub = _drop_edge_fragments(rgba[:, c.x0 : c.x1].copy(), w.features["headline_y"])
            rows = np.nonzero((sub[..., 3] > 90).any(axis=1))[0]
            if len(rows) == 0:
                continue
            y0, y1 = max(0, rows[0] - 1), min(sub.shape[0], rows[-1] + 2)
            f = w.features
            feats = {
                "width": int(c.x1 - c.x0),
                "height": int(y1 - y0),
                # Positions relative to this crop, so aksharas can be aligned on a shared headline.
                "headline_y": round(f["headline_y"] - y0, 1),
                "baseline_y": int(f["baseline_y"] - y0),
                "position": k,
                "n_aksharas": len(cuts),
                "align_cost": round(cost, 3),
                **{key: f[key] for key in INHERITED if key in f},
            }
            s = Sample(f"{w.id}_a{k}", "akshara", c.text, "derived", w.source, w.writer, w.split,
                       script=w.script, parent_id=w.id, features=feats)
            out.append(write_sample(root, s, sub[y0:y1]))
    return out, tried, accepted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", type=Path, default=Path("data/library"))
    ap.add_argument("--max-cost", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    lib = Library(a.lib)
    lib.samples = [s for s in lib.samples if s.level != "akshara"]
    words = [s for s in lib.samples if s.level == "word" and s.label_status == "gold"]
    chunk = 1000
    jobs = [(a.lib, words[i : i + chunk], a.max_cost) for i in range(0, len(words), chunk)]
    added, tried, accepted = [], 0, 0
    with ProcessPoolExecutor(a.workers) as ex:
        for recs, t, acc in ex.map(_process, jobs):
            added += recs
            tried += t
            accepted += acc
    lib.samples += [Sample(**r) for r in added]
    lib.save_index()

    print(f"words cut: {accepted}/{tried} ({accepted / max(tried, 1):.0%}), aksharas added: {len(added)}")
    synth = Counter(r["text"] for r in added if r["split"] == "synth")
    print(f"distinct aksharas (synth): {len(synth)}; with >=5 samples: {sum(v >= 5 for v in synth.values())}")


if __name__ == "__main__":
    main()
