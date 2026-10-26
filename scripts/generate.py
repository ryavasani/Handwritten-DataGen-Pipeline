"""Generate synthetic handwritten Hindi FIR pages.

    python scripts/generate.py --n 10 --out outputs/synth [--seed 0] [--no-scan]

Writes page_XXXXX.jpg plus page_XXXXX.json (field texts, line/word boxes and
the provenance of every word: whole-word sample, assembled aksharas, or drawn
strokes).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hwgen.library import Library  # noqa: E402
from hwgen.synth.bank import SampleBank  # noqa: E402
from hwgen.synth.page import generate_page  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", type=Path, default=Path("data/library"))
    ap.add_argument("--out", type=Path, default=Path("outputs/synth"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-scan", action="store_true", help="skip scanner effects")
    a = ap.parse_args()

    bank = SampleBank(Library(a.lib))
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i in range(a.n):
        rng = np.random.default_rng([a.seed, i])  # each page reproducible on its own
        page, ann = generate_page(bank, rng, scan=not a.no_scan)
        ann["seed"] = [a.seed, i]
        stem = a.out / f"page_{a.seed:03d}_{i:05d}"
        cv2.imwrite(str(stem.with_suffix(".jpg")), page, [cv2.IMWRITE_JPEG_QUALITY, 95])
        stem.with_suffix(".json").write_text(json.dumps(ann, ensure_ascii=False, indent=1))
        if ann["skipped"]:
            print(f"{stem.name}: skipped fields {ann['skipped']}")
    print(f"{a.n} pages in {time.time() - t0:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
