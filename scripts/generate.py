"""Generate synthetic handwritten Hindi or Telugu pages.

    python scripts/generate.py --n 10 [--seed 0]                # -> outputs/documents_hi
    python scripts/generate.py --messiness 0.9 --age 0.8    # pin the difficulty
    python scripts/generate.py --degrade 0.7 [--capture photocopy] # add damage/scan/sharing layers
    python scripts/generate.py --script te --n 10            # Telugu documents (data/library_te)
    python scripts/generate.py --script te --size 0.65 --spacing 0.7 --irregular 0.7   # small, uneven hand
    python scripts/generate.py --script te --degrade 0.8 --capture archive --tight 0.7   # old torn sheet, dense hand

Each page is a plain handwritten petition/affidavit on aged court paper or
notebook paper (plain or ruled). Writes page_XXXXX.jpg plus page_XXXXX.json
(block texts, line/word boxes and the provenance of every word: whole-word
sample, assembled aksharas, or drawn strokes).
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
from hwgen.synth.document import generate_document  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", choices=["hi", "te"], default="hi")
    ap.add_argument("--lib", type=Path, help="sample library; data/library (hi) or data/library_te (te) if unset")
    ap.add_argument("--out", type=Path, help="outputs/documents_hi or outputs/documents_te if unset")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--messiness", type=float, help="0 (neat) to 1 (cramped scrawl); random 0.4-1 if unset")
    ap.add_argument("--age", type=float, help="paper age 0-1; random if unset")
    ap.add_argument("--paper", choices=["aged", "notebook"], help="court paper or notebook; random if unset")
    ap.add_argument("--degrade", type=float, default=0.0,
                    help="damage + capture + sharing layers, 0 (off) to 1 (heavy)")
    ap.add_argument("--capture", choices=["phone", "flatbed", "photocopy", "fax", "archive"],
                    help="with --degrade: how the page was digitised; random if unset")
    ap.add_argument("--no-scan", action="store_true", help="skip scanner/photo effects")
    ap.add_argument("--size", type=float, default=1.0,
                    help="letter height x this (e.g. 0.65 = small writing, more text per page)")
    ap.add_argument("--spacing", type=float, default=0.0, help="uneven word/letter spacing, 0-1")
    ap.add_argument("--irregular", type=float, default=0.0,
                    help="random slant per word; per-letter size/tilt/lean/bounce (Telugu), 0-1")
    ap.add_argument("--tight", type=float, default=0.0, help="letters, words and lines closer together, 0-1")
    a = ap.parse_args()
    te = a.script == "te"
    a.lib = a.lib or Path("data/library_te" if te else "data/library")
    a.out = a.out or Path(f"outputs/documents_{a.script}")

    bank = SampleBank(Library(a.lib))
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i in range(a.n):
        rng = np.random.default_rng([a.seed, i])  # each page reproducible on its own
        page, ann = generate_document(bank, rng, a.messiness, a.age, photo=not a.no_scan, paper=a.paper,
                                      degrade=a.degrade, capture=a.capture, size=a.size,
                                      spacing=a.spacing, irregular=a.irregular, tight=a.tight)
        ann["seed"] = [a.seed, i]
        stem = a.out / f"page_{a.seed:03d}_{i:05d}"
        cv2.imwrite(str(stem.with_suffix(".jpg")), page, [cv2.IMWRITE_JPEG_QUALITY, 95])
        stem.with_suffix(".json").write_text(json.dumps(ann, ensure_ascii=False, indent=1))
    print(f"{a.n} pages in {time.time() - t0:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
