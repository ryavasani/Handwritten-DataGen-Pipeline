"""Build the Hindi handwriting sample library from IIIT-HW-Hindi.

    python scripts/build_library.py [--max-height 160] [--limit N]
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hwgen.datasets.iiit_hw import load_words  # noqa: E402
from hwgen.extract.features import measure_devanagari  # noqa: E402
from hwgen.extract.word import clean_word  # noqa: E402
from hwgen.library import Sample, write_sample  # noqa: E402


def _process(job):
    words, root, out, max_height = job
    cv2.setNumThreads(1)  # parallelism comes from the process pool
    records, failed = [], 0
    for w in words:
        img = cv2.imread(str(root / w.path))
        res = clean_word(img, max_height) if img is not None else None
        if res is None:
            failed += 1
            continue
        rgba, mask = res
        s = Sample(w.id, "word", w.text, "gold", w.path, w.writer, w.library_split,
                   features={**measure_devanagari(rgba, mask), "dataset_split": w.dataset_split})
        records.append(write_sample(out, s, rgba))
    return records, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/raw/iiit_hw_hindi"))
    ap.add_argument("--out", type=Path, default=Path("data/library"))
    ap.add_argument("--max-height", type=int, default=160)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="process only N words (debugging)")
    a = ap.parse_args()

    words = load_words(a.data)
    if a.limit:
        words = words[:: max(1, len(words) // a.limit)][: a.limit]  # spread over writers
    if a.out.exists():
        shutil.rmtree(a.out)
    a.out.mkdir(parents=True)

    chunk = 500
    jobs = [(words[i : i + chunk], a.data, a.out, a.max_height) for i in range(0, len(words), chunk)]
    records, failed = [], 0
    with ProcessPoolExecutor(a.workers) as ex:
        for k, (recs, f) in enumerate(ex.map(_process, jobs), 1):
            records += recs
            failed += f
            if k % 20 == 0:
                print(f"{min(k * chunk, len(words))}/{len(words)} words", flush=True)

    with open(a.out / "index.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(records)} samples written, {failed} images unusable")
    for k, v in sorted(Counter((r["split"], r["writer"]) for r in records).items()):
        print(f"  {k[0]:9s} writer {k[1]:>3s}: {v}")


if __name__ == "__main__":
    main()
