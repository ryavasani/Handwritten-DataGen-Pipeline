"""Real-vs-fake test: can a VLM tell a real held-out handwritten word from a
synthetic assembly of the same word, rendered identically? Accuracy near 50%
means the assembly is indistinguishable to the judge.

    python scripts/real_vs_fake.py --n 200 --out outputs/real_vs_fake [--dry-run]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hwgen.library import Library  # noqa: E402
from hwgen.synth.assemble import MissingText  # noqa: E402
from hwgen.synth.bank import SampleBank  # noqa: E402
from hwgen.verify.checks import check_pair, make_pair, pair_candidates  # noqa: E402
from hwgen.verify.vlm import VLMClient, VLMRefusal  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", type=Path, default=Path("data/library"))
    ap.add_argument("--out", type=Path, default=Path("outputs/real_vs_fake"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    lib = Library(a.lib)
    ref_bank, synth_bank = SampleBank(lib, split="reference"), SampleBank(lib, split="synth")
    cands = pair_candidates(ref_bank, synth_bank)
    print(f"{len(cands)} candidate words with a real held-out sample and full akshara coverage")
    rng = np.random.default_rng(a.seed)
    vlm = VLMClient(effort=a.effort, dry_run=a.dry_run, dry_run_dir=a.out / "dry_run")
    (a.out / "pairs").mkdir(parents=True, exist_ok=True)

    rows, correct, judged = [], 0, 0
    for i, text in enumerate(rng.choice(cands, size=min(a.n, len(cands)), replace=False)):
        try:
            pair, truth, meta = make_pair(str(text), ref_bank, synth_bank, rng)
        except MissingText:
            continue
        cv2.imwrite(str(a.out / "pairs" / f"pair_{i:04d}.jpg"), pair)
        row = {"id": i, "truth": truth, **meta}
        try:
            res = check_pair(vlm, pair, str(text))
        except VLMRefusal as e:
            row["error"] = f"refusal: {e}"
            res = {}
        if res:
            row.update(res)
            row["correct"] = res["assembled"] == truth
            correct += row["correct"]
            judged += 1
        rows.append(row)
    with open(a.out / "results.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    u = vlm.usage
    summary = {"pairs": len(rows), "judged": judged, "api_calls": u.calls, "cache_hits": u.cached,
               "cost_usd": round(u.cost(), 4)}
    if judged:
        lo, hi = wilson(correct, judged)
        summary.update({"detection_accuracy": correct / judged, "ci95": [round(lo, 3), round(hi, 3)],
                        "note": "0.5 = chance (indistinguishable); 1.0 = always detected"})
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
