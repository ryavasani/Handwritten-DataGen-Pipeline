"""Blind pilot of the task 3 checks with a human/in-session judge instead of an API.

    python scripts/pilot.py make  --out outputs/pilot     # sheets to judge; answers hidden in truth.json
    (judge writes outputs/pilot/answers.json)
    python scripts/pilot.py score --out outputs/pilot

answers.json: {"pairs": {"0": "A", ...},
               "ocr": {"0": "transcription", ...},
               "realism": {"page_000_00000": {"scores": {...}, "overall": n, "artifacts": [...]}}}
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hwgen.library import Library  # noqa: E402
from hwgen.synth.assemble import MissingText  # noqa: E402
from hwgen.synth.bank import SampleBank  # noqa: E402
from hwgen.synth.page import generate_page  # noqa: E402
from hwgen.verify.checks import RUBRIC, line_crops, make_pair, pair_candidates, reference_panel  # noqa: E402
from hwgen.verify.text import aer, cer, wer  # noqa: E402


def _numbered(img: np.ndarray, label: str) -> np.ndarray:
    bar = np.full((26, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 180), 2)
    return np.vstack([bar, img])


def _stack(imgs: list[np.ndarray], width: int | None = None) -> np.ndarray:
    W = width or max(i.shape[1] for i in imgs)
    rows = []
    for i in imgs:
        if i.shape[1] > W:
            k = W / i.shape[1]
            i = cv2.resize(i, (W, int(i.shape[0] * k)), interpolation=cv2.INTER_AREA)
        rows.append(np.pad(i, ((0, 8), (0, W - i.shape[1]), (0, 0)), constant_values=255))
    return np.vstack(rows)


def make(out: Path, n_pairs: int, n_pages: int, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    lib = Library(Path("data/library"))
    ref_bank, synth_bank = SampleBank(lib, split="reference"), SampleBank(lib, split="synth")
    rng = np.random.default_rng(seed)
    truth = {"pairs": {}, "ocr": {}, "pages": []}

    # Real vs fake pairs, 8 per sheet.
    cands = pair_candidates(ref_bank, synth_bank)
    pairs = []
    for text in rng.choice(cands, size=n_pairs * 2, replace=False):
        if len(pairs) == n_pairs:
            break
        try:
            img, fake, meta = make_pair(str(text), ref_bank, synth_bank, rng)
        except MissingText:
            continue
        k = len(pairs)
        truth["pairs"][str(k)] = {"assembled": fake, **{m: meta[m] for m in ("text", "real_writer", "fake_writer")}}
        pairs.append(_numbered(img, f"#{k}"))
    for s in range(0, len(pairs), 8):
        cv2.imwrite(str(out / f"pairs_{s // 8}.png"), _stack(pairs[s : s + 8]))

    # Fresh pages for OCR and realism.
    k = 0
    for p in range(n_pages):
        page, ann = generate_page(synth_bank, np.random.default_rng([seed, 1000 + p]))
        name = f"page_{p:02d}"
        cv2.imwrite(str(out / f"{name}.jpg"), page)
        truth["pages"].append(name)
        crops = []
        for c in line_crops(page, ann):
            truth["ocr"][str(k)] = {"page": name, "field": c["field"], "text": c["text"]}
            crops.append(_numbered(c["image"], f"#{k}"))
            k += 1
        for s in range(0, len(crops), 12):
            cv2.imwrite(str(out / f"ocr_{name}_{s // 12}.png"), _stack(crops[s : s + 12], width=1100))
    cv2.imwrite(str(out / "reference_panel.jpg"), reference_panel(ref_bank, np.random.default_rng(0)))
    (out / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1))
    print(f"{len(pairs)} pairs, {k} OCR lines, {n_pages} pages -> {out} (answers in truth.json; do not open before judging)")


def score(out: Path) -> None:
    truth = json.loads((out / "truth.json").read_text())
    ans = json.loads((out / "answers.json").read_text())
    report = {}

    t, a = truth["pairs"], ans.get("pairs", {})
    judged = [k for k in t if k in a]
    correct = [k for k in judged if a[k] == t[k]["assembled"]]
    report["real_vs_fake"] = {"judged": len(judged), "correct": len(correct),
                              "detection_accuracy": round(len(correct) / max(len(judged), 1), 3),
                              "missed": [t[k]["text"] for k in judged if k not in correct]}

    t, a = truth["ocr"], ans.get("ocr", {})
    rows = [{"id": k, "field": t[k]["field"], "ref": t[k]["text"], "hyp": a[k],
             "cer": round(cer(a[k], t[k]["text"]), 3), "aer": round(aer(a[k], t[k]["text"]), 3),
             "wer": round(wer(a[k], t[k]["text"]), 3)} for k in t if k in a]
    if rows:
        report["ocr"] = {"lines": len(rows),
                         "mean_cer": round(float(np.mean([r["cer"] for r in rows])), 3),
                         "mean_aer": round(float(np.mean([r["aer"] for r in rows])), 3),
                         "mean_wer": round(float(np.mean([r["wer"] for r in rows])), 3),
                         "lines_with_cer_gt_0.15": [r for r in rows if r["cer"] > 0.15]}
    rz = ans.get("realism", {})
    if rz:
        report["realism"] = {"pages": len(rz),
                             "mean_overall": round(float(np.mean([v["overall"] for v in rz.values()])), 2),
                             "mean_by_criterion": {c: round(float(np.mean([v["scores"][c] for v in rz.values()])), 2) for c in RUBRIC}}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "score"])
    ap.add_argument("--out", type=Path, default=Path("outputs/pilot"))
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--pages", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    make(a.out, a.pairs, a.pages, a.seed) if a.cmd == "make" else score(a.out)
