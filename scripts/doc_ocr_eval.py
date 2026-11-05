"""OCR benchmark for the handwritten documents (Google Cloud Vision, which
never saw our training data), across paper types and messiness levels.

Two scores per condition (pooled CER/WER over Devanagari code points):
- line: every written line cropped from its ground-truth word boxes, with the
        rest of the page painted with paper (struck-out words too), so the
        score is handwriting legibility alone;
- page: the whole photographed page in one call, scored against the page's
        text in reading order, so cramped lines running together, stains and
        perspective count as well (the margin note is left out of the reference).

    gcloud auth application-default login     # once
    python scripts/doc_ocr_eval.py --degrade 0.8 --size 0.65 --spacing 0.7 --irregular 0.7 --pages 3 --out outputs/doc_ocr_eval_hi
    python scripts/doc_ocr_eval.py --script te ... --out outputs/doc_ocr_eval_te   # Telugu (data/library_te)
    python scripts/doc_ocr_eval.py ... --dry-run   # count API calls without making them

Defaults: 2 papers x 4 levels x 5 pages = 40 pages, ~110 API calls
(free tier: 1,000/month). Responses are cached, so reruns are free.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hwgen.library import Library  # noqa: E402
from hwgen.synth.bank import SampleBank  # noqa: E402
from hwgen.synth.document import generate_document  # noqa: E402
from hwgen.verify.gocr import GoogleOCR  # noqa: E402
from hwgen.verify.text import char_errors, word_errors  # noqa: E402


def line_crops(page: np.ndarray, ann: dict, pad: int = 6, word_pad: int = 2) -> list[dict]:
    """One crop per written line, from the word boxes in the ground truth.
    Everything outside the line's own word boxes (neighbouring lines) is
    painted with the paper colour, so a reader sees only the text being scored."""
    out = []
    H, W = page.shape[:2]
    for f in ann["fields"]:
        for k, line in enumerate(f["lines"]):
            boxes = np.array([w["box"] for w in line["words"]])
            if len(boxes) == 0:
                continue
            x0, y0 = max(0, boxes[:, 0].min() - pad), max(0, boxes[:, 1].min() - pad)
            x1 = min(W, (boxes[:, 0] + boxes[:, 2]).max() + pad)
            y1 = min(H, (boxes[:, 1] + boxes[:, 3]).max() + pad)
            crop = page[y0:y1, x0:x1].copy()
            keep = np.zeros(crop.shape[:2], bool)
            for bx, by, bw, bh in boxes:
                keep[max(0, by - y0 - word_pad) : by - y0 + bh + word_pad,
                     max(0, bx - x0 - word_pad) : bx - x0 + bw + word_pad] = True
            crop[~keep] = np.median(crop.reshape(-1, crop.shape[2]), axis=0).astype(np.uint8)
            out.append({"field": f["name"], "line": k, "text": line["text"], "image": crop})
    return out


def pooled(rows: list[dict]) -> dict:
    if not rows:
        return {"lines": 0}
    ce = [char_errors(r["hyp"], r["ref"]) for r in rows]
    we = [word_errors(r["hyp"], r["ref"]) for r in rows]
    return {"lines": len(rows),
            "cer": round(sum(e for e, _ in ce) / max(sum(n for _, n in ce), 1), 4),
            "wer": round(sum(e for e, _ in we) / max(sum(n for _, n in we), 1), 4),
            "exact_lines": round(sum(e == 0 for e, _ in ce) / len(rows), 3)}


def readable_lines(ann: dict) -> dict:
    """The annotation without struck words and without the rotated margin note."""
    ann = copy.deepcopy(ann)
    ann["fields"] = [f for f in ann["fields"] if not f.get("rotation")]
    for f in ann["fields"]:
        for line in f["lines"]:
            line["words"] = [w for w in line["words"] if not w.get("struck")]
        f["lines"] = [line for line in f["lines"] if line["words"]]
    return ann


def page_text(ann: dict) -> str:
    return " ".join(line["text"] for f in ann["fields"] if not f.get("rotation") for line in f["lines"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", nargs="+", default=["aged", "notebook"])
    ap.add_argument("--levels", nargs="+", type=float, default=[0.2, 0.5, 0.8, 1.0], help="messiness levels")
    ap.add_argument("--pages", type=int, default=5, help="pages per paper x level")
    ap.add_argument("--degrade", type=float, default=0.0, help="damage/capture/sharing layers, 0-1")
    ap.add_argument("--size", type=float, default=1.0, help="letter height multiplier (generate.py --size)")
    ap.add_argument("--spacing", type=float, default=0.0, help="uneven word/letter spacing, 0-1")
    ap.add_argument("--irregular", type=float, default=0.0, help="per-word slant; per-letter jitter (Telugu), 0-1")
    ap.add_argument("--script", choices=["hi", "te"], default="hi")
    ap.add_argument("--lib", type=Path, help="data/library (hi) or data/library_te (te) if unset")
    ap.add_argument("--out", type=Path, default=Path("outputs/doc_ocr_eval"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    a.lib = a.lib or Path("data/library_te" if a.script == "te" else "data/library")
    a.out.mkdir(parents=True, exist_ok=True)
    ocr = GoogleOCR(language="te" if a.script == "te" else "hi", dry_run=a.dry_run)
    bank = SampleBank(Library(a.lib), split="synth")

    line_rows, page_rows, crops = [], [], []
    for pi, paper in enumerate(a.papers):
        for li, level in enumerate(a.levels):
            for i in range(a.pages):
                rng = np.random.default_rng([a.seed, pi, li, i])
                page, ann = generate_document(bank, rng, messiness=level, paper=paper, degrade=a.degrade,
                                              size=a.size, spacing=a.spacing, irregular=a.irregular)
                cond = {"paper": paper, "messiness": level, "page": i, "layers": ann["layers"]}
                if i == 0:
                    cv2.imwrite(str(a.out / f"{paper}_m{level}.jpg"), page, [cv2.IMWRITE_JPEG_QUALITY, 90])
                for c in line_crops(page, readable_lines(ann)):
                    line_rows.append({**cond, "field": c["field"], "ref": c["text"]})
                    crops.append(c["image"])
                page_rows.append({**cond, "ref": page_text(ann), "hyp": ocr.read_page(page).replace("\n", " ")})
    for r, hyp in zip(line_rows, ocr.read(crops)):
        r["hyp"] = hyp

    if a.dry_run:
        print(f"dry run: {ocr.calls} API calls needed ({ocr.cached} cached); "
              f"{len(page_rows)} pages, {len(line_rows)} lines")
        return

    def page_scores(rows):
        ce = [char_errors(r["hyp"], r["ref"]) for r in rows]
        we = [word_errors(r["hyp"], r["ref"]) for r in rows]
        return {"pages": len(rows), "cer": round(sum(e for e, _ in ce) / max(sum(n for _, n in ce), 1), 4),
                "wer": round(sum(e for e, _ in we) / max(sum(n for _, n in we), 1), 4)}

    summary = {"api_calls": ocr.calls, "cache_hits": ocr.cached, "degrade": a.degrade,
               "size": a.size, "spacing": a.spacing, "irregular": a.irregular, "conditions": []}
    for paper in a.papers:
        for level in a.levels:
            sel = lambda rows: [r for r in rows if r["paper"] == paper and r["messiness"] == level]  # noqa: E731
            summary["conditions"].append({"paper": paper, "messiness": level,
                                          "line": pooled(sel(line_rows)), "page": page_scores(sel(page_rows))})
    summary["all"] = {"line": pooled(line_rows), "page": page_scores(page_rows)}
    summary["note"] = "pooled CER/WER after normalize_ocr (digit script ignored); struck words excluded from refs"
    with open(a.out / "results.jsonl", "w", encoding="utf-8") as fh:
        for r in line_rows:
            fh.write(json.dumps({"kind": "line", **r}, ensure_ascii=False) + "\n")
        for r in page_rows:
            fh.write(json.dumps({"kind": "page", **r}, ensure_ascii=False) + "\n")
    (a.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))

    print(f"{'paper':9} {'mess':>5} {'lines':>6} {'line CER':>9} {'line WER':>9} {'page CER':>9} {'page WER':>9}")
    for c in summary["conditions"] + [{"paper": "ALL", "messiness": "", **summary["all"]}]:
        print(f"{c['paper']:9} {c['messiness']!s:>5} {c['line']['lines']:>6} {c['line']['cer']:>9.1%} "
              f"{c['line']['wer']:>9.1%} {c['page']['cer']:>9.1%} {c['page']['wer']:>9.1%}")


if __name__ == "__main__":
    main()
