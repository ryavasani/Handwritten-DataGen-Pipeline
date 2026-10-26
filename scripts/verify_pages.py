"""Verify generated pages: VLM transcription (OCR accuracy) and a realism
rating against real held-out handwriting. Writes one result per page and a
summary; pages failing the thresholds are listed for regeneration (task 4).

    export ANTHROPIC_API_KEY=...        # or `ant auth login`
    python scripts/verify_pages.py --pages outputs/synth --out outputs/verify
    python scripts/verify_pages.py --pages outputs/synth --dry-run    # no API calls
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hwgen.library import Library  # noqa: E402
from hwgen.synth.bank import SampleBank  # noqa: E402
from hwgen.verify.checks import check_ocr, check_realism, reference_panel  # noqa: E402
from hwgen.verify.vlm import VLMClient, VLMRefusal  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=Path, default=Path("outputs/synth"))
    ap.add_argument("--out", type=Path, default=Path("outputs/verify"))
    ap.add_argument("--lib", type=Path, default=Path("data/library"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-cer", type=float, default=0.15, help="fail a page whose mean line CER exceeds this")
    ap.add_argument("--min-realism", type=int, default=3, help="fail a page whose overall realism is below this")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--dry-run", action="store_true", help="build requests and save what would be sent; no API calls")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    vlm = VLMClient(effort=a.effort, dry_run=a.dry_run, dry_run_dir=a.out / "dry_run")
    ref_bank = SampleBank(Library(a.lib), split="reference")
    refs = reference_panel(ref_bank, np.random.default_rng(0))  # fixed panel: same calibration for every page
    cv2.imwrite(str(a.out / "reference_panel.jpg"), refs)

    pages = sorted(a.pages.glob("page_*.jpg"))[: a.limit or None]
    results = []
    for p in pages:
        page, ann = cv2.imread(str(p)), json.loads(p.with_suffix(".json").read_text())
        r = {"page": p.name}
        try:
            r["ocr"] = check_ocr(vlm, page, ann)
            r["realism"] = check_realism(vlm, page, refs)
        except VLMRefusal as e:
            r["error"] = f"refusal: {e}"
        if not a.dry_run and "error" not in r:
            cers = [x["cer"] for x in r["ocr"] if x["cer"] is not None]
            r["mean_cer"] = float(np.mean(cers)) if cers else None
            r["mean_aer"] = float(np.mean([x["aer"] for x in r["ocr"] if x["aer"] is not None])) if cers else None
            r["pass"] = (r["mean_cer"] is not None and r["mean_cer"] <= a.max_cer
                         and r["realism"]["overall"] >= a.min_realism)
            print(f"{p.name}: CER {r['mean_cer']:.3f}  realism {r['realism']['overall']}/5  "
                  f"{'PASS' if r['pass'] else 'FAIL'}", flush=True)
        (a.out / f"{p.stem}.verify.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
        results.append(r)

    u = vlm.usage
    summary = {"pages": len(results), "api_calls": u.calls, "cache_hits": u.cached, "refusals": u.refusals,
               "input_tokens": u.input_tokens, "output_tokens": u.output_tokens, "cost_usd": round(u.cost(), 4)}
    done = [r for r in results if "pass" in r]
    if done:
        summary["pass_rate"] = sum(r["pass"] for r in done) / len(done)
        summary["mean_cer"] = float(np.mean([r["mean_cer"] for r in done if r["mean_cer"] is not None]))
        summary["mean_realism"] = float(np.mean([r["realism"]["overall"] for r in done]))
        summary["failed_pages"] = [r["page"] for r in done if not r["pass"]]
    (a.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
