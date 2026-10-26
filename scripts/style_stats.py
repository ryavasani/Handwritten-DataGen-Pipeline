"""Measure the real Hindi handwriting style distribution in the library.

Writes <lib>/style_stats.json (overall and per-writer percentiles that the
generator samples from in task 2) and a histogram figure. Only the `synth`
split is used, so held-out reference writers stay unseen by the generator.

    python scripts/style_stats.py --fig outputs/style_stats.png
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hwgen.library import Library  # noqa: E402

PCTS = [5, 25, 50, 75, 95]
METRICS = {
    "slant_deg": "Slant (deg, + = right)",
    "headline_deg": "Headline angle (deg)",
    "stroke_over_body": "Stroke width / body height",
    "upper_extent": "Upper matra extent / body height",
    "lower_extent": "Lower matra extent / body height",
    "ink_darkness": "Ink darkness (mean alpha)",
}


def summarize(values) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return {"n": 0}
    return {"n": int(len(v)), "mean": round(float(v.mean()), 3), "std": round(float(v.std()), 3),
            **{f"p{p}": round(float(np.percentile(v, p)), 3) for p in PCTS}}


def plot(words, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series, text2, grid = "#2a78d6", "#52514e", "#e4e3df"
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, METRICS.items()):
        v = np.asarray([s.features[key] for s in words if key in s.features], dtype=float)
        if len(v):
            lo, hi = np.percentile(v, [1, 99])
            v = v[(v >= lo) & (v <= hi)]  # drop the extreme 2% rather than pile them at the edges
            ax.hist(v, bins=40, color=series, edgecolor="white", linewidth=0.6)
            ax.axvline(np.median(v), color=text2, linewidth=1, linestyle="--")
        ax.set_title(f"{title}   n={len(v)}", fontsize=10, color="#0b0b0b", loc="left")
        ax.tick_params(colors=text2, labelsize=8)
        ax.grid(axis="y", color=grid, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
    fig.suptitle("Real Hindi handwriting style, IIIT-HW-Hindi synth writers (dashed = median, 1-99th pct shown)",
                 fontsize=11, x=0.01, ha="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", type=Path, default=Path("data/library"))
    ap.add_argument("--fig", type=Path, default=None)
    a = ap.parse_args()
    words = Library(a.lib).query(level="word")

    out = {"overall": {k: summarize(s.features.get(k) for s in words) for k in METRICS}}
    out["overall"]["ink_family_share"] = {k: round(c / len(words), 3) for k, c in
                                          Counter(s.features.get("ink_family") for s in words).items()}
    by_writer = defaultdict(list)
    for s in words:
        by_writer[s.writer].append(s)
    out["per_writer"] = {w: {k: summarize(s.features.get(k) for s in ws) for k in ("slant_deg", "stroke_over_body", "headline_deg")}
                         for w, ws in sorted(by_writer.items(), key=lambda kv: int(kv[0]))}
    (a.lib / "style_stats.json").write_text(json.dumps(out, indent=2))

    for k, v in out["overall"].items():
        print(f"{k:18s} {v}")
    print("per-writer median slant:", {w: d["slant_deg"]["p50"] for w, d in out["per_writer"].items()})
    if a.fig:
        plot(words, a.fig)
        print(f"figure -> {a.fig}")


if __name__ == "__main__":
    main()
