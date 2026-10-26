"""Render library samples onto a grid for visual QA.

    python scripts/contact_sheet.py --level word --n 60 --out outputs/qa/words.png
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hwgen.library import Library  # noqa: E402

# Devanagari needs a font with the script and Pillow's raqm layout for conjuncts.
FONT_PATHS = ["/System/Library/Fonts/Kohinoor.ttc", "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"]


def _font(size: int = 13):
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def render(lib: Library, samples, cell_w=300, cell_h=70, cols=3, paper=(235, 245, 250)) -> np.ndarray:
    rows = (len(samples) + cols - 1) // cols
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 255, np.uint8)
    for k, s in enumerate(samples):
        img = lib.image(s)
        a = img[..., 3:4].astype(np.float32) / 255.0
        comp = (img[..., :3] * a + np.array(paper) * (1 - a)).astype(np.uint8)
        scale = min((cell_h - 18) / comp.shape[0], (cell_w - 6) / comp.shape[1], 3.0)
        comp = cv2.resize(comp, (max(1, int(comp.shape[1] * scale)), max(1, int(comp.shape[0] * scale))))
        r, c = divmod(k, cols)
        y, x = r * cell_h, c * cell_w
        sheet[y + 2 : y + 2 + comp.shape[0], x + 3 : x + 3 + comp.shape[1]] = comp
        cv2.rectangle(sheet, (x, y), (x + cell_w - 1, y + cell_h - 1), (220, 220, 220), 1)
    # Labels go on through Pillow: OpenCV's Hershey fonts are ASCII-only.
    pil = Image.fromarray(sheet[..., ::-1])
    draw, font = ImageDraw.Draw(pil), _font()
    for k, s in enumerate(samples):
        r, c = divmod(k, cols)
        draw.text((c * cell_w + 3, r * cell_h + cell_h - 17), f"{s.text or '-'} [{s.label_status}]",
                  font=font, fill=(190, 0, 0))
    return np.asarray(pil)[..., ::-1].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", type=Path, default=Path("data/library"))
    ap.add_argument("--level", default="word")
    ap.add_argument("--labeled", action="store_true", help="only samples with text")
    ap.add_argument("--n", type=int, default=45)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    lib = Library(a.lib)
    pool = lib.query(level=a.level, split=None)
    if a.labeled:
        pool = [s for s in pool if s.text]
    random.Random(a.seed).shuffle(pool)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), render(lib, pool[: a.n]))
    print(f"{min(a.n, len(pool))}/{len(pool)} samples -> {a.out}")


if __name__ == "__main__":
    main()
