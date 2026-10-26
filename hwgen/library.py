"""On-disk handwriting sample library.

Layout:
    <root>/index.jsonl            one JSON record per sample
    <root>/<level>/<id>.png       8-bit soft ink matte (alpha); the ink colour is
                                  features["ink_bgr"]. `Library.image` returns RGBA.

Levels: "word" (a whole handwritten word) and "akshara" (an orthographic
syllable cut from a word). Composite onto a page with a multiply blend:
page * (rgb / 255), restricted by alpha.

`label_status` says how far the text can be trusted:
    gold        human transcription from the source dataset
    derived     akshara cut from a gold word by segmentation alignment; the cut is unverified
    rejected    a reviewer (VLM / human) says the text does not match
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from dataclasses import field as dc_field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Sample:
    id: str
    level: str
    text: str | None
    label_status: str
    source: str               # dataset-relative path of the source image
    writer: str
    split: str                # "synth" (usable by the generator) or "reference" (held out)
    script: str = "deva"
    parent_id: str | None = None
    features: dict = dc_field(default_factory=dict)
    verification: dict = dc_field(default_factory=dict)

    @property
    def rel_path(self) -> str:
        return f"{self.level}/{self.id}.png"


class Library:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.samples: list[Sample] = []
        idx = self.root / "index.jsonl"
        if idx.exists():
            for line in idx.read_text(encoding="utf-8").splitlines():
                d = json.loads(line)
                self.samples.append(Sample(**d))

    def image(self, s: Sample) -> np.ndarray:
        """RGBA crop: flat ink colour, stroke texture carried by the alpha."""
        alpha = cv2.imread(str(self.root / s.rel_path), cv2.IMREAD_GRAYSCALE)
        bgr = np.empty((*alpha.shape, 3), np.uint8)
        bgr[:] = s.features.get("ink_bgr", (0, 0, 0))
        return np.dstack([bgr, alpha])

    def save_index(self) -> None:
        with open(self.root / "index.jsonl", "w", encoding="utf-8") as fh:
            for s in self.samples:
                fh.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")

    def query(self, level: str | None = None, text: str | None = None, split: str | None = "synth",
              statuses: tuple[str, ...] | None = None) -> list[Sample]:
        return [
            s for s in self.samples
            if (level is None or s.level == level)
            and (text is None or s.text == text)
            and (split is None or s.split == split)
            and (statuses is None or s.label_status in statuses)
        ]


def write_sample(root: Path, s: Sample, rgba: np.ndarray) -> dict:
    """Store only the alpha matte: the RGB of a scanned stroke is scanner noise
    around one ink colour (kept in features), and storing it costs ~4x the disk."""
    path = Path(root) / s.rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), rgba[..., 3], [cv2.IMWRITE_PNG_COMPRESSION, 9])
    # Round-trip so numpy scalars from OpenCV become plain JSON numbers.
    return json.loads(json.dumps(asdict(s), default=lambda o: o.item()))
