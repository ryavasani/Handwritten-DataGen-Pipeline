"""Loader for IIIT-HW-Hindi v1 / IIIT-HW-Dev (Dutta et al., DAS 2018).

`train.txt` / `val.txt` / `test.txt` hold "<relative image path> <word>" lines;
paths look like HindiSeg/<split>/<writer>/<page>/<n>.jpg. The dataset's splits
are writer-disjoint, so its test writers make a natural held-out reference set.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

DATASET_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class WordImage:
    path: str          # relative to the dataset root
    text: str
    dataset_split: str
    writer: str
    page: str
    index: str

    @property
    def id(self) -> str:
        return f"{self.dataset_split}_{self.writer}_{self.page}_{self.index}"

    @property
    def library_split(self) -> str:
        return "reference" if self.dataset_split == "test" else "synth"


def load_words(root: Path) -> list[WordImage]:
    out = []
    for split in DATASET_SPLITS:
        for line in (Path(root) / f"{split}.txt").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            path, text = line.split(" ", 1)
            _, _, writer, page, name = path.split("/")
            out.append(WordImage(path, unicodedata.normalize("NFC", text.strip()), split, writer, page, Path(name).stem))
    return out
