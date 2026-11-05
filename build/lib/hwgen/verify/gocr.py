"""Google Cloud Vision handwriting OCR, independent of our training data.

Crops are packed onto shared sheets (one API call per sheet instead of per
line) and Vision's word boxes are mapped back to the crop they fall in.
Responses are cached on disk, so reruns cost nothing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

GAP = 48  # white space between packed crops, so Vision never merges two lines


class GoogleOCR:
    def __init__(self, cache_dir: Path = Path("data/cache/gocr"), language: str = "hi", dry_run: bool = False):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.language = language
        self.dry_run = dry_run
        self.calls = 0
        self.cached = 0
        self._client = None

    def _vision_words(self, sheet: np.ndarray) -> list[dict]:
        """All recognised words on a sheet as {text, cx, cy}."""
        ok, buf = cv2.imencode(".png", sheet)
        data = buf.tobytes()
        cache = self.cache_dir / f"{hashlib.sha256(data + self.language.encode()).hexdigest()}.json"
        if cache.exists():
            self.cached += 1
            return json.loads(cache.read_text())
        self.calls += 1
        if self.dry_run:
            return []
        if self._client is None:
            from google.cloud import vision  # uses Application Default Credentials (gcloud auth)

            self._vision = vision
            self._client = vision.ImageAnnotatorClient()
        resp = self._client.document_text_detection(
            image=self._vision.Image(content=data), image_context={"language_hints": [self.language]})
        if resp.error.message:
            raise RuntimeError(f"Vision API error: {resp.error.message}")
        words = []
        for page in resp.full_text_annotation.pages:
            for block in page.blocks:
                for para in block.paragraphs:
                    for w in para.words:
                        vs = w.bounding_box.vertices
                        words.append({"text": "".join(s.text for s in w.symbols),
                                      "cx": sum(v.x for v in vs) / len(vs), "cy": sum(v.y for v in vs) / len(vs)})
        cache.write_text(json.dumps(words, ensure_ascii=False))
        return words

    def read_page(self, page: np.ndarray) -> str:
        """Full-page transcript in Vision's own reading order (blocks, then lines)."""
        ok, buf = cv2.imencode(".jpg", page, [cv2.IMWRITE_JPEG_QUALITY, 95])
        data = buf.tobytes()
        cache = self.cache_dir / f"page_{hashlib.sha256(data + self.language.encode()).hexdigest()}.json"
        if cache.exists():
            self.cached += 1
            return json.loads(cache.read_text())["text"]
        self.calls += 1
        if self.dry_run:
            return ""
        if self._client is None:
            from google.cloud import vision

            self._vision = vision
            self._client = vision.ImageAnnotatorClient()
        resp = self._client.document_text_detection(
            image=self._vision.Image(content=data), image_context={"language_hints": [self.language]})
        if resp.error.message:
            raise RuntimeError(f"Vision API error: {resp.error.message}")
        text = resp.full_text_annotation.text
        cache.write_text(json.dumps({"text": text}, ensure_ascii=False))
        return text

    def page_words(self, page: np.ndarray) -> list[dict]:
        """Every recognised word on a page as {text, box: [x, y, w, h], line},
        in Vision's reading order; `line` numbers the text lines Vision found
        (a new line starts after a word whose last symbol ends a line)."""
        ok, buf = cv2.imencode(".jpg", page, [cv2.IMWRITE_JPEG_QUALITY, 92])
        data = buf.tobytes()
        cache = self.cache_dir / f"words_{hashlib.sha256(data + self.language.encode()).hexdigest()}.json"
        if cache.exists():
            self.cached += 1
            return json.loads(cache.read_text())
        self.calls += 1
        if self.dry_run:
            return []
        if self._client is None:
            from google.cloud import vision

            self._vision = vision
            self._client = vision.ImageAnnotatorClient()
        resp = self._client.document_text_detection(
            image=self._vision.Image(content=data), image_context={"language_hints": [self.language]})
        if resp.error.message:
            raise RuntimeError(f"Vision API error: {resp.error.message}")
        brk = self._vision.TextAnnotation.DetectedBreak.BreakType
        words, line = [], 0
        for p in resp.full_text_annotation.pages:
            for block in p.blocks:
                for para in block.paragraphs:
                    for w in para.words:
                        xs = [v.x for v in w.bounding_box.vertices]
                        ys = [v.y for v in w.bounding_box.vertices]
                        words.append({"text": "".join(s.text for s in w.symbols), "line": line,
                                      "box": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]})
                        last = w.symbols[-1].property.detected_break.type_ if w.symbols else None
                        if last in (brk.EOL_SURE_SPACE, brk.LINE_BREAK, brk.HYPHEN):
                            line += 1
                    line += 1
        cache.write_text(json.dumps(words, ensure_ascii=False))
        return words

    def read(self, crops: list[np.ndarray], per_sheet: int = 12) -> list[str]:
        """Transcribe each crop (BGR). Returns one string per crop, in order."""
        out: list[str] = []
        for s in range(0, len(crops), per_sheet):
            group = crops[s : s + per_sheet]
            W = max(c.shape[1] for c in group) + 2 * GAP
            H = sum(c.shape[0] for c in group) + GAP * (len(group) + 1)
            sheet = np.full((H, W, 3), 255, np.uint8)
            spans, y = [], GAP
            for c in group:
                sheet[y : y + c.shape[0], GAP : GAP + c.shape[1]] = c
                spans.append((y, y + c.shape[0]))
                y += c.shape[0] + GAP
            per_crop: list[list[dict]] = [[] for _ in group]
            for w in self._vision_words(sheet):
                for i, (y0, y1) in enumerate(spans):
                    if y0 - GAP / 2 <= w["cy"] < y1 + GAP / 2:
                        per_crop[i].append(w)
                        break
            out += [" ".join(w["text"] for w in sorted(ws, key=lambda w: w["cx"])) for ws in per_crop]
        return out
