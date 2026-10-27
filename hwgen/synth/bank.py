"""Fast lookup of library samples by text and writer."""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from statistics import median

import numpy as np

from ..library import Library, Sample


class SampleBank:
    def __init__(self, lib: Library, split: str = "synth", cache_size: int = 4000):
        self.lib = lib
        self.words: dict[str, dict[str, list[Sample]]] = defaultdict(lambda: defaultdict(list))
        self.aksharas: dict[str, dict[str, list[Sample]]] = defaultdict(lambda: defaultdict(list))
        style: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for s in lib.query(split=split):
            if s.level == "word":
                self.words[_key(s.text)][s.writer].append(s)
                style[s.writer].append((s.features["slant_deg"], s.features["stroke_over_body"]))
            elif s.level == "akshara":
                self.aksharas[s.text][s.writer].append(s)
        self._drop_outlier_words()
        self._drop_outlier_aksharas()
        self.writers = sorted(style, key=int)
        # Writer similarity: median slant and stroke weight, scaled to comparable units.
        prof = {w: (median(v[0] for v in vs) / 10.0, median(v[1] for v in vs) / 0.02) for w, vs in style.items()}
        self.stroke_over_body = {w: median(v[1] for v in vs) for w, vs in style.items()}
        self.nearest = {
            w: sorted(self.writers, key=lambda o: (prof[o][0] - prof[w][0]) ** 2 + (prof[o][1] - prof[w][1]) ** 2)
            for w in self.writers
        }
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size

    def _drop_outlier_words(self, tol: float = 0.25) -> None:
        """Every source image has the same height, so a writer's body heights
        should agree; a word far from its writer's median had its headline or
        baseline mis-measured and would be scaled wrongly. Same for stroke weight."""
        per_writer = defaultdict(list)
        for by_writer in self.words.values():
            for w, ss in by_writer.items():
                per_writer[w] += ss
        med = {w: (np.median([s.features["body_height"] for s in ss]),
                   np.median([s.features["stroke_over_body"] for s in ss])) for w, ss in per_writer.items()}
        for by_writer in self.words.values():
            for w in list(by_writer):
                mb, ms = med[w]
                by_writer[w] = [s for s in by_writer[w]
                                if abs(s.features["body_height"] / mb - 1) <= tol
                                and s.features["stroke_over_body"] <= 1.6 * ms]

    def _drop_outlier_aksharas(self, tol: float = 0.35) -> None:
        """An akshara is scaled by its parent word's measured body height, so a
        bad measurement (or a bad cut) shows up as a width or height far from
        that akshara's typical size in body units. Keep only typical samples."""
        for text, by_writer in self.aksharas.items():
            all_s = [s for ss in by_writer.values() for s in ss]
            if len(all_s) < 5:
                continue
            rel = np.array([(s.features["width"] / s.features["body_height"],
                             s.features["height"] / s.features["body_height"]) for s in all_s])
            med = np.median(rel, axis=0)
            ok = {s.id for s, r in zip(all_s, rel) if np.all(np.abs(r / med - 1) <= tol)}
            for w in list(by_writer):
                by_writer[w] = [s for s in by_writer[w] if s.id in ok]

    def _pick(self, table, key: str, writer: str, rng: np.random.Generator) -> Sample | None:
        by_writer = table.get(key)
        if not by_writer:
            return None
        for w in self.nearest[writer]:  # own writer first, then the most similar ones
            if by_writer.get(w):
                cands = by_writer[w]
                return cands[rng.integers(len(cands))]
        return None

    def word(self, text: str, writer: str, rng: np.random.Generator) -> Sample | None:
        return self._pick(self.words, _key(text), writer, rng)

    def akshara(self, text: str, writer: str, rng: np.random.Generator) -> Sample | None:
        return self._pick(self.aksharas, text, writer, rng)

    def has_akshara(self, text: str) -> bool:
        return any(self.aksharas.get(text, {}).values())

    def alpha(self, s: Sample) -> np.ndarray:
        """Float32 ink matte in [0, 1], cached."""
        a = self._cache.get(s.id)
        if a is None:
            a = self.lib.image(s)[..., 3].astype(np.float32) / 255.0
            self._cache[s.id] = a
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(s.id)
        return a


def _key(text: str) -> str:
    # Exact match: a sample of "बाई।" contains the danda's ink and must not stand in for "बाई".
    return text.strip()
