"""Per-document handwriting style: everything the generator randomises.

Ranges come from the real data where we measured them (slant, stroke width, ink colour)
and from typical ballpoint/scan behaviour where we could not (word spacing:
the source dataset has isolated words only).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# Ballpoint ink colours (BGR), roughly matching the measured blue/black split.
INKS = {
    "blue": [(150, 60, 40), (170, 80, 50), (130, 50, 45), (160, 70, 70)],
    "black": [(40, 35, 35), (55, 50, 50), (30, 30, 40)],
}


@dataclass
class Style:
    writer: str
    body_px: float            # headline-to-baseline height on the output page
    ink_bgr: tuple[int, int, int]
    word_gap: float           # in body heights
    akshara_gap: float        # extra spacing between assembled aksharas, in body heights
    size_jitter: float        # per-word relative std of size
    rotation_jitter: float    # per-word std (deg)
    extra_slant: float        # shear added to every word (deg, + = right)
    baseline_drift: float     # amplitude of the slow baseline wave, in body heights
    drift_period: float       # wavelength of that wave, in body heights
    line_angle: float         # whole-line tilt (deg)
    pressure: float           # depth of slow pen-pressure variation, 0..1
    fade: float               # probability-weighted strength of faded patches, 0..1
    blot_rate: float          # ink blots per word

    def to_dict(self) -> dict:
        return asdict(self)


def sample_style(rng: np.random.Generator, writers: list[str], body_px: float | None = None) -> Style:
    family = "blue" if rng.random() < 0.85 else "black"
    inks = INKS[family]
    return Style(
        writer=str(rng.choice(writers)),
        body_px=float(body_px if body_px is not None else rng.uniform(13, 19)),
        ink_bgr=tuple(int(c) for c in inks[rng.integers(len(inks))]),
        word_gap=float(rng.uniform(0.55, 1.1)),
        akshara_gap=float(rng.uniform(-0.06, 0.04)),
        size_jitter=float(rng.uniform(0.02, 0.07)),
        rotation_jitter=float(rng.uniform(0.3, 1.5)),
        extra_slant=float(rng.normal(0, 4)),
        baseline_drift=float(rng.uniform(0.0, 0.25)),
        drift_period=float(rng.uniform(15, 40)),
        line_angle=float(rng.normal(0, 0.6)),
        pressure=float(rng.uniform(0.05, 0.3)),
        fade=float(rng.uniform(0.0, 0.35)),
        blot_rate=float(rng.uniform(0.0, 0.04)),
    )
