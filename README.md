# CV-Research — Realistic Hindi handwriting for Indian legal documents

Standalone pipeline that writes realistic **Hindi (Devanagari)** handwriting onto
clean legal document pages using **real handwriting samples** (OpenCV + NumPy),
not fonts.

| Task | Status |
|---|---|
| 1. Collect real handwriting samples → sample library | done |
| 2. Generation + overlay | done (v1) |
| 3. VLM / OCR verification | built; needs an API key to run |
| 4. Feedback loop | – |
| 5. Evaluation & optimization | – |
| 6. Paper material | – |

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p data/raw/iiit_hw_hindi && cd data/raw/iiit_hw_hindi
curl -fLO -C - https://cvit.iiit.ac.in/images/Projects/wordlevel-Indicscripts/IIIT-HW-Hindi_v1.tar.gz
gzip -t IIIT-HW-Hindi_v1.tar.gz   # the server can drop the connection; rerun curl until this passes
tar -xzf IIIT-HW-Hindi_v1.tar.gz && tar -xzf HindiSeg.tar.gz && rm HindiSeg.tar.gz
```

## Task 1: build the sample library

```bash
.venv/bin/python scripts/build_library.py    # 95K word samples (~25 min)
.venv/bin/python scripts/build_aksharas.py   # cut words into aksharas (~2 min)
.venv/bin/python scripts/style_stats.py --fig outputs/style_stats.png
.venv/bin/python scripts/contact_sheet.py --level akshara --out outputs/qa/aksharas.png
```

### Source data

**IIIT-HW-Hindi v1** (a.k.a. IIIT-HW-Dev; Dutta et al., DAS 2018 — cite the paper),
<https://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data>:
95,430 human-labelled handwritten Devanagari word images, 9,539 distinct words,
**12 writers**, mostly blue ballpoint, ~293 px tall. The dataset's splits are
writer-disjoint; its 3 test writers become our held-out `reference` split.

No public scans of handwritten Hindi legal documents exist, so the
"legal-document" look (forms, degradation, layout) has to come from elsewhere
and real Hindi legal pages for evaluation remain an open gap.

### Pipeline

1. **Word cleaning** (`hwgen/extract/word.py`) — background normalisation and a
   soft ink matte (`ink.py`); removal of scan-edge lines, paper ruling (pixel-level,
   so letters touching the line survive) and specks far from the word; tight crop;
   downscale to ≤160 px height.
2. **Devanagari features** (`features.py: measure_devanagari`) — headline
   (shirorekha) band and angle, baseline, body height (the Devanagari analogue of
   x-height), upper/lower matra extent, slant, stroke width, ink colour.
3. **Akshara segmentation** (`hwgen/text/devanagari.py`, `hwgen/extract/akshara.py`)
   — text is split into aksharas (conjuncts, pre-base ि and reph handled); the image
   is split into body units below the headline, typed as letter bodies or narrow
   bars; a DP assigns units to aksharas using per-akshara unit patterns (e.g. बा =
   body+bar, or body alone if the bar is joined) and width priors. Only words whose
   best alignment cost is ≤ 0.3 are cut (31% of words).

### Library (`data/library/`, see `hwgen/library.py`)

| Level | Samples | Labels |
|---|---|---|
| word | 95,386 | `gold` (human) |
| akshara | 91,875 (1,578 distinct; 962 with ≥5 samples) | `derived` (~88% correct on visual spot check) |

- `index.jsonl` — `id, level, text, label_status, source, writer, split, script, parent_id, features`.
- `<level>/<id>.png` — 8-bit soft ink matte only; the ink colour is `features.ink_bgr`.
  `Library.image()` returns RGBA. (Storing scanned RGB cost 9× the disk for noise.)
- Akshara features carry `headline_y` / `baseline_y` relative to the crop, so
  aksharas can be aligned on a shared headline when assembled.

**Coverage of legal vocabulary** (55 common FIR words): 21 exist as whole words,
32 more can be assembled from aksharas, 2 need finer composition (नां = ना + ं,
फ्ता = half फ + ता) → 96% coverable.

### Style statistics (synth writers; `data/library/style_stats.json`)

Slant median −3° (per-writer medians −15° … +3°), headline angle ±3°, stroke width
≈ 0.075 × body height, upper matras ≈ 0.6 × body height, 94% blue ink.

## Task 2: generate handwritten pages

```bash
.venv/bin/python scripts/generate.py --n 100 --out outputs/synth [--seed 0] [--no-scan]
```

Each page is a synthetic printed Hindi FIR form (CrPC §154 layout) filled in by
one simulated writer, saved as `page_*.jpg` plus `page_*.json` ground truth:
every field's written text, line and word boxes in page pixels, and each word's
provenance (whole-word sample, assembled aksharas with sample ids, or drawn
strokes). ~0.2 s per page on one core.

Pipeline (`hwgen/synth/`):

1. **Style** (`style.py`) — one per page: writer, size, ink colour, word/akshara
   spacing, size and rotation jitter, extra slant, baseline drift, line tilt,
   pen pressure, fading, blots.
2. **Sample bank** (`bank.py`) — words and aksharas by text and writer; the page's
   own writer first, then the writers with the most similar slant/stroke weight.
   Mis-measured word samples and atypically sized akshara samples are filtered.
3. **Word rendering** (`assemble.py`) — a whole-word sample if one exists; else
   aksharas scaled to a common body height, aligned on their headlines (digits and
   danda on the baseline), stroke widths normalised to the writer's, and a headline
   bridged across the gaps only (never doubled). Backoffs: ं/ँ drawn on the base
   akshara; a missing conjunct written with an explicit halant (फ्ता → फ् + ता);
   `/ . , :` drawn as pen strokes (no samples in the data).
4. **Layout** (`layout.py`) — slant/rotation per word, wavering baseline, greedy
   wrapping on real rendered widths.
5. **Ink and page** (`degrade.py`, `compose.py`, `page.py`) — pressure/fade/blots,
   multiply-blend onto the page, fit to each dotted field (shrinking long values),
   signatures larger and more slanted; optional light scan effects (shading, noise,
   skew, JPEG).
6. **Content** (`content.py`, `forms.py`) — synthetic, consistent field values
   (random name combinations, generic places, template narrative). Not real people.

Measured on 200 pages: no field ever failed to render; 14% of words are
whole-word samples, the rest assembled; 20% of pieces come from a fallback writer.

## Task 3: verification

```bash
export ANTHROPIC_API_KEY=...                     # or: ant auth login
.venv/bin/python scripts/verify_pages.py --pages outputs/synth --out outputs/verify
.venv/bin/python scripts/real_vs_fake.py --n 200 --out outputs/real_vs_fake
# add --dry-run to either: builds every request and saves what would be sent, no API calls
```

Three checks (`hwgen/verify/`), all with Claude Opus 5.5 vision, structured JSON
output, the server-side refusal fallback, and an on-disk response cache
(`data/cache/vlm/`) so reruns cost nothing:

1. **OCR accuracy** — every handwritten line is cropped from the page (using the
   ground-truth word boxes) and transcribed by the VLM, which never sees the
   intended text. Scored as CER (code points), AER (akshara error rate — the
   generator's unit) and WER.
2. **Realism rating** — the page is rated 1–5 on six criteria (letterforms, joins
   and headline, size/spacing, ink, single-writer consistency, plausibility as a
   real FIR) against a fixed panel of **real** held-out handwriting, with a list of
   concrete artifacts per field (input for task 4).
3. **Real vs fake** — a real held-out word and a synthetic assembly of the *same*
   word (other writers' aksharas), rendered with identical ink, paper and
   degradation, side by side in random order; the VLM picks the assembled one.
   Detection accuracy with a 95% Wilson interval: 0.5 = indistinguishable.

**Pilot without an API** (`scripts/pilot.py`): blind sheets judged in-session — see
`outputs/pilot/FINDINGS.md` (40 pairs: detection 0.575, CI 0.42–0.72; OCR CER 0.032,
digits 0.078; realism 3.33/5).

Pages pass when mean line CER ≤ 0.15 and overall realism ≥ 3 (both flags).
Rough cost at the default effort: ~$0.10 per page, ~$0.03 per pair (estimates;
the scripts report actual token usage and cost).

### Known limitations

- **12 writers only** — limited style diversity; per-writer variation must be
  amplified by the generator's randomisation.
- Isolated words: no real **word spacing** or line flow; needs a line-level
  source (e.g. DohaScript, continuous handwritten Hindi).
- Akshara cuts are column cuts: a ि hook or reph that overhangs a neighbour can be
  clipped or carry a fragment. Task 3's verification should filter bad assemblies.
- **Digits are Devanagari only** and `/ . , :` are drawn: the source data has no
  Arabic numerals or punctuation, though Hindi FIRs often use Arabic numerals.
- Word spacing and line flow are hand-set ranges, not measured (isolated-word data).
- Forms are one synthetic template; real FIR layouts vary by state.
