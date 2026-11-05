# Synthetic Hindi and Telugu handwriting

Writes realistic Hindi (Devanagari) and Telugu handwriting from real handwriting
samples (OpenCV + NumPy), with exact ground truth for every word.

## Use it in your own document pipeline

**1. Install** (Python 3.10+):

```bash
pip install git+https://github.com/ryavasani/Handwritten-DataGen-Pipeline
```

**2. Build a sample library once.** The handwriting is cut from public research datasets that can't be redistributed, so the libraries aren't included. Clone the
repo and follow [Data](#data) and [Build the sample library](#build-the-sample-library-once)
for either language (Telugu needs Google Cloud Vision for this one-time build only).
This gives you `data/library` (Hindi) or `data/library_te` (Telugu).

**3. Write your text:**

```python
from hwgen import Handwriter, degrade

hw = Handwriter("data/library_te")             # load once (a few seconds), reuse for every page
hand = hw.write("మీ వచనం ఇక్కడ", height=20, max_width=900, seed=0)
labels = hand.paste(page, x=100, y=200)        # page: your BGR uint8 image, changed in place
page, labels, layers = degrade(page, labels, level=0.8, seed=0)   # optional: age, damage, re-photograph
```

- `labels` is one entry per word: `{"text", "line", "box": [x, y, w, h]}` in page pixels.
- `hand.text` is exactly what was written. Words the library can't write (rare
  letters, `[ ] *`) are left out and listed in `hand.skipped`, so use `hand.text`
  as the label, not your input string.
- `write` options: `height` (letter height in px), `max_width` (wrap; newlines
  always break), `irregular` and `spacing` (0-1, an uneven hand), `writer` (one of
  `hw.writers`), `line_spacing`, `color` (BGR), `seed` (same seed, same handwriting).
- `hand.ink` is the bare ink (float 0-1) with `hand.color`, if you composite
  yourself; `hand.image()` puts it on white.
- `degrade` moves the boxes along with the page; `capture` picks `phone`,
  `flatbed`, `photocopy`, `fax` or `archive`.

## Run this repo

```bash
git clone https://github.com/ryavasani/Handwritten-DataGen-Pipeline.git
cd Handwritten-DataGen-Pipeline
python3 -m venv .venv
.venv/bin/pip install -e ".[ocr]"
```

### Data

**Hindi:** IIIT-HW-Hindi v1 (Dutta et al., DAS 2018; ~1.8 GB), single handwritten
word images with labels, from
<https://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data>:

```bash
mkdir -p data/raw/iiit_hw_hindi && cd data/raw/iiit_hw_hindi
curl -fLO -C - https://cvit.iiit.ac.in/images/Projects/wordlevel-Indicscripts/IIIT-HW-Hindi_v1.tar.gz
gzip -t IIIT-HW-Hindi_v1.tar.gz      # the server can drop the connection; rerun curl until this passes
tar -xzf IIIT-HW-Hindi_v1.tar.gz && tar -xzf HindiSeg.tar.gz && rm HindiSeg.tar.gz IIIT-HW-Hindi_v1.tar.gz
cd ../../..
```

**Telugu:** ICDAR 2024 handwritten Indic competition, page-level task (PLHWTR-1.0),
Telugu training set (IIIT Hyderabad, <https://ilocr.iiit.ac.in/icdar_2024_hwd/dataset.html>):
phone photos of whole handwritten pages, each with its transcription. The zip is
3 GB, so the script reads only the pages it picks (~220 MB):

```bash
.venv/bin/python scripts/fetch_real_telugu.py    # 50 pages + 200-page pool -> data/raw/telugu_real
```

### Build the sample library (once)

```bash
# Hindi
.venv/bin/python scripts/build_library.py     # ~25 min -> data/library/ (word samples)
.venv/bin/python scripts/build_aksharas.py    # ~2 min  -> adds akshara (syllable) samples

# Telugu: Google Vision finds the words on each page (one call per page, cached)
gcloud auth application-default login         # once
.venv/bin/python scripts/build_library_te.py --pool   # ~1 min -> data/library_te (179 writers)
```

Each Vision word box gets the ground-truth word it aligns to, and words are cut
into aksharas along connected components.

### Generate pages

```bash
.venv/bin/python scripts/generate.py --n 10                                    # plain handwritten legal documents -> outputs/documents_hi
.venv/bin/python scripts/generate.py --n 10 --messiness 0.9 --age 0.8 --out ...  # pin difficulty / paper age (0-1)
.venv/bin/python scripts/generate.py --n 10 --paper notebook --out ...          # paper: aged | notebook (plain/ruled)
.venv/bin/python scripts/generate.py --n 10 --degrade 0.8 --out ...            # + damage, capture (phone/flatbed/photocopy/fax), WhatsApp
.venv/bin/python scripts/generate.py --n 12 --degrade 0.8 --size 0.65 --spacing 0.7 --irregular 0.7   # what outputs/documents_hi holds
.venv/bin/python scripts/doc_ocr_eval.py --degrade 0.8 --size 0.65 --spacing 0.7 --irregular 0.7 --pages 3 --out outputs/doc_ocr_eval_hi
```

Each page is saved as `page_*.jpg` plus `page_*.json` (the text of every block
and a box for every word; struck-out words are marked `struck`). `doc_ocr_eval.py`
scores the pages with Google Vision OCR (needs the `gcloud` login above).

### Telugu

With `data/library_te` built (see [Data](#data)), add `--script te`:

```bash
.venv/bin/python scripts/generate.py --script te --n 12 --degrade 0.8 --size 0.65 --spacing 0.7 --irregular 0.7   # -> outputs/documents_te
.venv/bin/python scripts/doc_ocr_eval.py --script te --degrade 0.8 --size 0.65 --spacing 0.7 --irregular 0.7 --pages 3 --out outputs/doc_ocr_eval_te
```

All the document options above (`--messiness`, `--paper`, `--age`, `--degrade`,
`--capture`) work for Telugu. Four more shape the hand:

```bash
.venv/bin/python scripts/generate.py --script te --size 0.65 --spacing 0.7 --irregular 0.7 --out outputs/documents_te_small
```

- `--size`: letter height multiplier; below 1 the page holds more text instead of growing the letters to fill it.
- `--spacing` (0-1): uneven word and letter gaps, with the odd break inside a word.
- `--irregular` (0-1): slant that changes from word to word; in Telugu also per-letter size, tilt, lean and bounce, and a slight stroke wobble.
- `--tight` (0-1): letters, words and lines packed closer together.

`--capture archive` (with `--degrade`) makes a century-old sheet photographed on a
copy stand: tan mottled paper with crumbled edges, split along the folds, faded
brown ink, a worn rubber stamp, on a blue-grey backdrop.

```bash
.venv/bin/python scripts/generate.py --script te --degrade 0.8 --capture archive --tight 0.7 --out outputs/documents_te_archive
```

These also work for Hindi (`--size`, word spacing and per-word slant); at their defaults the output is unchanged.
