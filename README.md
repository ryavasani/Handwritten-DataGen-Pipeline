# CV-Research — Synthetic Hindi and Telugu handwriting for legal documents

Writes realistic Hindi (Devanagari) and Telugu handwriting onto clean document pages using
real handwriting samples (OpenCV + NumPy), and outputs exact ground truth.

## Setup

Requires Python 3.10+ (tested on 3.13, macOS).

```bash
git clone https://github.com/ryavasani/CV-Research.git
cd CV-Research
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Data

Download IIIT-HW-Hindi v1 (Dutta et al., DAS 2018; ~1.8 GB) from
<https://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data>:

```bash
mkdir -p data/raw/iiit_hw_hindi && cd data/raw/iiit_hw_hindi
curl -fLO -C - https://cvit.iiit.ac.in/images/Projects/wordlevel-Indicscripts/IIIT-HW-Hindi_v1.tar.gz
gzip -t IIIT-HW-Hindi_v1.tar.gz      # the server can drop the connection; rerun curl until this passes
tar -xzf IIIT-HW-Hindi_v1.tar.gz && tar -xzf HindiSeg.tar.gz && rm HindiSeg.tar.gz IIIT-HW-Hindi_v1.tar.gz
cd ../../..
```

### Build the sample library (once)

```bash
.venv/bin/python scripts/build_library.py     # ~25 min -> data/library/ (word samples)
.venv/bin/python scripts/build_aksharas.py    # ~2 min  -> adds akshara (syllable) samples
```

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
scores the pages with Google Vision OCR (see Telugu below for the one-time `gcloud` login).

### Telugu

The same pipeline writes Telugu, with samples cut from real handwritten Telugu
pages (ICDAR 2024 PLHWTR, page-level, with ground truth) instead of a word
dataset. Google Vision finds the word boxes (`gcloud auth application-default
login` once; one call per page, cached), each box gets the ground-truth word it
aligns to, and words are cut into aksharas along connected components.

```bash
.venv/bin/python scripts/fetch_real_telugu.py           # 50 pages + 200-page pool -> data/raw/telugu_real
.venv/bin/python scripts/build_library_te.py --pool     # ~1 min -> data/library_te (179 writers)
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

### Notes

- Please cite the IIIT-HW-Hindi paper if you use the data, and the ICDAR 2024
  Indic handwriting competition for the Telugu pages ("All Rights Reserved":
  research use, do not redistribute).
