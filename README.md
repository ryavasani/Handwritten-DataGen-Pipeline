# CV-Research — Synthetic Hindi handwriting for legal documents

Writes realistic Hindi (Devanagari) handwriting onto clean document pages using
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
.venv/bin/python scripts/generate.py --n 10 --out outputs/synth
```

Each page is saved as `page_*.jpg` plus `page_*.json` (the text of every field
and a box for every word).

### Verification (optional)

Uses Claude as a VLM judge and needs an API key:

```bash
export ANTHROPIC_API_KEY=...
.venv/bin/python scripts/verify_pages.py --pages outputs/synth --out outputs/verify
.venv/bin/python scripts/real_vs_fake.py --n 200 --out outputs/real_vs_fake
```

Add `--dry-run` to either to build the requests without calling the API.
Without a key, `scripts/pilot.py` builds blind sheets for a manual evaluation.

### Notes

- The printed form uses the macOS Kohinoor Devanagari font
  (`hwgen/synth/forms.py`, `FONT`). On Linux, point it at a Devanagari font such
  as Noto Sans Devanagari.
- Please cite the IIIT-HW-Hindi paper if you use the data.
