# HBAC Paper (camera-ready PDF)

Build the PDF from Paper v2 content:

```bash
cd paper
make pdf
```

Output: `paper/main.pdf` (also copied to `research docs/Paper v2.pdf`).

## Contents

- `main.tex` — LaTeX source (Track A stub + Track B V3, fixed n=300 oracle)
- `generate_figures.py` — per-benchmark + floor dose-response figures
- `references.bib` — symlink to `../references.bib`
- `figures/` — generated PDF plots

## Requirements

- TeX Live (`pdflatex`, `bibtex`)
- Python 3 + `matplotlib` (for figures)

## Venue formatting

Uses standard `article` class (11pt, 1in margins, Times/LM fonts). For a specific venue (NeurIPS, ICML, ACL), swap the document class and style file in `main.tex`.
