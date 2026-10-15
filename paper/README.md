# Tensor-Structured Dictionary Learning — Beamer Talk

Source for the presentation of *A Unified Projection Framework for
Tensor-Structured Dictionary Learning, with an Efficiency Ranking System*
(Sahaj Chhetri).

## Uploading to Overleaf

1. Go to Overleaf → **New Project** → **Upload Project**.
2. Drop in `tensor-dl-beamer.zip` (this whole folder, zipped).
3. Overleaf should pick `main.tex` automatically. If it doesn't, open
   *Menu → Main document* and select `main.tex`.
4. **Compiler: pdfLaTeX** (Menu → Compiler). This is Overleaf's default; the
   deck is written to need nothing else — no XeLaTeX, no shell-escape, no
   bibliography pass.

## Files

```
main.tex          the whole talk — 31 frames incl. section dividers
figures/          all 15 figures extracted from the manuscript PDF
```

| File | Manuscript figure |
|---|---|
| `fig01_decompositions.png`    | Fig 1 — CP / TT / TR / BTD as dictionary tensors |
| `fig02_tucker_dictionary.png` | Fig 2 — Tucker: core as sparse matrix |
| `fig03_pareto_concept.png`    | Fig 3 — Pareto dominance schematic |
| `fig04_synthetic_atoms.png`   | Fig 4 — synthetic ground-truth atoms and patches |
| `fig05_frontier_synthetic.png`| Fig 5 — efficiency frontier, synthetic |
| `fig06_rank_sensitivity.png`  | Fig 6 — rank sensitivity sweep |
| `fig07_init_sensitivity.png`  | Fig 7 — initialization sensitivity boxplot |
| `fig08_rgb_data.png`          | Fig 8 — astronaut image and RGB patches |
| `fig09_frontier_rgb.png`      | Fig 9 — efficiency frontier, real RGB |
| `fig10_cp_instability.png`    | Fig 10 — CP degeneracy diagnostic |
| `fig11_btd_continuum.png`     | Fig 11 — BTD traces the CP↔Tucker continuum |
| `fig12_ranking_rgb.png`       | Fig 12 — ranking system, RGB patches |
| `fig13_ranking_synthetic.png` | Fig 13 — ranking system, synthetic patches |
| `fig14_matched_budget.png`    | Fig 14 — *see note below* |
| `fig15_hsi_cube.png`          | Fig 15 — synthetic HSI cube |

All 15 appear in the deck. Tables 1–7 are typeset natively with `booktabs`, and
Table 7 is additionally drawn as a `pgfplots` bar chart. The K-SVD/OMP loop is a
TikZ diagram drawn from scratch.

## Three things to fix in the manuscript

1. **Figure 14 is a duplicate of Figure 11.** The caption describes the
   matched-budget rate–distortion curve, but the embedded image is the BTD
   continuum plot. The deck shows it with a note (slide "Rate–distortion sweep
   across noise levels"); replace the image and delete that note.
2. **Figures 2 and 3 are captioned "Enter Caption"** in the manuscript. The deck
   supplies real captions — worth copying back.
3. **Number mismatches.** Table 4 gives the initial Tucker as 30.74 dB while
   Eq. 12 quotes 30.66 dB; Table 4 gives CP as 31.65 dB while Table 5 gives
   31.71 dB. The deck reports each as printed in its own table.

## Customising

- **Colours** are defined near the top of `main.tex` (`ink`, `amber`, `teal`,
  `indigo`, `rust`). Change them in one place.
- **Length**: the deck runs ~31 frames. For a 20-minute slot, the natural cuts
  are "Orthogonal matching pursuit, in two mechanics", "Rank sensitivity",
  "Why the matched-budget study is restricted to HSI", and the Fig 14 note slide.
- **Base font** is 9pt to fit dense frames. If you cut content, bump it to 10pt
  in the `\documentclass` line and recompile.
