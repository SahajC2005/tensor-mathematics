# Tensor Mathematics

Tensor decompositions and structured tensor dictionary learning — implementations,
benchmarks, notes, and paper source.

The central question in this repo: **if a dictionary atom is a tensor, what low-rank
structure should you constrain it to?** Each structure trades parameter count against
reconstruction quality, and `benchmarks/` measures that frontier directly.

## Layout

```
src/
  decompositions/        CP, Tucker, Tensor-Train, mode products
  dictionary_learning/   structured tensor DL models and solvers
  utils/                 data generation, checkpoint inspection
benchmarks/
  cp-tucker-frontier/    main benchmark: PSNR vs dictionary size, 4 studies
  tensor-basic/          small CP error-vs-rank study
notebooks/               walkthroughs and benchmark explorations
notes/                   HTML writeups (greedy coding, PCA/SVD, convexity, Tucker)
paper/                   LaTeX source + figures
docs/                    proposals, slides, reference PDFs
```

## Core module

`src/dictionary_learning/tensor_dl.py` learns an over-complete dictionary of K atoms,
each a 3rd-order tensor. Models differ *only* in the structure each atom lives on:

| Model | Atom structure | Params per atom |
|---|---|---|
| `vectorized` | dense, no structure | `prod(shape)` |
| `cp` | CP rank-R | `R * sum(shape)` |
| `tt` | Tensor-Train, internal rank r | `r*s0 + r*s1*r + r*s2` |
| `tr` | Tensor-Ring, all ranks r | `r^2 * sum(shape)` |
| `btd` | b blocks of rank-(L,L,L) Tucker | `b*(L^N + L*sum(shape))` |
| `tucker` | one separable/Kronecker dictionary | `sum_n s_n * k_n` (all atoms) |

## Benchmark result

Mixed-structure 3rd-order patches (6,6,6), SNR 12 dB, true sparsity 3:

| Model | Atoms | Dict params | Compression | Test PSNR (dB) |
|---|---:|---:|---:|---:|
| Vectorized | 120 | 25920 | 1.0x | 41.49 |
| CP | 120 | 2160 | 12.0x | 33.14 |
| TT | 120 | 5760 | 4.5x | 38.29 |
| TR | 120 | 8640 | 3.0x | 38.95 |
| Tucker | 343 | 126 | 205.7x | 27.01 |

Dense atoms win on raw PSNR; TT/TR hold most of that quality at 3-4x fewer parameters;
Tucker is a 200x compression at a real accuracy cost. Full tables in
`benchmarks/cp-tucker-frontier/results/`.

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python benchmarks/cp-tucker-frontier/run_benchmark.py
```

## Related work (not included here)

This repo contains only original code. Two third-party reference implementations were
studied alongside it and are **not** vendored:

- [cohenjer/Tensor_codes](https://github.com/cohenjer/Tensor_codes) — DCPD, M2PALS, coupled and nonlinear CPD (J. Cohen)
- [srambhatla/TensorNOODL](https://github.com/srambhatla/TensorNOODL) — tensor NOODL / FISTA, MATLAB (S. Rambhatla)

`docs/references/third-party/` holds personal copies of published papers for reading
only — see the NOTICE there.
