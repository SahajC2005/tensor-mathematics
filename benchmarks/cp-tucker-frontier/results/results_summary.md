# Tensor Dictionary Learning Benchmark — Results

Dataset: mixed-structure 3rd-order patches (6, 6, 6), SNR=12.0 dB, true sparsity=3, encode sparsity=6.

| Model | Atoms | Dict params | Compression vs Vec | Fit time (s) | Test PSNR (dB) | PSNR std (init) |
|---|---:|---:|---:|---:|---:|---:|
| Vectorized | 120 | 25920 |   1.0x |  0.37 | 41.48 | 0.24 |
| CP | 120 | 2160 |  12.0x |  1.14 | 33.19 | 0.09 |
| Tucker | 343 | 126 | 205.7x |  0.11 | 27.42 | 0.10 |
| TT | 120 | 5760 |   4.5x |  0.43 | 38.29 | 0.23 |
| TR | 120 | 8640 |   3.0x |  0.47 | 38.95 | 0.23 |
