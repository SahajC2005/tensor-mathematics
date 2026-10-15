# Tensor Dictionary Learning Benchmark — Results

Dataset: mixed-structure 3rd-order patches (6, 6, 6), SNR=12.0 dB, true sparsity=3, encode sparsity=6.

| Model | Atoms | Dict params | Compression vs Vec | Fit time (s) | Test PSNR (dB) | PSNR std (init) |
|---|---:|---:|---:|---:|---:|---:|
| Vectorized | 120 | 25920 |   1.0x |  0.85 | 41.49 | 0.29 |
| CP | 120 | 2160 |  12.0x |  3.50 | 33.14 | 0.11 |
| Tucker | 343 | 126 | 205.7x |  1.42 | 27.01 | 0.11 |
| TT | 120 | 5760 |   4.5x |  1.13 | 38.29 | 0.23 |
| TR | 120 | 8640 |   3.0x |  1.24 | 38.95 | 0.23 |
