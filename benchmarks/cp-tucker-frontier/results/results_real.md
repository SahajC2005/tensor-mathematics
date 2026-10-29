# Extended results — real RGB image patches

Image: astronaut, patch (8, 8, 3), SNR=12.0 dB, encode sparsity=6.

| Model | Atoms | Dict params | Compression | Fit (s) | Test PSNR (dB) |
|---|---:|---:|---:|---:|---:|
| Vectorized | 100 | 19200 |  1.0x | 0.30 | 30.83 |
| CP | 100 | 1900 | 10.1x | 0.74 | 31.68 |
| Tucker | 243 | 153 | 125.5x | 0.09 | 32.17 |
| TT | 100 | 5400 |  3.6x | 0.38 | 31.27 |
| TR | 100 | 7600 |  2.5x | 0.40 | 31.27 |
| BTD | 100 | 4600 |  4.2x | 1.00 | 31.37 |
