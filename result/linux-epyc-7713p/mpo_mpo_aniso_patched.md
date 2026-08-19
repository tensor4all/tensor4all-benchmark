# Patched MPO-MPO fit contraction

Profile metadata: [`run.yaml`](run.yaml). Raw records: [`raw/`](raw/).

## Conditions

- `N = 2048`, `R = 10`, relative squared-value discarded-tail cutoff `1e-8`
- patch bond cap `128`; cap `64` is intentionally excluded
- one pinned CPU core; Rayon, OpenMP, OpenBLAS, MKL and vecLib limited to one thread
- two measured runs, no warmup
- identical cached global MPO inputs for the global and patched arms
- input generation, HDF5 loading, patch construction and output conversion excluded from contraction timing

## Results

The reported median is the harness's upper median for two runs.

| Input χ | Input patches (left/right) | Global runs [s] | Global median [s] | Patched runs [s] | Patched median [s] | Global / patched | Output patches | Max output-patch χ | Patched error |
| ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 / 1 | 3.576, 3.515 | 3.576 | 3.321, 3.292 | 3.321 | 1.08× | 1 | 78 | 3.455e-4 |
| 192 | 8 / 8 | 11.791, 11.682 | 11.791 | 13.353, 13.244 | 13.353 | 0.88× | 1 | 68 | 5.547e-4 |
| 224 | 8 / 8 | 15.348, 15.468 | 15.468 | 13.074, 13.103 | 13.103 | 1.18× | 1 | 68 | 4.917e-4 |
| 256 | 8 / 8 | 19.606, 19.558 | 19.606 | 12.274, 12.096 | 12.274 | **1.60×** | 1 | 68 | 6.281e-4 |

The meaningful crossover is between input χ `192` and `224`: patched takes
1.13× the global time at χ `192`, while global takes 1.18× the patched time at
χ `224` and 1.60× at χ `256`. The χ `128` point has one input patch per side
and therefore does not measure a partitioning advantage.

The χ `128`, `192`, and `224` inputs hit their respective TCI construction caps;
they are cap-limited views of the same instance, while the χ `256` input was
built with cap `384` and converged below it.

## Input MPO size

Pure `f64` core storage, excluding index and HDF5 metadata:

| Input χ | Left values / MiB | Right values / MiB | Pair MiB | HDF5 MiB |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 166,440 / 1.270 | 167,552 / 1.278 | 2.548 | 2.930 |
| 192 | 282,964 / 2.159 | 281,472 / 2.147 | 4.306 | 4.689 |
| 224 | 314,912 / 2.403 | 314,912 / 2.403 | 4.805 | 5.187 |
| 256 | 346,176 / 2.641 | 347,976 / 2.655 | 5.296 | 5.678 |

## Accuracy status

All points exceed the runner's unchanged `1e-4` sanity gate under this
ITensor-style cutoff, so each process exits nonzero after writing its records.
The records are retained because the comparison and crossover are valid, but
these runs do **not** claim that the accuracy gate passed. No tolerance or gate
was relaxed.

The recorded rerun uses the merged tensor4all-rs revision `9e9aeda`, which is
also the revision pinned by this repository.
