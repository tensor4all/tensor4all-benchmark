# tensor4all-rs PR 694 SRC performance study

This directory freezes the single-thread adaptive successive randomized compression investigation performed for [tensor4all-rs PR 694](https://github.com/tensor4all/tensor4all-rs/pull/694). It is a historical optimization study, not a fourth maintained benchmark case. The SRC arms themselves are added to the maintained Gaussian MPO contraction case by this branch.

`run.yaml` records the machine, revisions, thread policy, benchmark shape, and limitations. `raw/` contains the generated JSON records. `scripts/` contains the measurement scripts used during the investigation. `sha256.txt` protects the archived inputs. The two patch files reconstruct the measured Rust states from base revision `48ad4021d02a98a52dcffffb226ca48fbb6e2748`:

- `tensor4all-rs-pre-typed-probe.patch.gz`: generalized untracked adoption, before the final probe construction and no-gradient inspection changes.
- `tensor4all-rs-candidate.patch.gz`: final uncommitted candidate, including generalized untracked adoption, typed probe construction, and no-gradient inspection removal.

Decompress and apply a patch to a clean checkout at the recorded base revision (`gzip -dc FILE.patch.gz | git apply`), then build `benchmark_src` with:

```bash
CARGO_PROFILE_RELEASE_CODEGEN_UNITS=256 cargo build --release \
  -p tensor4all-treetn --example benchmark_src
```

The scripts use absolute `/tmp` paths for the two checkouts and benchmark binaries. These are deliberate records of the executed commands. Adjust only those paths when reproducing the study.

## Comparison contract

All timed Rust and Python runs used one Rayon and BLAS thread, warmup where recorded, and fixed CPU affinity. The common Rust case was a 10-site MPO-MPS contraction with physical dimension 2, equal MPO and MPS input bonds, adaptive maximum rank 32, and rank increment 3.

The Python and Rust arms match settings and tensor shapes but generate different tensor values and use different tensor implementations. Their ratio is a workload comparison, not an identical-input kernel comparison. Alternating Rust before/after pairs are the primary optimization evidence. Absolute time moved with shared-host load.

The Python checkout is used only for measurement and algorithm comparison. No implementation was copied because no license was detected.

## Retained optimization results

The percentages below come from separate paired experiments and must not be multiplied together. Each row compares the named change with its immediate predecessor.

| Change | Bond 32 | Bond 64 | Bond 128 | Record |
| --- | ---: | ---: | ---: | --- |
| Favorable SRC operand order | -6.1% | -15.0% | -19.5% | `raw/pr694_probe_order_ab.json` |
| Session-native borrowed einsum | -33.2% | -18.3% | -8.1% | `raw/pr694_session_ab.json` |
| Generalized untracked result adoption | -7.44% | -1.37% | -2.75% | `raw/pr694_adopt_untracked_ab.json` and related records |
| Typed probes plus no-gradient inspection removal | -9.05% | -3.00% | noise-level | `raw/pr694_typed_probe_final_ab.json` |
| Reuse planner validation for dense derived results | -2.18% | not measured | not measured | `raw/pr694_validated_wrap_safe_b32.json` |

Lazy prefix generation reduced the bond-128 time by 18.75% by generating only the requested adaptive columns. Its detailed measurements and arithmetic counts are preserved in the tensor4all-rs worklog included by the candidate patch.

The final same-period bond-32 triplet measured:

| Rust baseline | Rust candidate | Python | Rust candidate / Python | Candidate gap |
| ---: | ---: | ---: | ---: | ---: |
| 36.463 ms | 33.582 ms | 20.022 ms | 1.677 | 13.560 ms |

In that triplet the latest Rust change reduced its paired baseline by 7.22%. The Rust/Python ratio moved from 1.821 to 1.677 and the absolute gap from 16.441 ms to 13.560 ms. See `raw/pr694_final_b32_triplet.json`. A longer Rust-only run measured a 9.05% paired median, so the report keeps both rather than selecting the more favorable number.

After the derived-result validation optimization, a fresh same-period five-pair sweep measured the final candidate across all input bond dimensions. Positive gaps mean Rust was slower.

| Input bond | Rust | Python | Rust - Python | Rust / Python |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 36.158 ms | 20.289 ms | +15.869 ms | 1.782 |
| 64 | 77.072 ms | 79.303 ms | -2.231 ms | 0.972 |
| 128 | 338.204 ms | 482.313 ms | -144.109 ms | 0.701 |

See `raw/pr694_final_matched_python.json`. Absolute bond-32 time was higher than in the preceding triplet because of shared-host load; the alternating Rust-only A/B result remains the evidence for the optimization itself.

## Post-merge tenferro-rs PR 1751 follow-up

After [tenferro-rs PR 1751](https://github.com/tensor4all/tenferro-rs/pull/1751) merged as `0190e43b1a331670c7d4eeb2d1d138c811ed9eae`, the final tensor4all-rs candidate was rebuilt and compared in four alternating lanes: the original tenferro revision with BCGS2, the merged revision with BCGS2, the merged revision with the compact Householder adapter, and Python. Processes used CPU 63 and one Rayon/BLAS thread. Each accepted process had at least 95% CPU-time/wall-time utilization for Rust and 75% for Python. Bond 32 uses 11 accepted pairs; bonds 64 and 128 use seven.

| Input bond | Original BCGS2 | Merged BCGS2 | Merged Householder | Python | Merged BCGS2 / Python | Householder / Python |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 36.435 ms | 35.611 ms | 34.960 ms | 20.251 ms | 1.758 | 1.726 |
| 64 | 77.829 ms | 76.685 ms | 76.383 ms | 79.270 ms | 0.967 | 0.964 |
| 128 | 339.667 ms | 325.798 ms | 326.859 ms | 468.302 ms | 0.696 | 0.698 |

The merged tenferro revision improved the BCGS2 median by 2.26%, 1.47%, and 4.08% at input bonds 32, 64, and 128. The compact Householder adapter changed the corresponding merged-revision BCGS2 medians by -1.83%, -0.39%, and +0.33%; it is therefore effectively neutral for this SRC workload rather than a broadly established replacement. Rust remains 75.8% slower than Python at bond 32, while it is 3.3% and 30.4% faster at bonds 64 and 128. The dense-oracle relative error was at most `6.021e-26` for merged BCGS2 and `4.703e-26` for merged Householder.

The accepted rows and paired statistics are in `raw/pr694_post_tenferro_1751_final.json`; independent dense-oracle runs are in `raw/pr694_post_tenferro_1751_correctness.json`. A later bond-64 run crossed into a shared-host memory/frequency-contention regime even though per-process CPU utilization remained high. It was excluded and retained transparently as `raw/pr694_post_tenferro_1751_rejected_b64_mixed_regime.json`.

## Remaining tensor4all-owned fixed costs

Temporary section instrumentation was applied to the final candidate and removed after measurement. One bond-32 run contained 254 N-ary contractions and 545 result/index validations. The median section totals below are nested. They are not additive.

| Section | Median total | Calls | Interpretation |
| --- | ---: | ---: | --- |
| Core contraction-plan construction | 1.649 ms | 254 | Fresh index identities prevent exact-key plan reuse. |
| Native operand preparation | 0.763 ms | 254 | Borrowed read wrappers and ID slices. |
| Untracked result adoption and wrapping | 1.232 ms | 254 | Eager adoption, shape checks, axis-class handling, and storage assembly. |
| Backend session shell | 0.633 ms | 254 | Derived as session total minus concrete-plan execution. |
| Backend input and label preparation | 0.423 ms | 254 | `prepare_inputs` plus `build_subscripts`. |
| Core size validation | 0.132 ms | 254 | Repeated internal-label dimension checks. |
| Result index validation | 0.282 ms | 545 | Expected dimensions plus unique-index validation. |
| No-gradient scan | 0.026 ms | 254 | Remaining `tracks_grad` check after removing AD-only dtype/layout inspection. |

The deep profile is stored in `raw/pr694_final_deep_fixed_cost_profile.json`. Its instrumentation inflates `plan_lookup` and total execution time, so those values are diagnostic only. The earlier low-overhead profile measured backend plan lookup at 0.449 ms and is retained in `raw/pr694_untracked_full_profile_medians.json`.

A follow-up split of the result boundary measured 0.641 ms in backend-value adoption, 0.265 ms in unique-index validation, and 0.196 ms in axis-class validation. Contraction planners already establish the result-index invariants. Reusing that proof for dense derived results while retaining native-shape and structured-layout validation improved bond 32 by a 2.18% paired median over ten alternating pairs (`raw/pr694_validated_wrap_safe_b32.json`).

The main remaining tensor4all-side opportunities are therefore a caller-owned positional prepared-plan API and a safely reusable execution-session boundary. The generic `TensorLike` SRC implementation currently has no minimal prepared-plan seam, and the attempted transparent caches and larger fused contractions did not establish a win.

## Rejected experiments

| Experiment | Result | Record |
| --- | --- | --- |
| Explicit probe-axis permutation | About 18% slower | tensor4all-rs worklog |
| Whole-SRC caller-owned session | Neutral at low bond, 5.76% slower at bond 128, and unsafe re-entry without fallback | `raw/pr694_sweep_session_ab.json` |
| Compact probe-storage deferral | 3.53% slower at bond 32 | `raw/pr694_compact_probe_ab.json` |
| Exact-index contraction-plan cache | 0 hits and 263 misses | tensor4all-rs worklog |
| Planner HashMap to Vec rewrite | -0.08%, effectively neutral | `raw/pr694_vec_plan_b32_ab.json` |
| Backend plan-cache linear lookup | +0.14%, effectively neutral | `raw/pr694_vec_backend_plan_cache_b32.json` |
| Linear internal-label grouping | 1.63% slower | `raw/pr694_linear_internal_ids_b32.json` |
| Prefix and sketch N-ary fusion | Removed 57 of 230 boundaries but was neutral over 100 repetitions | `raw/pr694_fusion_b32_100reps.json` |
| Compact Householder adapter | 41% to 52% slower on repeated small appends | `raw/pr694_incremental_qr_general_ab.json` |

The compact Householder follow-up is tracked in [tenferro-rs issue 1750](https://github.com/tensor4all/tenferro-rs/issues/1750).

## Correctness and validation

The final dense-oracle relative errors were at most `6.255e-26`, `1.176e-28`, and `2.240e-31` for input bonds 32, 64, and 128. The final tensor4all candidate passed:

- 442 tensor4all-core library tests, with one ignored diagnostic test;
- 498 tensor4all-treetn library tests, with one ignored diagnostic test;
- focused public API doctests;
- Clippy with CI deny flags;
- formatting, diff, API inventory, and repository-rules checks.

The tensor4all-rs worklog in the archived patch contains the full chronological interpretation and links each retained production change to its verification.
