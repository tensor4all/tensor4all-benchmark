# Gaussian MPO patching decision log

This document records the experiment history, invalidated interpretations, and restart criteria for Case 3. It is the durable context for future work; raw JSON and profile reports remain the evidence for individual runs.

## Question

The maintained question is:

> For the same organically high-rank two-dimensional Gaussian-mixture inputs and the same external accuracy gate, when does cap-128 spatial patching make `f(x,y) g(y,z)` contraction faster or more feasible than global fit?

A separate scalability question is whether adaptive patched input construction and compatible-pair contraction can be distributed with Hataori once the tensor integration is ready.

## Stable benchmark definition

- Input functions are positive randomly rotated anisotropic Gaussian mixtures `f(x,y)` and `g(y,z)`.
- Quantics resolution is fixed at `R=16`.
- Centers occupy an active square inside a computational square with four times the active half-width.
- Production input construction applies Global TCI to the whole mixture, followed by one relative-L2 compression.
- Localized evaluation omits tails only under a rigorous absolute bound and deterministic pivots include component centers and principal-axis points.
- Contraction integrates the shared `y` axis; the reference is the analytic infinite-`y` integral of retained significant Gaussian pairs.
- The maintained patch cap is 128. `BENCH_PATCH_CAP` exists only for explicit diagnostics; production profiles use 128.
- Timings exclude input generation, cache I/O, input compression, conversion, patch preparation, reconstruction validation, output conversion, reference construction, and accuracy evaluation.

## What was implemented

The Case 3 patched path now:

1. converts the two global MPOs to chain TreeTNs;
2. discovers adaptive left `x/y` and right `y/z` partitions;
3. refines nonzero leaves to disjoint regular Cartesian binary partitions;
4. contracts only projector-compatible pairs with cap-bounded contributions;
5. forms each existing `x/z` group with a cap-bounded initial sum and uncapped `fit_sum`;
6. applies no recursive output splitting or hard fitted-output cap;
7. performs one final adaptive truncation;
8. validates both arms at the same retained analytic-reference centers.

The unused legacy `PartitionedTT` construction and contraction path was removed in commit `132d5da`; Case 3 now eagerly prepares only the maintained global and chain-TreeTN representations.

## Accuracy-policy correction

TreeTN partition patching exposes a local SVD cutoff, not a whole-chain tolerance. Earlier runs reused the requested whole-chain cutoff at every local SVD. Those runs could not support a same-accuracy timing claim.

For an `R`-node chain, the benchmark now uses

```text
local_rtol = requested_rtol / sqrt(2 * (R - 1)).
```

At `R=16` and requested `1e-6`, this gives 30 edge visits, local norm tolerance `1.8257418583505536e-7`, and squared local SVD cutoff `3.333333333333333e-14`. Exact reconstructed-input residuals remain authoritative. The corrected 14-record layout study has `patch_tolerance_met=true` throughout, with maximum measured residual `8.934675197232979e-7`.

Consequences:

- timings under `result/linux-epyc-7713p-padded-r16/` are explicitly pre-correction historical evidence;
- their indicative χ115/χ263/χ381 speedups must not be cited as current same-accuracy results;
- the corrected layout report is structural and contraction-free, so it also does not establish a speedup.

## Balanced versus shared-y-only study

The corrected layout study compared:

- balanced left `x/y`, right `y/z` partitions;
- shared-`y`-only partitions.

It records exact reconstruction residuals, patch ranks, parameter counts, compatible pairs, output-projector groups, and structural work proxies. Shared-y-only often reduces compatible-pair and output-group counts, but these proxies are not FLOPs, timings, or speedups. Layout alone does not determine runtime because local ranks, fitted-output ranks, parameter distributions, and summation costs also matter.

Evidence: [`../result/linux-epyc-7713p-patch-layout-scaling-r16/report.md`](../result/linux-epyc-7713p-patch-layout-scaling-r16/report.md).

## Three-dimensional input detour

A fully correlated positive Gaussian family `A(b,x,y)` was embedded as the batch-diagonal MPO

```text
A(b,x;b',y) = delta(b,b') A(b,x,y).
```

The embedding preserves QTT bond dimensions exactly. The archived default-`rtol=1e-6` records reached compressed χ67, 125, 176, 201, 223, and 251 for `N=1,2,4,8,16,32`, with principal-axis errors below `1e-6`; `N=64` input construction exceeded 570 seconds. A later interactive cache-recompression session at requested `rtol=1e-8` observed χ115, 215, 291, 311, 326, and 370 with principal-axis errors below `1e-8`, but those records were not archived and must be rerun before citation as repository evidence.

The expanding active box keeps Gaussian density approximately constant, so this family did not show `N proportional to chi^2`. A fixed-box mode was implemented to increase overlap density, but its sweep was stopped when the research direction was reconsidered. The 3D work remains an input-rank study and performs no patching or contraction.

Evidence: [`../result/linux-epyc-7713p-gaussian3d-rank-r16/report.md`](../result/linux-epyc-7713p-gaussian3d-rank-r16/report.md).

## Direct-product detour and invalid comparison

Exact doubled-space self-products were tested:

```text
F(x,x';y,y') = f(x,y) tensor_product f(x',y')
G(y,y';z,z') = g(y,z) tensor_product g(y',z').
```

They verify bond by bond that `chi_product = chi_factor^2`, reaching materialized χ2304. This is valid as a synthetic representation and memory study.

It is not a suitable contraction-speed benchmark because

```text
(F contract G) = (f contract g) tensor_product (f contract g).
```

The structure-aware solution contracts the original factor and forms the result's direct product. A generic doubled-space contraction would intentionally ignore known separability.

A later cap diagnostic used factor cap 11 because `11^2 <= 128`. It measured a 2D global factor contraction against a 2D patched factor contraction and squared only the structural patch counts. The global timing therefore exploited factorization relative to the hypothetical generic doubled problem. It does **not** answer whether generic cap-128 patching beats generic doubled-space global fit. Any earlier statement that it did is retracted.

Valid conclusions from the detour are limited to:

- exact direct products square every factor bond dimension;
- the tested factor-level cap-11 patch decomposition had large overhead;
- no doubled-space contraction timing or speedup was measured.

Evidence:

- [`../result/linux-epyc-7713p-direct-product-r16/report.md`](../result/linux-epyc-7713p-direct-product-r16/report.md)
- [`../result/linux-epyc-7713p-direct-product-cap128-proxy-r16/report.md`](../result/linux-epyc-7713p-direct-product-cap128-proxy-r16/report.md)

Direct products are not part of the planned Case 3 contraction study.

## Current evidence boundary

### Established

- Organic 2D Gaussian mixtures provide spatial locality and projector-compatible `y` joins, so patching has a plausible algorithmic advantage at sufficiently high χ.
- Corrected cap-128 input partitions can satisfy the requested reconstructed-input tolerance.
- Preliminary pre-correction timings suggested a crossover, but are only motivation for rerunning.
- Case 3 is now TreeTN-only and no longer pays unused legacy `PartitionedTT` preparation.

### Not established

- No corrected-tolerance high-χ global-versus-patched contraction speedup has been measured.
- No fair generic direct-product global-versus-patched comparison exists.
- Structural pair/rank proxies are not measured FLOPs or timings.
- A global timeout is not a timing; at most it provides a censored lower bound when the patched arm completes under the same command limit.

## Why χ near 1000 is difficult

A dense `R=16` MPO with χ near 1000 is already roughly gigabyte-scale for two inputs, and global fit creates larger intermediates. Global TCI construction also becomes expensive; point-coordinate evaluation is currently sequential. The global variational sweep has site-to-site dependencies and cannot be distributed as independent coarse tasks, although each local dense kernel may use threaded tenferro operations.

Therefore a χ≈1000 study is likely to have:

- a bounded single-rank global attempt rather than an exact global timing;
- patched execution as the only arm that naturally exposes independent compatible-pair tasks;
- separate reporting of single-node algorithmic comparison and multi-node patched scaling.

## Hataori decision

Further high-χ work is paused until Hataori's tensor integration is tested.

Hataori P0 dynamically assigns a fixed frontier of coarse tasks across MPI ranks and rank-local Rayon domains. It does not currently add child tasks during one `pmap`, but adaptive patching can use repeated frontier collectives:

```text
run local TCI on current frontier
  -> accept patches meeting rank/error criteria
  -> split rejected patches
  -> pmap the next frontier
```

This is a better fit than coordinate-level distributed evaluation:

- each patch-local TCI retains its sequential pivot logic internally;
- separate patches are coarse independent tasks;
- accepted partitioned inputs flow directly into compatible-pair contraction;
- compatible pairs can be distributed, while each contraction uses the rank-local tenferro/Rayon domain.

Required integration gates are:

1. tensor4all-rs issue #663: canonical explicit execution contexts;
2. Hataori Phase 20b: tensor wire metadata and target-context reconstruction;
3. Hataori Phase 20c: joint Hataori/tensor4all/tenferro MPI validation.

Hataori will not distribute the global fit sweep. Its expected benefit is making adaptive patched TCI and patched contraction scalable and feasible. Multi-rank patched results must be reported as scalability/feasibility evidence, not as a same-resource algorithm-only speedup over a single-rank global arm.

## Restart plan

After the Hataori integration gates pass:

1. implement frontier-based adaptive patched TCI for the existing 2D Gaussian family;
2. allocate a global squared-L2 budget across disjoint patches and validate the reconstructed input exactly;
3. run compatible-pair contractions as Hataori coarse tasks with cap 128;
4. group and `fit_sum` contributions by output projector, then apply one final truncation;
5. first validate a single-node corrected-tolerance comparison around χ350–500;
6. then test χ≈1000-equivalent patched feasibility and 1/2/4/8-rank scaling;
7. run global fit only as a bounded single-rank attempt at the largest point;
8. report wall time, CPU-seconds, communication bytes, patch/pair distributions, memory, and external error separately.

The project should stop if corrected single-node results show that patch preparation and compatible-pair overhead dominate throughout the reachable organic-rank range. It should not add another synthetic high-rank family merely to manufacture a speedup.
