# Update to latest tensor4all-rs and expand the default sweeps

Date: 2026-08-07
Status: approved by repo user (lingrui96), pending execution

## Goal

Benchmark the current tip of tensor4all-rs with heavier sweeps than the two and a half
minute defaults, and produce a question list for Hiroshi about the next benchmark case.

## Decisions

1. **Pin moves to the latest origin/main of tensor4all-rs** (`ae655a9` at design time,
   re-check at execution). The single new commit over the old pin is #575, a treetci
   convergence fix authored by the repo user, expected to affect input TCI construction
   time in cases 2 and 3, not the measured arm errors. If arm errors move, that is a
   finding to record, not to paper over.
2. **The pin update and the sweep expansion land together in one commit**, by explicit
   instruction of the repo user, overriding the AGENTS.md rule that a bump gets its own
   commit. Old results are superseded, not preserved for comparison.
3. **Default sweeps grow to a roughly 20 minute full run**, and `result/mac-cpu` is
   regenerated as the single standard result set. No separate heavy profile.
4. **No new benchmark case yet.** Candidate directions and open questions go to Hiroshi
   first (see below).

## Plan of work

1. Update every `rev` in `Cargo.toml` to the latest tensor4all-rs origin/main. All seven
   crates move together.
2. Probe actual costs on the new rev with `OUT_DIR` pointed at a scratch directory:
   cases 2 and 3 at R = 12 and 14, case 1 at K = 128. The old R = 12 naive figure
   (12.6 s) predates #575 and may be stale.
3. Fix the new defaults from the probe, targeting about 20 minutes for
   `scripts/run_all.sh mac-cpu`. Working hypothesis: `BENCH_RS` default `6,8,10,12,14`
   for cases 2 and 3, `BENCH_KS` default extended to 128 for case 1. If R = 14 naive
   blows the budget, stop the default at 12 and note the cost of 14 in the README.
   Keep every default arm enabled: comparability at equal budget is the point of the
   suite, so shrinking `BENCH_ALGOS` is not on the table for defaults.
4. Run `scripts/run_all.sh mac-cpu` from a clean tree. All sanity gates must pass.
5. Update the README: pinned rev references, the quoted error and timing numbers in the
   case 2 and 3 descriptions and the cost notes, the environment knob defaults table,
   and the known issue 6 wording that names the pinned rev.
6. Commit everything as one commit: Cargo.toml, Cargo.lock, runner defaults, README,
   `result/mac-cpu`.

## Expectations to verify, not assume

- Case 2 and 3 arm errors should be unchanged by #575. Compare against the superseded
  numbers before discarding them.
- Case 2 error curves sit on the 1e-8 reference floor from R = 6 (known issue 4), so
  the expanded R range adds timing information, not accuracy information. That is
  expected and already documented.
- Input chi saturates around 70 to 80 for the default mixtures, so `BENCH_MAX_BOND`
  (512) should stay slack at R = 14. If chi_in drifts up instead, the fixed output
  budget `chi_in` changes meaning across R and the report note should say so.

## Questions for Hiroshi (next case, direction)

Candidate directions for a fourth case:

- (a) Benchmark TCI or quantics construction itself. All current cases exclude input
  construction from the timed region, but construction performance is live upstream
  work (#575 is a construction fix).
- (b) Tree topologies. treetn supports non-chain topologies, every current case is a
  chain. A tree contraction case would exercise what treetn uniquely offers.
- (c) Cross-language timing against the Julia tensor4all stack on identical instances.
  Half the infrastructure exists (HDF5 export, Julia readback), but a timing harness
  with proper warmup would be new work (AGENTS.md warns about JIT).
- (d) Higher dimension: 3D quantics, fused site dimension 8.

Open questions:

1. Which of (a) through (d) first, or something else entirely?
2. Is a fix planned for the simplett `contract_fit` stub
   (tensor4all-rs#571)? That would enable the missing simplett fit arm in case 2.
3. Is a simplett elementwise product for tensor trains planned (known issue 7)? That
   would give cases 1 and 3 a second engine on the same algorithm.
4. Are machine profiles beyond mac-cpu wanted (Linux, cluster)?
5. Should official sweeps pin the thread count (`RAYON_NUM_THREADS`) instead of
   recording `threads: default`?
6. Case 2's reference floor near 1e-8 comes from the tail outside the box. If higher
   accuracy comparisons are ever wanted, should `BENCH_BOX_L` grow?
