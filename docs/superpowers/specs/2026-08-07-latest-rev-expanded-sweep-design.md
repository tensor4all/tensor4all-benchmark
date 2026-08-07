# Update to latest tensor4all-rs and expand the default sweeps

Date: 2026-08-07
Status: implemented

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
3. **Default sweeps grow to a roughly 20 minute full run.** Amended later the same day
   after user feedback: instead of regenerating `result/mac-cpu` in place, profiles are
   split per physical machine. `mac-cpu` stays frozen as the maintainer's machine's
   record at the previous pin, and this sweep lands in `mac-m1-8gb`. `run.yaml` gains
   chip and memory fields and drops the hostname for privacy.
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

## Next case and direction, asked and answered

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

Answers, from the maintainer's review of the pull request on 2026-08-07:

1. Fourth case: benchmark TCI or quantics construction first, option (a), since #575
   changes that path directly and every current case excludes construction from the
   timed region.
2. simplett `contract_fit`: tensor4all-rs#571 is still open with no assignee or
   milestone, so keep the simplett fit arm excluded.
3. simplett elementwise product for tensor trains: no public issue or plan exists.
   Track it separately only when it blocks planned work.
4. Keep `mac-m1-8gb` as a machine-specific profile, but do not use its memory-bound
   naive timings as the cross-machine headline.
5. `threads: default` is acceptable for this profile. Pin a numeric thread count for a
   future official cross-machine sweep.
6. Keep `BENCH_BOX_L` at 6. If higher accuracy comparisons are needed, use a
   finite-box analytic reference rather than only enlarging the box at fixed R.
7. Per-machine profiles, the chip and memory fields, and the hostname removal are
   accepted as implemented here.

The branch, pull request and human merge policy proposed alongside this work was
removed from the change: it belongs in a separate change, made as accepted maintainer
policy rather than as a proposal awaiting veto.
