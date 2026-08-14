//! Case 5 runner: patched elementwise product, the larger-N and rtol-controlled
//! version of case 4.
//!
//! Two instance families, `BENCH_FAMILY`, and the default is `aniso`.
//!
//! * `aniso`, the default, is `N` anisotropic narrow spikes: minor width
//!   `BENCH_ANISO_SIGMA` fixed, aspect ratio log-uniform in
//!   `[1, BENCH_ANISO_RHO_MAX]` and orientation uniform in `[0, pi)` drawn per
//!   spike, weights `U[0.5, 1.5]`, centers uniform in `0.9` of the box. The mean
//!   spacing is held at `BENCH_ANISO_SPACING` minor widths, so the box half-width
//!   is `L = s sqrt(N) / 2` and `R` is the smallest bit count whose grid step
//!   `2L / 2^R` resolves the minor width to a quarter. That family is the point of
//!   the case: measured at these settings its global rank grows like `N^0.5`, 45
//!   at `N` = 8, 64 at 64, 88 at 128, 120 at 256, 182 at 512 and 256 at 1024,
//!   where it is exactly the geometric bound of `R` = 9 (a bound-censored point:
//!   the resolution rule moves to `R` = 10 at `N` = 2048 and the rank stays 256
//!   under a bound of 1024). So a global representation keeps paying rank that
//!   grows with `N` while a patched one is held at the cap by construction, and
//!   that growth-rate contest is where patching has something to win.
//!
//!   The isotropic control of the same family is one knob away,
//!   `BENCH_ANISO_RHO_MAX=1`, which draws circular spikes of one common shape.
//!   Measured at the same settings it gives 49, 64, 94, 126, 196 and 256 over the
//!   same `N`, which is the same growth: at this spacing-to-width ratio the rank
//!   comes from the density of narrow spikes and not from the anisotropy. The
//!   aspect ratio and the orientation are still drawn per spike because a family
//!   of one common shape is a pure shift family, which is the degenerate case this
//!   study should not rest on, but the measurement does not credit them with the
//!   growth.
//! * `smooth` is case 4's family, kept reachable unchanged: `N` isotropic
//!   Gaussians of log-uniform inverse width in `[BENCH_ALPHA_LO, BENCH_ALPHA_HI]`
//!   at constant density, box half-width `L = L0 sqrt(N / N0)` and bit count
//!   `R = R0 + round(log2(L / L0))`. It is smooth everywhere, so it has no hard
//!   region for the patching to isolate, and the case-5 verdict on it is that
//!   patching costs rather than saves.
//!
//! Both families hold the ratio of spacing to width fixed as `N` grows, which is
//! what keeps the degeneracy guard passing: the sampled product stays at a
//! constant fraction of `max|f| max|g|` instead of collapsing. The two mixtures of
//! an instance come from the seeds `BENCH_SEED + 1` and `BENCH_SEED + 2`, and the
//! family is recorded in every record's params.
//!
//! Representation. Each input is a `PartitionedTT`: a set of tensor trains over
//! disjoint subdomains, where a subdomain is a set of quantics digits held fixed,
//! so one fixed fused site is one quadrant of the box. A subdomain that does not
//! fit under the per-patch rank cap `BENCH_PATCH_MAX_BOND` is split again, so the
//! cap is honoured everywhere and a hard region of the box costs patches instead
//! of global rank. The product is then formed pair by pair over compatible patches
//! and budgeted once at the end by `truncate_adaptive`. Each patched record also
//! carries the cost breakdown of those two halves, `n_pairs`, `pairs_secs` and
//! `truncate_secs`, since which of them dominates is not guessable from the total.
//!
//! Input construction, `BENCH_PATCH_INPUT`. Two constructions produce that
//! patched input and the default is `norm`: build one global train per input
//! exactly as case 4 does, then hand it to `partitionedtt::add_with_patching`,
//! which truncates each subdomain against its volume share of the global squared
//! budget and splits whatever still exceeds the cap, choosing the split index by
//! `BENCH_PATCH_SPLIT`. No TCI runs in that splitting loop. The alternative,
//! `BENCH_PATCH_INPUT=tci`, is `partitionedtt::adaptiveinterpolate`, which never
//! forms a global train at all: it runs a TCI2 per patch on the function itself
//! and splits a patch whose own TCI does not converge under the cap. That is the
//! construction this case is eventually written for, since it is the one whose
//! cost never passes through a global rank. It used to be blocked by
//! an upstream TCI2 defect that returned site tensors with mismatched bonds, fixed
//! by tensor4all-rs#602 (README known issue 11); it now completes at smooth `N` = 32
//! and 64 and at aniso `N` = 64. `norm` remains the default on the measurement: for
//! the same cap the `tci` path splits far harder, 514 and 622 input patches at
//! smooth `N` = 32 against 6 and 7, so it holds four times the input parameters and
//! returns four times the product parameters at the same accuracy. The two are not
//! the same measurement and `input_path` records which one ran: the `norm` path
//! pays for a global `chi_in` train before it splits, so its `input_build_secs`
//! includes that build, the bridge onto the case's site indices, and the
//! splitting.
//!
//! Control knob. Case 4 pins the output budget at `chi_out <= chi_in` and reads
//! the error that budget buys. That comparison has no meaning here, since a
//! patched representation has no single rank to cap: its size is a parameter
//! count spread over patches. Case 5 therefore fixes the accuracy instead,
//! `BENCH_RTOL` (default 1e-8), and reads the size and the time each arm needs to
//! reach it. Every arm is compared at one and the same `rtol`, and `n_params` is
//! the size metric, since it is the one quantity that means the same thing for a
//! single global train and for a set of patch trains.
//!
//! Both `aci` paths of this case, the global baseline and the per-patch engine,
//! read that `rtol` as an ABSOLUTE pivot budget rather than as the upstream
//! default scale-relative one, since the error this case reports is normalized by
//! the largest sampled magnitude of the whole box and a per-bond or per-patch
//! normalization is the per-region relative tolerance the patched budgeting
//! deliberately refuses. It is recorded as `aci_tolerance`, and README known
//! issue 13 records what the choice costs the size verdict.
//!
//! Arms. `patched_fit_treetn`, `patched_naive` and `patched_aci` are three of the
//! four engines of case 3 run on the projected patch trains instead of on the
//! global trains, so a difference between a patched arm and its global namesake is
//! the patching and not the engine. The fourth, `patched_zipup_treetn`, is
//! excluded from the defaults on cost alone and is one `BENCH_ALGOS` away. The
//! runner also measures the two global arms `fit_treetn` and `aci` at the same
//! `rtol` and with no rank cap binding, so the report can put patched and global
//! side by side at equal accuracy. On the norm path those two arms run on the very
//! trains the patched inputs were split out of, so one global construction is
//! counted once and shared. The two have their own `N` ceilings, since they differ
//! by two orders of magnitude in cost: see `max_n_with_fit_baseline` and
//! `max_n_with_aci_baseline`. Set `BENCH_BASELINES=0` to skip both.
//!
//! Input construction is timed and recorded separately from the product, as
//! `input_build_secs`, for both kinds of arm. The two measurements answer
//! different questions and the inputs are shared by every arm of an instance, so
//! adding them into the product time would count one build many times. A patched
//! arm on the norm path therefore reports the global build plus the splitting,
//! while a global arm on the same instance reports the global build alone.

use std::path::PathBuf;
use std::time::Instant;

use t4a_bench::elementwise::{
    check_product_not_degenerate, elementwise_product, max_rel_error_vs_product, tt_n_params,
    AciTolerance, ElementwiseAlgo, FIT_NFULLSWEEPS,
};
use t4a_bench::gaussian::{to_quantics_fused_tt_field, AnisoMixture2D, Field2D, GaussianMixture2D};
use t4a_bench::harness::time_median;
use t4a_bench::patched::{
    fused_site_indices, max_patch_bond, max_rel_error_patched, parse_input_path,
    parse_patched_engine, parse_split_strategy, patched_elementwise_with_stats, patched_input,
    patched_input_from_global, split_strategy_label, total_params, NormPatchedInputOptions,
    PatchedEngine, PatchedInputOptions, PatchedInputPath, PatchedProductOptions,
};
use t4a_bench::record::{write_record, RunRecord, SCHEMA_VERSION};
use tensor4all_simplett::AbstractTensorTrain;

const CASE: &str = "elementwise_gauss2d_patched";

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Which instance family the sweep runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    /// Case 4's isotropic Gaussians at constant density.
    Smooth,
    /// Anisotropic narrow spikes at constant spacing-to-width ratio.
    Aniso,
}

impl Family {
    fn label(self) -> &'static str {
        match self {
            Family::Smooth => "smooth",
            Family::Aniso => "aniso",
        }
    }

    fn parse(name: &str) -> Option<Self> {
        match name {
            "smooth" => Some(Family::Smooth),
            "aniso" => Some(Family::Aniso),
            _ => None,
        }
    }

    /// Default sweep of the family.
    ///
    /// The smooth family keeps case 4's four points. The aniso family reaches
    /// further because its inputs are far cheaper to build at these settings: its
    /// grid is coarser at equal `N`, `R` = 9 at `N` = 512 against `R` = 15 for the
    /// smooth family, so a global train costs seconds where the smooth one costs
    /// minutes. Measured single-threaded at the pinned revision, the aniso default
    /// sweep runs in about three minutes, of which the `N` = 512 point is two.
    ///
    /// `N` = 1024 is left out on cost alone, and it is an interesting point: there
    /// the global rank reaches 256, exactly the geometric bound of `R` = 9, so the
    /// measured rank is bound-censored at that resolution. It costs 302 s on its own, of
    /// which 242 s is `add_with_patching`, so it doubles the sweep and stays one
    /// `BENCH_NS` away.
    fn default_ns(self) -> &'static str {
        match self {
            Family::Smooth => "8,16,32,64",
            Family::Aniso => "8,16,32,64,128,256,512",
        }
    }

    /// Largest `N` that still runs the global `fit_treetn` baseline.
    ///
    /// The uncapped tolerance-driven global fit is the most expensive arm of the
    /// sweep, and it grows fast in `chi_in`. The ceiling is the last `N` measured
    /// to stay under about two minutes for it, since a baseline that costs more
    /// than everything else together would decide the sweep's cost without adding
    /// a conclusion. On the norm path the global input trains are built anyway,
    /// since the patches are split out of them, so what this skips is the product
    /// and not the construction.
    fn max_n_with_fit_baseline(self) -> usize {
        match self {
            Family::Smooth => 64,
            // Measured single-threaded at the pinned revision, on the aniso
            // family this arm is far cheaper than on the smooth one: 0.18 s at
            // N = 64, 1.5 s at 128, 1.9 s at 256, 12.9 s at 512 and 25.1 s at
            // 1024, at chi_in 64, 88, 120, 182 and 256. So it runs over the whole
            // default sweep and one point beyond it, and 1024 is where the
            // measurement stops rather than where the cost does.
            Family::Aniso => 1024,
        }
    }

    /// Largest `N` that still runs the global `aci` baseline.
    ///
    /// Kept separate because the two baselines are not in the same cost class:
    /// `aci` is interpolation-based and costs a fraction of a second where the fit
    /// costs seconds to minutes (0.18 s at `N` = 1024 on the aniso family against
    /// 25.1 s for the fit), so on that family it runs everywhere and gives the
    /// report a global size to compare against at every point.
    fn max_n_with_aci_baseline(self) -> usize {
        match self {
            Family::Smooth => 64,
            Family::Aniso => usize::MAX,
        }
    }
}

/// Box half-width and bit count for `n` Gaussians at constant density, the same
/// construction as case 4. Duplicated as a call into the case-4 binary is not
/// possible, and kept identical by `box_and_bits_matches_case_four`.
pub fn box_and_bits(n: usize, n0: usize, l0: f64, r0: usize) -> (f64, usize) {
    let l = l0 * (n as f64 / n0 as f64).sqrt();
    let r = r0 as f64 + (l / l0).log2().round();
    (l, r.max(1.0) as usize)
}

/// Box half-width and bit count of the anisotropic spike family.
///
/// The mean spacing between spikes is held at `spacing` minor widths, so `N`
/// spikes at that spacing fill a box of half-width `L = s sqrt(N) / 2` with
/// `s = spacing * sigma`. Holding the spacing-to-width ratio fixed is what keeps
/// the elementwise product non degenerate as `N` grows: two independent draws
/// overlap by a constant fraction at every `N`.
///
/// `R` is the smallest bit count whose grid step resolves the minor width to a
/// quarter, `2L / 2^R <= sigma / 4`, which is the resolution the case needs for
/// the sampled error to mean anything: a spike unresolved by the grid is not a
/// hard function, it is a missing one.
pub fn aniso_box_and_bits(n: usize, sigma: f64, spacing: f64) -> (f64, usize) {
    let l = spacing * sigma * (n as f64).sqrt() / 2.0;
    let r = (8.0 * l / sigma).log2().ceil().max(1.0) as usize;
    (l, r)
}

/// One patched input through the TCI-driven path, with the historical upstream
/// failure spelled out.
///
/// Before the current pin the TCI2 of a single patch could return site tensors
/// whose bonds did not match, which surfaced here as an opaque dimension mismatch
/// from deep inside the construction. tensor4all-rs#602 fixed it (README known
/// issue 11) and this path now runs the sweep, but the diagnosis is kept: it is
/// what tells a reader on an older revision what they are looking at instead of
/// leaving them with an upstream stack trace.
fn build_input_tci<M: Field2D>(
    mix: &M,
    sites: &[tensor4all_core::DynIndex],
    box_l: f64,
    options: PatchedInputOptions,
    n: usize,
) -> anyhow::Result<tensor4all_partitionedtt::PartitionedTT> {
    patched_input(mix, sites, box_l, options).map_err(|e| {
        anyhow::anyhow!(
            "TCI-driven patched input construction failed at N={n}: {e}. A dimension mismatch \
             here is README known issue 11, an upstream TCI2 defect that a single patch of some \
             instances triggered before the pin was moved to b160bb7, independent of \
             BENCH_PATCH_MAX_BOND and BENCH_PATCH_MAX_ITER. It is fixed at the current pin, so \
             on this revision the message means something new and is worth reporting upstream. \
             The default input construction of this case is the norm-driven one, \
             which does not go through that code at all: unset BENCH_PATCH_INPUT to use it."
        )
    })
}

/// Everything the two families share, parsed once from the environment.
struct Config {
    family: Family,
    rtol: f64,
    patch_max_bond: usize,
    input_path: PatchedInputPath,
    split_strategy: tensor4all_partitionedtt::PatchSplitStrategy,
    patch_max_iter: usize,
    product_max_bond: usize,
    global_max_bond: usize,
    runs: usize,
    warmups: usize,
    seed: u64,
    sanity: f64,
    baselines: bool,
    engines: Vec<PatchedEngine>,
    out_dir: PathBuf,
    n_error_samples: usize,
    error_seed: u64,
}

#[allow(clippy::too_many_lines)]
fn main() -> anyhow::Result<()> {
    let family = Family::parse(
        std::env::var("BENCH_FAMILY")
            .unwrap_or_else(|_| "aniso".into())
            .trim(),
    )
    .unwrap_or_else(|| panic!("BENCH_FAMILY must be aniso or smooth"));
    let ns: Vec<usize> = std::env::var("BENCH_NS")
        .unwrap_or_else(|_| family.default_ns().into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    // Smooth family knobs, case 4's.
    let l0: f64 = env_or("BENCH_L0", 6.0);
    let n0: usize = env_or("BENCH_N0", 8);
    let r0: usize = env_or("BENCH_R0", 10);
    let alpha_lo: f64 = env_or("BENCH_ALPHA_LO", 0.5);
    let alpha_hi: f64 = env_or("BENCH_ALPHA_HI", 8.0);
    // Aniso family knobs. The minor width is what the grid has to resolve and the
    // spacing is quoted in minor widths, so these three numbers fix the box, the
    // bit count and the overlap of the product at every N.
    let aniso_sigma: f64 = env_or("BENCH_ANISO_SIGMA", 0.05);
    let aniso_rho_max: f64 = env_or("BENCH_ANISO_RHO_MAX", 8.0);
    let aniso_spacing: f64 = env_or("BENCH_ANISO_SPACING", 3.0);
    // The one accuracy knob of the case: it sets the input TCI tolerance of both
    // representations and the output budget of both products.
    let rtol: f64 = env_or("BENCH_RTOL", 1e-8);
    // Per-patch rank cap. A patch that cannot be brought below it is split, so
    // this is what decides how deep the patch tree goes.
    let patch_max_bond: usize = env_or("BENCH_PATCH_MAX_BOND", 64);
    // Which construction produces the patched inputs. The default is the
    // norm-driven splitting of a global train; `tci` is the adaptiveinterpolate
    // path, which runs at the pinned revision and is not the default because it
    // splits far harder for the same cap (README known issue 11).
    let input_path = parse_input_path(
        std::env::var("BENCH_PATCH_INPUT")
            .unwrap_or_else(|_| "norm".into())
            .trim(),
    )
    .unwrap_or_else(|| panic!("BENCH_PATCH_INPUT must be norm or tci"));
    // How the norm-driven path picks the site to split. The upstream default
    // `gain` forms and budget-truncates the children of every candidate site and
    // keeps the cheapest; `sequential` takes the first unprojected site of the
    // patch order, so the splitting runs strictly coarse to fine. Ignored by the
    // tci path, whose splitting is always sequential in the patch order.
    let split_strategy = parse_split_strategy(
        std::env::var("BENCH_PATCH_SPLIT")
            .unwrap_or_else(|_| "gain".into())
            .trim(),
    )
    .unwrap_or_else(|| panic!("BENCH_PATCH_SPLIT must be gain or sequential"));
    // Half-sweep limit of each patch's TCI run on the tci path. A run that stops
    // at its iteration limit is not converged, so its patch is split whatever its
    // rank was, which is why this knob exists at all: if it bound, the patch tree
    // would be a measurement of it rather than of the rank cap. It does not bind.
    // Measured at N = 8 on the smooth family, raising it from the upstream 20 to
    // 200 left the patch counts and the patch ranks unchanged and cost six times
    // the build time.
    let patch_max_iter: usize = env_or("BENCH_PATCH_MAX_ITER", 20);
    // Rank cap of the per-patch product and of the budgeted output. Left
    // non-binding by default at the exact product rank of two capped patches, so
    // that rtol is the only thing that truncates; the runner fails if it binds.
    let product_max_bond: usize =
        env_or("BENCH_PATCH_OUT_MAX_BOND", patch_max_bond * patch_max_bond);
    // Construction cap of the global input trains, and rank cap of the global
    // baseline product. Non-binding by design, as above. It applies to the global
    // baselines and, on the norm path, to the global trains the patched inputs are
    // split out of, which are one and the same trains.
    let global_max_bond: usize = env_or("BENCH_MAX_BOND", 512);
    // One timed pass per arm, against three in the other cases. This case is the
    // most expensive per point and its arms cost tens of seconds each at the top
    // of the default sweep, so a median of three would triple a sweep that is
    // already the longest in the repository, and it would buy little: an arm here
    // does one deterministic pass of real work over many patch pairs, and the
    // spread of three passes measured at N = 8 is about three percent (2.67, 2.72
    // and 2.77 s for patched_fit_treetn), well under the spread between two
    // constructions of the same instance, whose input TCI is not bit-reproducible:
    // the same arm on a rebuilt instance moved from 2.85e-8 to 3.83e-8 in error.
    // Raise it if you want the median.
    let runs: usize = env_or("BENCH_RUNS", 1);
    let warmups: usize = env_or("BENCH_WARMUPS", 0);
    let seed: u64 = env_or("BENCH_SEED", 0);
    // Nothing here spends a budget it cannot afford, so every arm is expected to
    // land near rtol and the gate only screens order-unity wrongness, as in the
    // other cases.
    let sanity: f64 = env_or("BENCH_SANITY", 1e-2);
    let baselines: usize = env_or("BENCH_BASELINES", 1);
    // `patched_zipup_treetn` is excluded from the default arms, the way case 4
    // excludes `naive`: on the norm path a patch carries a rank near the global
    // one, and a single-pass zip-up of two such patches with no binding output cap
    // has to form the full product bond before it truncates. Measured at N = 8 on
    // the smooth family, where the construction returns one patch, it costs 98 s
    // per pass against 3.5 s for each of the other three arms and returns the same
    // product to every reported digit, so it would dominate the default sweep
    // without adding a conclusion. Run it explicitly through BENCH_ALGOS.
    let engines: Vec<PatchedEngine> = std::env::var("BENCH_ALGOS")
        .unwrap_or_else(|_| "patched_fit_treetn,patched_naive,patched_aci".into())
        .split(',')
        .map(|name| {
            parse_patched_engine(name.trim())
                .unwrap_or_else(|| panic!("unknown patched arm {name}"))
        })
        .collect();
    let out_dir =
        PathBuf::from(std::env::var("OUT_DIR").unwrap_or_else(|_| "result/dev/raw".into()));

    let cfg = Config {
        family,
        rtol,
        patch_max_bond,
        input_path,
        split_strategy,
        patch_max_iter,
        product_max_bond,
        global_max_bond,
        runs,
        warmups,
        seed,
        sanity,
        baselines: baselines != 0,
        engines,
        out_dir,
        // Accuracy check sampling, recorded in every record's params, and the same
        // as case 4's so the two cases report the same kind of number.
        n_error_samples: 128,
        error_seed: seed.wrapping_add(99),
    };

    let mut failures = Vec::new();
    for &n in &ns {
        match family {
            Family::Smooth => {
                let (box_l, r) = box_and_bits(n, n0, l0, r0);
                let f =
                    GaussianMixture2D::random(n, box_l, (alpha_lo, alpha_hi), seed.wrapping_add(1));
                let g =
                    GaussianMixture2D::random(n, box_l, (alpha_lo, alpha_hi), seed.wrapping_add(2));
                run_point(
                    &cfg,
                    n,
                    box_l,
                    r,
                    &f,
                    &g,
                    serde_json::json!({
                        "l0": l0, "n0": n0, "r0": r0,
                        "alpha_range": [alpha_lo, alpha_hi],
                    }),
                    &mut failures,
                )?;
            }
            Family::Aniso => {
                let (box_l, r) = aniso_box_and_bits(n, aniso_sigma, aniso_spacing);
                let f = AnisoMixture2D::random(
                    n,
                    box_l,
                    aniso_sigma,
                    aniso_rho_max,
                    seed.wrapping_add(1),
                );
                let g = AnisoMixture2D::random(
                    n,
                    box_l,
                    aniso_sigma,
                    aniso_rho_max,
                    seed.wrapping_add(2),
                );
                run_point(
                    &cfg,
                    n,
                    box_l,
                    r,
                    &f,
                    &g,
                    serde_json::json!({
                        "aniso_sigma_minor": aniso_sigma,
                        "aniso_rho_max": aniso_rho_max,
                        "aniso_spacing_widths": aniso_spacing,
                    }),
                    &mut failures,
                )?;
            }
        }
    }
    if !failures.is_empty() {
        anyhow::bail!("sanity failures:\n{}", failures.join("\n"));
    }
    Ok(())
}

/// One instance: build both representations, run every arm, write the records.
///
/// Generic over the family, so the patched inputs, the degeneracy guard, the error
/// metric and both global baselines run on one code path for both of them and a
/// difference between two families is the instance and not the harness.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn run_point<M: Field2D>(
    cfg: &Config,
    n: usize,
    box_l: f64,
    r: usize,
    f: &M,
    g: &M,
    family_params: serde_json::Value,
    failures: &mut Vec<String>,
) -> anyhow::Result<()> {
    let rtol = cfg.rtol;
    // Fail fast, before any timing: the box grows with N at fixed density, so this
    // is where too narrow a Gaussian or too sparse a spike field would empty the
    // product and make the relative error meaningless.
    let scales = check_product_not_degenerate(f, g, r, box_l, cfg.n_error_samples, cfg.error_seed)?;

    let sites = fused_site_indices(r);
    let t0 = Instant::now();
    // The global trains of the norm path, kept so the baselines can be measured on
    // the very trains the patched inputs were split out of instead of on a second,
    // identical construction. `None` on the tci path, which builds no global train
    // and leaves that to the baselines.
    let mut global_inputs = None;
    // Rank of those global trains, recorded with the patched arms of the norm
    // path: it is the size the patching starts from, so it belongs in the record
    // even at the N where no global baseline arm runs.
    let mut norm_global_chi = None;
    let (fp, gp) = match cfg.input_path {
        PatchedInputPath::NormDriven => {
            // Case 4's input construction, tolerance and cap, so the patched
            // inputs of this case start from the same trains that case measures.
            // Its own build time is kept separately, since it is what the
            // baselines pay and only a part of what the patched arms pay.
            let t_global = Instant::now();
            let (fa, _step) = to_quantics_fused_tt_field(f, r, box_l, rtol, cfg.global_max_bond)?;
            let (gb, _) = to_quantics_fused_tt_field(g, r, box_l, rtol, cfg.global_max_bond)?;
            let global_build_secs = t_global.elapsed().as_secs_f64();
            let chi_in = fa.rank().max(gb.rank());
            norm_global_chi = Some(chi_in);
            // Case 4's loud failure, and it matters more here: a cap-limited
            // global train is not the tolerance-limited object the patched
            // representation is supposed to be split out of, so everything
            // downstream of it would be a measurement of the cap.
            if chi_in >= cfg.global_max_bond {
                failures.push(format!(
                    "n={n}: the global input chi {chi_in} of the norm-driven patched \
                     construction reached the construction cap \
                     BENCH_MAX_BOND={}, so the train the patches were split out of is \
                     cap-limited rather than tolerance-limited",
                    cfg.global_max_bond
                ));
            }
            // The splitting is relative to the norm of the whole function, not per
            // patch: `add_with_patching` spends rtol^2 ||F||^2 over the patches by
            // volume, so a patch in a near-empty corner of the box is never asked
            // for eight relative digits of a quantity that contributes nothing to
            // the global norm.
            let options = NormPatchedInputOptions {
                rtol,
                max_bond_dim: cfg.patch_max_bond,
                strategy: cfg.split_strategy,
            };
            let fp = patched_input_from_global(&fa, &sites, options)?;
            let gp = patched_input_from_global(&gb, &sites, options)?;
            global_inputs = Some((fa, gb, global_build_secs));
            (fp, gp)
        }
        PatchedInputPath::TciDriven => {
            // The per-patch TCI tolerance is absolute, rtol times the sampled
            // scale of the function, rather than relative per patch, for the same
            // reason: a patch in a near-empty corner has to be allowed to converge
            // at rank one.
            let fp = build_input_tci(
                f,
                &sites,
                box_l,
                PatchedInputOptions {
                    abs_tol: rtol * scales.input_scale_f,
                    max_bond_dim: cfg.patch_max_bond,
                    max_iter: cfg.patch_max_iter,
                    seed: cfg.seed.wrapping_add(1),
                },
                n,
            )?;
            let gp = build_input_tci(
                g,
                &sites,
                box_l,
                PatchedInputOptions {
                    abs_tol: rtol * scales.input_scale_g,
                    max_bond_dim: cfg.patch_max_bond,
                    max_iter: cfg.patch_max_iter,
                    seed: cfg.seed.wrapping_add(2),
                },
                n,
            )?;
            (fp, gp)
        }
    };
    let patched_build_secs = t0.elapsed().as_secs_f64();
    let patched_input_bond = max_patch_bond(&fp).max(max_patch_bond(&gp));
    let patched_input_params = total_params(&fp, &sites)? + total_params(&gp, &sites)?;
    eprintln!(
        "family={} n={n} box_l={box_l:.3} r={r} ref_scale={:.3e} input_path={} split={} \
         patches f={} g={} input_bond={patched_input_bond} \
         input_params={patched_input_params} build={patched_build_secs:.3}s",
        cfg.family.label(),
        scales.ref_scale,
        cfg.input_path.label(),
        split_strategy_label(cfg.split_strategy),
        fp.len(),
        gp.len()
    );

    for &engine in &cfg.engines {
        let arm = engine.arm_name();
        let options = PatchedProductOptions {
            engine,
            // Safely below the budget the result is judged at: the real budgeting
            // is done once at the end by truncate_adaptive, which is the only
            // place that knows the output norm and the patch volumes.
            product_tol: rtol * 1e-2,
            product_max_bond_dim: cfg.product_max_bond,
            rtol,
            max_bond_dim: cfg.product_max_bond,
        };
        let ((h, stats), timing) = time_median(cfg.warmups, cfg.runs, || {
            patched_elementwise_with_stats(&fp, &gp, &sites, options)
                .expect("patched product failed")
        });
        let max_error =
            max_rel_error_patched(&h, &sites, f, g, box_l, cfg.n_error_samples, cfg.error_seed)?;
        let n_params = total_params(&h, &sites)?;
        let out_bond = max_patch_bond(&h);
        let mut params = serde_json::json!({
            "n_gauss": n, "box_l": box_l, "r": r,
            "family": cfg.family.label(),
            "arm_kind": "patched",
            // Which construction produced the patched inputs, and how it chose
            // the split site. The two paths are different measurements, so this
            // is part of what identifies the run.
            "input_path": cfg.input_path.label(),
            "split_strategy": split_strategy_label(cfg.split_strategy),
            // Per-patch cap that decides how deep the patch tree goes.
            "patch_max_bond": cfg.patch_max_bond,
            // Only the tci path runs a TCI per patch, so this is inert on the
            // norm path and recorded for completeness.
            "patch_max_iter": cfg.patch_max_iter,
            // Per-patch product tolerance and cap, both deliberately
            // non-binding: rtol and truncate_adaptive decide the output.
            "product_tol": rtol * 1e-2,
            "product_max_bond": cfg.product_max_bond,
            "n_patches_f": fp.len(), "n_patches_g": gp.len(),
            "input_n_params": patched_input_params,
            // Rank of the global trains the norm path split, null on the tci
            // path, which never builds one.
            "input_global_chi": norm_global_chi,
            // Cost breakdown of the last timed pass: the pair loop against the
            // one final budgeting. Which half dominates changes with the family
            // and with N, so the total alone is not interpretable.
            "n_pairs": stats.n_pairs,
            "pairs_secs": stats.pairs_secs,
            "truncate_secs": stats.truncate_secs,
            "runs": cfg.runs, "warmups": cfg.warmups,
            "n_error_samples": cfg.n_error_samples, "error_seed": cfg.error_seed,
            "error_metric": "max_rel_vs_analytic",
            "ref_scale": scales.ref_scale,
            "input_scale_f": scales.input_scale_f,
            "input_scale_g": scales.input_scale_g,
            "engine": engine.engine(),
            // The per-patch aci engine runs on an absolute budget, as the global
            // baseline does, so the two are asked for the same thing.
            "aci_tolerance": "absolute",
            "fit_nsweeps": FIT_NFULLSWEEPS,
            // The output is a set of patch trains, so there is no single per-site
            // bond profile to report: `output_bond_dims` is left empty and the
            // size is carried by n_params, n_patches and max_patch_bond.
            "output_bond_dims_note": "patched output, see n_params and n_patches",
        });
        merge_params(&mut params, &family_params);
        let rec = RunRecord {
            schema_version: SCHEMA_VERSION,
            case: CASE.into(),
            algorithm: arm.into(),
            params,
            seed: cfg.seed,
            tolerance: rtol,
            wall_time_median_secs: timing.median_secs,
            wall_times_secs: timing.runs_secs,
            max_error,
            input_max_bond_dim: patched_input_bond,
            output_max_bond_dim: out_bond,
            output_bond_dims: Vec::new(),
            n_params: Some(n_params),
            n_patches: Some(h.len()),
            max_patch_bond: Some(out_bond),
            rtol: Some(rtol),
            input_build_secs: Some(patched_build_secs),
        };
        // The family is part of the record name: the two families share their
        // arm names, so a sweep of one would otherwise overwrite the records of
        // the other in a profile that holds both.
        write_record(
            &cfg.out_dir,
            &format!("{CASE}-{}-{arm}-n{n}", cfg.family.label()),
            &rec,
        )?;
        eprintln!(
            "  {arm}: t={:.4}s rel_err={max_error:.2e} patches={} bond={out_bond} \
             params={n_params} pairs={} pairs_t={:.3}s truncate_t={:.3}s",
            timing.median_secs,
            h.len(),
            stats.n_pairs,
            stats.pairs_secs,
            stats.truncate_secs
        );
        if !max_error.is_finite() || max_error > cfg.sanity {
            failures.push(format!(
                "{arm} n={n}: rel err {max_error:.2e} > {:.2e}",
                cfg.sanity
            ));
        }
        // A binding cap would mean the arm was truncated by a rank rather than by
        // rtol, which would break the equal-accuracy comparison.
        if out_bond >= cfg.product_max_bond {
            failures.push(format!(
                "{arm} n={n}: output patch bond {out_bond} reached the cap \
                 BENCH_PATCH_OUT_MAX_BOND={}, so the arm was rank-truncated and is no \
                 longer comparable at equal rtol",
                cfg.product_max_bond
            ));
        }
    }

    let global_arms: Vec<(&str, ElementwiseAlgo)> = if cfg.baselines {
        [
            (
                "fit_treetn",
                ElementwiseAlgo::Fit,
                cfg.family.max_n_with_fit_baseline(),
            ),
            (
                "aci",
                ElementwiseAlgo::Aci,
                cfg.family.max_n_with_aci_baseline(),
            ),
        ]
        .into_iter()
        .filter(|&(_, _, max_n)| n <= max_n)
        .map(|(arm, algo, _)| (arm, algo))
        .collect()
    } else {
        Vec::new()
    };
    if global_arms.is_empty() {
        return Ok(());
    }

    // Global baselines at the same rtol, with no binding rank cap: the tolerance
    // alone decides where they stop, which is what makes them comparable with the
    // patched arms. On the norm path these are the same two trains the patched
    // inputs were split out of, so they are reused rather than built a second
    // time: an identical construction repeated would cost the sweep a second
    // global build without measuring anything the first one did not.
    let (fa, gb, global_build_secs) = match global_inputs {
        Some(built) => built,
        None => {
            let t0 = Instant::now();
            let (fa, _step) = to_quantics_fused_tt_field(f, r, box_l, rtol, cfg.global_max_bond)?;
            let (gb, _) = to_quantics_fused_tt_field(g, r, box_l, rtol, cfg.global_max_bond)?;
            (fa, gb, t0.elapsed().as_secs_f64())
        }
    };
    let global_input_bond = fa.rank().max(gb.rank());
    let global_input_params = tt_n_params(&fa) + tt_n_params(&gb);
    eprintln!(
        "  global inputs: chi_in={global_input_bond} params={global_input_params} \
         build={global_build_secs:.3}s"
    );
    // Already checked above where the norm path built these very trains, so this
    // covers the tci path, whose global trains are built only here.
    if norm_global_chi.is_none() && global_input_bond >= cfg.global_max_bond {
        failures.push(format!(
            "n={n}: global input chi {global_input_bond} reached the construction cap \
             BENCH_MAX_BOND={}, so the baseline is cap-limited rather than \
             tolerance-limited",
            cfg.global_max_bond
        ));
    }
    for (arm, algo) in global_arms {
        let (h, timing) = time_median(cfg.warmups, cfg.runs, || {
            elementwise_product(
                algo,
                &fa,
                &gb,
                rtol,
                cfg.global_max_bond,
                AciTolerance::Absolute,
            )
            .expect("global product failed")
        });
        let max_error =
            max_rel_error_vs_product(&h, f, g, r, box_l, cfg.n_error_samples, cfg.error_seed);
        let n_params = tt_n_params(&h);
        let mut params = serde_json::json!({
            "n_gauss": n, "box_l": box_l, "r": r,
            "family": cfg.family.label(),
            "arm_kind": "global",
            // Which construction the instance ran, as on the patched arms: on the
            // norm path this arm runs on the very trains the patches were split
            // out of, so its input_build_secs is a part of theirs, while on the
            // tci path it builds the only global trains of the point.
            "input_path": cfg.input_path.label(),
            "max_bond": cfg.global_max_bond,
            "contract_max_bond": cfg.global_max_bond,
            "contract_tol": rtol,
            // The aci arm runs on an absolute pivot budget, not on the upstream
            // default scale-relative one. This case measures a GLOBAL relative
            // error, normalized by the largest sampled |f g| of the whole box,
            // and BENCH_RTOL times the sampled scale of the function is exactly
            // that budget. The upstream scale-relative criterion normalizes each
            // bond by its own largest sampled output instead, which is the
            // per-region relative tolerance this case deliberately refuses
            // everywhere else (see the tolerance handling of the patched inputs).
            // Measured, it is not a cosmetic difference: on the smooth family at
            // N = 32 the same arm at the same pin returns chi_out = 138, 147972
            // parameters and 8.4e-9 with an absolute budget, against chi_out = 473
            // capped at BENCH_MAX_BOND, 1.7e6 parameters and 2.7e-1 with the
            // scale-relative one, which fails the sanity gate.
            "aci_tolerance": "absolute",
            "aci_scale_tolerance": false,
            "input_n_params": global_input_params,
            "runs": cfg.runs, "warmups": cfg.warmups,
            "n_error_samples": cfg.n_error_samples, "error_seed": cfg.error_seed,
            "error_metric": "max_rel_vs_analytic",
            "ref_scale": scales.ref_scale,
            "input_scale_f": scales.input_scale_f,
            "input_scale_g": scales.input_scale_g,
            "engine": algo.engine(),
            "fit_nsweeps": FIT_NFULLSWEEPS,
        });
        merge_params(&mut params, &family_params);
        let rec = RunRecord {
            schema_version: SCHEMA_VERSION,
            case: CASE.into(),
            algorithm: arm.into(),
            params,
            seed: cfg.seed,
            tolerance: rtol,
            wall_time_median_secs: timing.median_secs,
            wall_times_secs: timing.runs_secs,
            max_error,
            input_max_bond_dim: global_input_bond,
            output_max_bond_dim: h.rank(),
            output_bond_dims: h.link_dims(),
            n_params: Some(n_params),
            n_patches: None,
            max_patch_bond: None,
            rtol: Some(rtol),
            input_build_secs: Some(global_build_secs),
        };
        // The family is part of the record name: the two families share their
        // arm names, so a sweep of one would otherwise overwrite the records of
        // the other in a profile that holds both.
        write_record(
            &cfg.out_dir,
            &format!("{CASE}-{}-{arm}-n{n}", cfg.family.label()),
            &rec,
        )?;
        eprintln!(
            "  {arm}: t={:.4}s rel_err={max_error:.2e} chi_out={} params={n_params}",
            timing.median_secs,
            h.rank()
        );
        if !max_error.is_finite() || max_error > cfg.sanity {
            failures.push(format!(
                "{arm} n={n}: rel err {max_error:.2e} > {:.2e}",
                cfg.sanity
            ));
        }
        // The same check the patched arms get, and for the same reason: a global
        // baseline that ran into its rank cap was truncated by a rank rather than
        // by rtol, so it is no longer the tolerance-driven object this case puts
        // next to the patched arms at equal accuracy. It is a separate failure
        // from the error gate, since a cap-limited arm can still land under the
        // gate while measuring the cap.
        if h.rank() >= cfg.global_max_bond {
            failures.push(format!(
                "{arm} n={n}: output chi {} reached the cap BENCH_MAX_BOND={}, so the \
                 baseline was rank-truncated and is no longer comparable at equal rtol",
                h.rank(),
                cfg.global_max_bond
            ));
        }
    }
    Ok(())
}

/// Fold the family-specific params into a record's params object.
fn merge_params(into: &mut serde_json::Value, extra: &serde_json::Value) {
    let (Some(target), Some(source)) = (into.as_object_mut(), extra.as_object()) else {
        panic!("params of a record have to be JSON objects");
    };
    for (key, value) in source {
        target.insert(key.clone(), value.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::{aniso_box_and_bits, box_and_bits, Family};

    /// The smooth instance family has to stay case 4's, or the two cases would not
    /// be measuring the same problem at two different controls.
    #[test]
    fn box_and_bits_matches_case_four() {
        let (l0, n0, r0) = (6.0, 8, 10);
        for &n in &[8usize, 16, 32, 64, 128, 256] {
            let (l, r) = box_and_bits(n, n0, l0, r0);
            let expected_l = l0 * (n as f64 / n0 as f64).sqrt();
            assert!((l - expected_l).abs() < 1e-12);
            assert_eq!(r, (r0 as f64 + (l / l0).log2().round()) as usize);
            // Area per Gaussian is exactly constant, by construction.
            let area_per = (2.0 * l).powi(2) / n as f64;
            assert!((area_per / ((2.0 * l0).powi(2) / n0 as f64) - 1.0).abs() < 1e-12);
        }
        assert_eq!(box_and_bits(8, 8, 6.0, 10), (6.0, 10));
    }

    /// The aniso family holds the spacing-to-width ratio fixed and resolves the
    /// minor width to a quarter of a grid step, which are the two properties its
    /// non degenerate product and its meaningful error metric rest on.
    #[test]
    fn aniso_box_and_bits_holds_spacing_and_resolution() {
        let (sigma, spacing) = (0.05, 3.0);
        for &n in &[8usize, 64, 128, 256, 1024, 2048] {
            let (l, r) = aniso_box_and_bits(n, sigma, spacing);
            // Mean spacing, box side over sqrt(N), is exactly spacing * sigma.
            let mean_spacing = 2.0 * l / (n as f64).sqrt();
            assert!((mean_spacing - spacing * sigma).abs() < 1e-12, "n={n}");
            let step = 2.0 * l / (1u64 << r) as f64;
            assert!(step <= sigma / 4.0, "n={n}: step {step} too coarse");
            // And R is the smallest such bit count, so the grid is not oversized.
            let coarser = 2.0 * l / (1u64 << (r - 1)) as f64;
            assert!(coarser > sigma / 4.0, "n={n}: R={r} is larger than needed");
        }
        // The measured chi table of the README quotes R = 9 at N = 1024, where the
        // global rank reaches the geometric bound, and R = 10 at 2048.
        assert_eq!(aniso_box_and_bits(1024, 0.05, 3.0).1, 9);
        assert_eq!(aniso_box_and_bits(2048, 0.05, 3.0).1, 10);
    }

    #[test]
    fn family_parses_both_names_and_nothing_else() {
        assert_eq!(Family::parse("aniso"), Some(Family::Aniso));
        assert_eq!(Family::parse("smooth"), Some(Family::Smooth));
        assert_eq!(Family::parse("gaussian"), None);
    }
}
