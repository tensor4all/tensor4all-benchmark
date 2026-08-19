//! Patched (domain decomposed) elementwise product of quantics tensor trains,
//! built on `tensor4all-partitionedtt`.
//!
//! The global arms of `elementwise.rs` all pay for one tensor train that has to
//! resolve the function everywhere at once, so their rank is set by the hardest
//! region of the domain. The patched arms instead represent each input as a
//! [`PartitionedTT`]: a set of tensor trains over disjoint subdomains, each one
//! obtained by fixing quantics digits (one fused site fixed is one quadrant of
//! the box, the next one a sub-quadrant, and so on). A subdomain that cannot be
//! represented below a per-patch rank cap is split further, so the rank cap is
//! honoured everywhere and the price of a hard region is paid in patch count
//! rather than in global rank.
//!
//! There are two constructions of that patched input here, and they differ only
//! in what decides a split.
//!
//! * [`patched_input_from_global`], the default of case 5, builds one global
//!   train first and hands it to `partitionedtt::add_with_patching`, which
//!   truncates each subdomain against its volume share of the global squared
//!   budget and splits whatever still sits above the rank cap. The split index is
//!   chosen by a [`PatchSplitStrategy`], and no TCI runs inside the loop: the
//!   whole decision is made from Frobenius norms of an already-built train.
//! * [`patched_input`] instead runs `partitionedtt::adaptiveinterpolate`, which
//!   never forms a global train at all: it runs a TCI2 per patch on the function
//!   itself and splits a patch whose own TCI does not converge under the cap.
//!   That is the construction the case is eventually written for, since it is the
//!   one whose cost never passes through a global rank. It used to trip a TCI2
//!   defect on a patch of some instances, fixed upstream and included at the
//!   pinned revision (README known issue 11); it is not the default because for
//!   the same cap it splits far harder and returns a much larger representation.
//!
//! The product is then formed patch pair by patch pair. Two patches contribute
//! only if their projectors are compatible, in which case the product lives on
//! the intersection of the two subdomains. Both partitions cover the domain
//! disjointly, so the set of pairwise intersections is again a disjoint cover
//! and no tensor train addition is ever needed: each output patch comes from
//! exactly one input pair.
//!
//! Three things about this module are worth stating explicitly, because they are
//! design decisions rather than consequences of the upstream API.
//!
//! 1. The per-patch product runs on the FREE sites only. The sites fixed by the
//!    merged projector are sliced out of both inputs first (which is exactly the
//!    projection, since a projected site of an `adaptiveinterpolate` patch train
//!    carries a one-hot core), the product is formed over the remaining sites,
//!    and the fixed sites are put back afterwards as one-hot cores. This keeps
//!    the work proportional to the patch volume instead of the box volume, and
//!    it is what makes the `aci` engine usable at all: on the embedded train the
//!    product is zero outside a `4^-k` fraction of the index space, so a pivot
//!    search seeded at random points would see nothing but zeros.
//! 2. The tolerance handed to the per-patch engine is deliberately tighter than
//!    the tolerance the result is judged at. The real budgeting is done once, at
//!    the end, by [`truncate_adaptive`], which distributes `rtol^2 * ||F||^2`
//!    over the patches proportionally to patch volume and drops patches whose
//!    norm is below their share. That is the correct treatment of shrinking
//!    patch norms: a plain relative tolerance applied per patch would demand the
//!    same relative accuracy from a patch that contributes nothing to the global
//!    norm as from one that carries all of it.
//! 3. A patch pair with at most one free site is multiplied by the `naive`
//!    engine whatever engine was asked for, because a one-site product is a
//!    single elementwise multiplication and the sweeping engines have no sweep
//!    to run. Which engine ran is therefore only meaningful on patches with two
//!    or more free sites, which is every patch of a realistic instance.

use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_partitionedtt::{
    adaptiveinterpolate, add_with_patching, truncate_adaptive, AdaptiveInterpolateOptions,
    MultiIndex, PartitionedTT, PatchSplitStrategy, PatchingOptions, Projector, SubDomainTT,
    TCI2Options, TensorTrain as ItTensorTrain,
};
use tensor4all_simplett::{
    tensor3_from_data, AbstractTensorTrain, SimpleTensorTrain as SimpleTT, Tensor3Ops,
};

use crate::elementwise::{elementwise_product, AciTolerance, ElementwiseAlgo};
use crate::gaussian::{grid_coord, Field2D};
use crate::harness::{index_to_bits, sample_grid_indices};

/// Which engine forms the product inside one patch pair.
///
/// The four variants are the four global arms of case 3, run on the projected
/// patch trains instead of on the global trains, so a difference between a
/// patched arm and its global namesake is the patching and not the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PatchedEngine {
    /// Local bond-Kronecker product plus an SVD sweep, written in this crate.
    Naive,
    /// `tensor4all_treetn::hadamard` with the one-pass zip-up contraction.
    ZipupTreetn,
    /// `tensor4all_treetn::hadamard` with the variational fit contraction.
    FitTreetn,
    /// `tensor4all_aci::elementwise` cross interpolation of the product.
    Aci,
}

impl PatchedEngine {
    /// Arm name as it appears in the records.
    pub fn arm_name(self) -> &'static str {
        match self {
            PatchedEngine::Naive => "patched_naive",
            PatchedEngine::ZipupTreetn => "patched_zipup_treetn",
            PatchedEngine::FitTreetn => "patched_fit_treetn",
            PatchedEngine::Aci => "patched_aci",
        }
    }

    /// Upstream engine that actually runs the per-patch product.
    pub fn engine(self) -> &'static str {
        self.algo().engine()
    }

    fn algo(self) -> ElementwiseAlgo {
        match self {
            PatchedEngine::Naive => ElementwiseAlgo::Naive,
            PatchedEngine::ZipupTreetn => ElementwiseAlgo::Zipup,
            PatchedEngine::FitTreetn => ElementwiseAlgo::Fit,
            PatchedEngine::Aci => ElementwiseAlgo::Aci,
        }
    }
}

/// Parse an arm name back into an engine, for `BENCH_ALGOS`.
pub fn parse_patched_engine(name: &str) -> Option<PatchedEngine> {
    [
        PatchedEngine::Naive,
        PatchedEngine::ZipupTreetn,
        PatchedEngine::FitTreetn,
        PatchedEngine::Aci,
    ]
    .into_iter()
    .find(|engine| engine.arm_name() == name)
}

/// One fused quantics site index per bit, dimension 4, in the layout
/// `gaussian::to_quantics_fused_tt` produces: local index `x_bit + 2 * y_bit`,
/// most significant bit first.
///
/// The two inputs of a product MUST be built over the same vector: projector
/// compatibility is decided by index identity, so patches of `f` and patches of
/// `g` can only be paired when they name the same site objects.
pub fn fused_site_indices(r: usize) -> Vec<DynIndex> {
    (0..r).map(|_| DynIndex::new_dyn(4)).collect()
}

/// Grid coordinates of one fused quantics multi-index.
///
/// Site `n` carries `x_bit + 2 * y_bit` of bit position `n` counted from the
/// most significant, which is the layout of `fused_site_indices`.
fn fused_to_coords(index: &MultiIndex, r: usize, box_l: f64) -> (f64, f64) {
    let mut ix = 0u64;
    let mut iy = 0u64;
    for (n, &fused) in index.iter().enumerate() {
        let shift = r - 1 - n;
        ix |= ((fused & 1) as u64) << shift;
        iy |= (((fused >> 1) & 1) as u64) << shift;
    }
    (grid_coord(ix, r, box_l), grid_coord(iy, r, box_l))
}

/// Options for the patched construction of one input.
#[derive(Clone, Copy, Debug)]
pub struct PatchedInputOptions {
    /// Absolute TCI tolerance for every patch.
    ///
    /// Absolute, not relative: `TCI2Options::normalize_error` is switched off so
    /// that a patch sitting in a near-empty region of the box converges at rank
    /// one instead of being asked for eight relative digits of a quantity that
    /// contributes nothing. The caller sets this to `rtol` times the sampled
    /// scale of the function, so the accuracy is uniform across the box.
    pub abs_tol: f64,
    /// Per-patch rank cap. A patch whose TCI run does not converge below this
    /// cap is split at the next site, which is the stopping rule: split until
    /// the bond dimension fits under the cap.
    pub max_bond_dim: usize,
    /// Half-sweep limit of each patch's TCI run.
    ///
    /// Exposed because a run that stops at its iteration limit is not converged,
    /// so its patch is split whatever its rank was, which would make the patch
    /// tree a measurement of the iteration limit. Measured at the pinned revision
    /// on the case-5 instance at `N` = 8, that is not what happens: raising this
    /// from the upstream 20 to 200 left the patch counts and the patch ranks
    /// unchanged and cost six times the build time, so the splitting there is
    /// driven by the tolerance and the rank cap, and 20 is the right default.
    pub max_iter: usize,
    /// Seed of the deterministic initial-pivot search.
    pub seed: u64,
}

/// Build the patched representation of a 2D Gaussian mixture on `[-L, L)^2`.
///
/// The patch order is the site order, most significant digit first, so the first
/// split fixes a quadrant of the box, the second a sub-quadrant, and so on:
/// coarse to fine, which is what makes a patch a contiguous region of the box.
pub fn patched_input<M: Field2D>(
    mix: &M,
    sites: &[DynIndex],
    box_l: f64,
    options: PatchedInputOptions,
) -> anyhow::Result<PartitionedTT> {
    let r = sites.len();
    let mixture = mix.clone();
    let evaluate = move |index: &MultiIndex| -> f64 {
        let (x, y) = fused_to_coords(index, r, box_l);
        mixture.eval(x, y)
    };
    let tci_options = TCI2Options {
        tolerance: options.abs_tol,
        max_bond_dim: Some(options.max_bond_dim),
        max_iter: options.max_iter,
        normalize_error: false,
        seed: Some(options.seed),
        ..TCI2Options::default()
    };
    let opts = AdaptiveInterpolateOptions {
        tci_options,
        patch_order: sites.to_vec(),
        ..AdaptiveInterpolateOptions::default()
    };
    adaptiveinterpolate::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        evaluate,
        None,
        sites.to_vec(),
        Vec::new(),
        opts,
    )
    .map_err(|e| anyhow::anyhow!("adaptiveinterpolate failed: {e}"))
}

/// Which construction turns a function into a patched input.
///
/// Recorded in every case-5 record as `input_path`, because the two are not the
/// same measurement: one of them pays for a global train before it splits and the
/// other never builds one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PatchedInputPath {
    /// Frobenius-norm-driven splitting of a global train,
    /// [`patched_input_from_global`].
    NormDriven,
    /// TCI2-driven splitting of the function itself, [`patched_input`].
    TciDriven,
}

impl PatchedInputPath {
    /// Name as it appears in `BENCH_PATCH_INPUT` and in the records.
    pub fn label(self) -> &'static str {
        match self {
            PatchedInputPath::NormDriven => "norm",
            PatchedInputPath::TciDriven => "tci",
        }
    }
}

/// Parse `BENCH_PATCH_INPUT`.
pub fn parse_input_path(name: &str) -> Option<PatchedInputPath> {
    [PatchedInputPath::NormDriven, PatchedInputPath::TciDriven]
        .into_iter()
        .find(|path| path.label() == name)
}

/// Parse `BENCH_PATCH_SPLIT` into an upstream split strategy.
///
/// `gain` is `ExactParameterGain`, the upstream default, which forms and
/// budget-truncates the children of every candidate index and keeps the cheapest.
/// `sequential` takes the first unprojected index of the patch order instead, so
/// the splitting runs strictly coarse to fine.
pub fn parse_split_strategy(name: &str) -> Option<PatchSplitStrategy> {
    match name {
        "gain" => Some(PatchSplitStrategy::ExactParameterGain),
        "sequential" => Some(PatchSplitStrategy::Sequential),
        _ => None,
    }
}

/// Name of a split strategy as it appears in `BENCH_PATCH_SPLIT` and in the
/// records.
pub fn split_strategy_label(strategy: PatchSplitStrategy) -> &'static str {
    match strategy {
        PatchSplitStrategy::ExactParameterGain => "gain",
        PatchSplitStrategy::Sequential => "sequential",
    }
}

/// Options for the norm-driven patched construction of one input.
#[derive(Clone, Copy, Debug)]
pub struct NormPatchedInputOptions {
    /// Global relative tolerance of the patched representation.
    ///
    /// Spent as `rtol^2 * ||F||^2` distributed over the patches proportionally to
    /// patch volume, both while the splitting decides whether a patch is over its
    /// cap and in the final truncation. Relative to the norm of the whole
    /// function, so a patch in a near-empty corner of the box is not asked for
    /// digits of a quantity that contributes nothing, which is what the TCI-driven
    /// path needs its absolute tolerance for.
    pub rtol: f64,
    /// Per-patch rank cap. A subdomain whose budget-truncated bond dimension is
    /// still above this is split, which is the stopping rule.
    pub max_bond_dim: usize,
    /// How the split index is chosen among the unprojected sites.
    pub strategy: PatchSplitStrategy,
}

/// Build the patched representation of an already-built global train.
///
/// This is `partitionedtt::add_with_patching` on the single subdomain that is the
/// whole domain: it truncates against volume-proportional budgets, splits every
/// subdomain still above `max_bond_dim` at an index chosen by the strategy, and
/// repeats until the cap holds everywhere, then truncates once more. Nothing here
/// evaluates the function, so the accuracy of the result is the accuracy of the
/// train it was handed, and the cost of the construction is the cost of that
/// train plus the splitting.
///
/// `patch_order` is the site list, most significant fused digit first, so a
/// `Sequential` strategy splits coarse to fine and a patch is then a contiguous
/// region of the box. `ExactParameterGain` may pick any unprojected site of that
/// list instead, so its patches are unions of quadrants at a fixed set of digits
/// rather than single quadrants.
pub fn patched_input_from_global(
    tt: &SimpleTT<f64>,
    sites: &[DynIndex],
    options: NormPatchedInputOptions,
) -> anyhow::Result<PartitionedTT> {
    let bridged = tt_to_patch_train(tt, sites)?;
    let patching = PatchingOptions {
        rtol: options.rtol,
        max_bond_dim: Some(options.max_bond_dim),
        patch_order: sites.to_vec(),
        split_strategy: options.strategy,
    };
    add_with_patching(vec![SubDomainTT::from_tt(bridged)], &patching)
        .map_err(|e| anyhow::anyhow!("add_with_patching failed: {e}"))
}

/// Bridge a global `simplett` train onto the fused site indices of this module.
///
/// The patch machinery decides projector compatibility by index identity, so the
/// bridged train has to carry the very [`DynIndex`] objects the rest of the case
/// uses and not fresh ones of the same dimension. That rules out the generic
/// `treetn` bridge, which mints its own site indices, so the cores are copied
/// across directly: a `simplett` core is already column major in
/// `(left, site, right)`, which is the index order the itensorlike tensor is
/// built in, and the boundary bonds of dimension one are dropped rather than
/// carried, exactly as [`embed_free`] does it.
pub fn tt_to_patch_train(tt: &SimpleTT<f64>, sites: &[DynIndex]) -> anyhow::Result<ItTensorTrain> {
    anyhow::ensure!(
        tt.len() == sites.len(),
        "bridge: the train has {} sites, expected {}",
        tt.len(),
        sites.len()
    );
    let edges: Vec<DynIndex> = tt
        .link_dims()
        .into_iter()
        .map(DynIndex::new_dyn)
        .collect::<Vec<_>>();
    let mut tensors = Vec::with_capacity(sites.len());
    for (position, site) in sites.iter().enumerate() {
        let core = tt.site_tensor(position);
        anyhow::ensure!(
            core.site_dim() == site.dim,
            "bridge: core at site {position} has site dimension {}, expected {}",
            core.site_dim(),
            site.dim
        );
        let left = position.checked_sub(1).map(|edge| &edges[edge]);
        let right = edges.get(position);
        anyhow::ensure!(
            left.map_or(1, |index| index.dim) == core.left_dim()
                && right.map_or(1, |index| index.dim) == core.right_dim(),
            "bridge: core at site {position} has bonds {} and {}, and a boundary bond of a \
             tensor train has to be 1",
            core.left_dim(),
            core.right_dim()
        );
        let mut order = Vec::with_capacity(3);
        order.extend(left.cloned());
        order.push(site.clone());
        order.extend(right.cloned());
        tensors.push(IdxTensor::from_dense(order, core.to_col_major_vec())?);
    }
    ItTensorTrain::new(tensors)
        .map_err(|e| anyhow::anyhow!("bridge: the copied train is not a valid tensor train: {e}"))
}

/// Options for the patched product.
#[derive(Clone, Copy, Debug)]
pub struct PatchedProductOptions {
    /// Engine that forms the product inside each patch pair.
    pub engine: PatchedEngine,
    /// Tolerance handed to the per-patch engine.
    ///
    /// Kept safely below the tolerance the result is judged at, since the real
    /// budgeting happens once at the end in [`truncate_adaptive`]. For the three
    /// SVD-based engines this is a singular value threshold relative to the
    /// largest singular value of the patch; the `aci` engine runs
    /// [`AciTolerance::Absolute`], which is the same rule as the per-patch input
    /// TCI tolerance and for the same reason: a patch in a near-empty corner of
    /// the box must not be held to a tolerance relative to its own magnitude,
    /// since its share of the global error budget is what matters. Measured on
    /// the smooth family at `N` = 32, the upstream scale-relative criterion left
    /// the `patched_aci` arm at `5.3e-4` while the three other engines returned
    /// `3.5e-8`; on an absolute budget all four agree again.
    pub product_tol: f64,
    /// Rank cap for the per-patch product, before the final budgeting.
    pub product_max_bond_dim: usize,
    /// Global relative tolerance of the output, spent by [`truncate_adaptive`]
    /// as `rtol^2 * ||F||^2` distributed over patches by volume.
    pub rtol: f64,
    /// Rank cap of the output patches.
    pub max_bond_dim: usize,
}

/// Elementwise product of two patched representations.
///
/// Every compatible pair of patches contributes one output patch on the
/// intersection of the two subdomains. Both inputs cover the domain disjointly,
/// so those intersections are disjoint too and each one is produced exactly
/// once: there is no patch to add to another patch. The collected patches are
/// then handed to [`truncate_adaptive`], which computes the output norm and
/// distributes the global budget over them.
pub fn patched_elementwise(
    f: &PartitionedTT,
    g: &PartitionedTT,
    sites: &[DynIndex],
    options: PatchedProductOptions,
) -> anyhow::Result<PartitionedTT> {
    patched_elementwise_with_stats(f, g, sites, options).map(|(result, _stats)| result)
}

/// Where the wall time of one patched product went.
///
/// The two halves of the product are a loop over compatible patch pairs and one
/// final [`truncate_adaptive`], and they scale differently: the pair loop grows
/// with the patch count and with the cube of the patch rank, while the final
/// budgeting has to gauge and truncate every collected patch once. Which half
/// dominates moves with the family and with the engine: measured on the smooth
/// family at `N` = 64, the budgeting was 23.1 s of the 40.1 s of `patched_aci`
/// against 4.7 s of the 36.0 s of `patched_fit_treetn`, while on the aniso family
/// at `N` = 512 the pair loop dominates both. So a case-5 total is not
/// interpretable without this split, and the runner records it in every patched
/// record.
#[derive(Clone, Copy, Debug, Default)]
pub struct PatchedProductStats {
    /// Compatible patch pairs contracted, which is the number of output patches
    /// before the budgeting drops any.
    pub n_pairs: usize,
    /// Summed wall time of the per-pair products, restriction and embedding
    /// included.
    pub pairs_secs: f64,
    /// Wall time of the final [`truncate_adaptive`], the one place the global
    /// budget is spent, together with the disjointness check of the collected
    /// patches that precedes it. So `pairs_secs + truncate_secs` is the whole
    /// product to within the loop bookkeeping.
    pub truncate_secs: f64,
}

/// [`patched_elementwise`] with the cost breakdown of the two halves.
pub fn patched_elementwise_with_stats(
    f: &PartitionedTT,
    g: &PartitionedTT,
    sites: &[DynIndex],
    options: PatchedProductOptions,
) -> anyhow::Result<(PartitionedTT, PatchedProductStats)> {
    // BENCH_PATCH_TRACE=1 prints one line per contracted pair and one for the
    // final truncation, for cost-breakdown sessions; it is not part of any record.
    let trace = std::env::var("BENCH_PATCH_TRACE").is_ok();
    let mut stats = PatchedProductStats::default();
    let mut patches = Vec::new();
    for (proj_f, sub_f) in f.iter() {
        for (proj_g, sub_g) in g.iter() {
            let Some(merged) = proj_f.intersection(proj_g) else {
                continue;
            };
            let pair_t0 = std::time::Instant::now();
            let free: Vec<usize> = (0..sites.len())
                .filter(|&i| !merged.is_projected_at(&sites[i]))
                .collect();
            let a = restrict_to_free(sub_f.data(), sites, &merged, &free)?;
            let b = restrict_to_free(sub_g.data(), sites, &merged, &free)?;
            let product = match (a, b) {
                (Restricted::Scalar(x), Restricted::Scalar(y)) => Restricted::Scalar(x * y),
                (Restricted::Train(ta), Restricted::Train(tb)) => {
                    // A single free site has no sweep to run, so the sweeping
                    // engines have nothing to contribute and the local product
                    // is the whole computation. Documented at the module top.
                    let algo = if ta.len() >= 2 {
                        options.engine.algo()
                    } else {
                        ElementwiseAlgo::Naive
                    };
                    Restricted::Train(elementwise_product(
                        algo,
                        &ta,
                        &tb,
                        options.product_tol,
                        options.product_max_bond_dim,
                        AciTolerance::Absolute,
                    )?)
                }
                _ => anyhow::bail!("restriction of a patch pair disagreed on the free site count"),
            };
            let tt = embed_free(&product, sites, &merged, &free)?;
            stats.pairs_secs += pair_t0.elapsed().as_secs_f64();
            if trace {
                eprintln!(
                    "  pair fixed_f={} fixed_g={} fixed={} bonds=({},{}) t={:.3}s",
                    proj_f.len(),
                    proj_g.len(),
                    merged.len(),
                    sub_f.max_bond_dim(),
                    sub_g.max_bond_dim(),
                    pair_t0.elapsed().as_secs_f64()
                );
            }
            patches.push(SubDomainTT::new(tt, merged)?);
        }
    }
    stats.n_pairs = patches.len();
    let trunc_t0 = std::time::Instant::now();
    let partitioned = PartitionedTT::from_subdomains(patches)
        .map_err(|e| anyhow::anyhow!("pairwise patch intersections were not disjoint: {e}"))?;
    let result = truncate_adaptive(&partitioned, options.rtol, Some(options.max_bond_dim))
        .map_err(|e| anyhow::anyhow!("truncate_adaptive failed: {e}"))?;
    stats.truncate_secs = trunc_t0.elapsed().as_secs_f64();
    if trace {
        eprintln!(
            "  pairs={} truncate_adaptive t={:.3}s kept={}",
            stats.n_pairs,
            stats.truncate_secs,
            result.len()
        );
    }
    Ok((result, stats))
}

/// A patch train restricted to the sites the merged projector leaves free.
enum Restricted {
    /// Every site was fixed, so the patch holds one number.
    Scalar(f64),
    /// The free sites, in site order.
    Train(SimpleTT<f64>),
}

/// Slice a patch train at the values of `projector` and drop the fixed sites.
///
/// This is exactly the projection of the patch onto the merged subdomain: the
/// value at a fixed site is read off rather than kept as a one-hot core, and the
/// resulting matrix is absorbed into the next free core, so the returned train
/// has one core per free site and represents the same function on the subdomain.
fn restrict_to_free(
    tt: &ItTensorTrain,
    sites: &[DynIndex],
    projector: &Projector,
    free: &[usize],
) -> anyhow::Result<Restricted> {
    // `carry` maps the right bond of the last emitted free core to the left bond
    // of the site being visited, stored column major as carry[row + rows * col].
    let mut carry = vec![1.0f64];
    let mut carry_rows = 1usize;
    let mut carry_cols = 1usize;
    let mut cores: Vec<_> = Vec::with_capacity(free.len());

    for (position, site) in sites.iter().enumerate() {
        let (core, l, s, r) = core_as_col_major(tt, position, site)?;
        anyhow::ensure!(
            carry_cols == l,
            "restriction: carried bond {carry_cols} does not match the left bond {l} at site \
             {position}"
        );
        match projector.get(site) {
            Some(value) => {
                // carry <- carry * core[:, value, :]
                let mut next = vec![0.0f64; carry_rows * r];
                for rr in 0..r {
                    for ll in 0..l {
                        let a = core[ll + l * (value + s * rr)];
                        if a == 0.0 {
                            continue;
                        }
                        for row in 0..carry_rows {
                            next[row + carry_rows * rr] += carry[row + carry_rows * ll] * a;
                        }
                    }
                }
                carry = next;
                carry_cols = r;
            }
            None => {
                // Emit a core whose left bond is the carried one.
                let mut data = vec![0.0f64; carry_rows * s * r];
                for rr in 0..r {
                    for si in 0..s {
                        for ll in 0..l {
                            let a = core[ll + l * (si + s * rr)];
                            if a == 0.0 {
                                continue;
                            }
                            for row in 0..carry_rows {
                                data[row + carry_rows * (si + s * rr)] +=
                                    carry[row + carry_rows * ll] * a;
                            }
                        }
                    }
                }
                cores.push(tensor3_from_data(data, carry_rows, s, r)?);
                carry = identity(r);
                carry_rows = r;
                carry_cols = r;
            }
        }
    }
    anyhow::ensure!(
        carry_cols == 1,
        "restriction: the trailing bond of the patch train is {carry_cols}, expected 1"
    );

    if cores.is_empty() {
        anyhow::ensure!(carry_rows == 1, "restriction: a scalar patch kept a bond");
        return Ok(Restricted::Scalar(carry[0]));
    }
    // Trailing fixed sites leave a column vector that belongs to the last core.
    if carry_rows > 1 {
        let last = cores.pop().expect("cores is not empty");
        let (l, s) = (last.left_dim(), last.site_dim());
        anyhow::ensure!(
            last.right_dim() == carry_rows,
            "restriction: trailing factor does not match the last core"
        );
        let mut data = vec![0.0f64; l * s];
        for (rr, &w) in carry.iter().enumerate().take(carry_rows) {
            if w == 0.0 {
                continue;
            }
            for si in 0..s {
                for ll in 0..l {
                    data[ll + l * si] += *last.get3(ll, si, rr) * w;
                }
            }
        }
        cores.push(tensor3_from_data(data, l, s, 1)?);
    }
    Ok(Restricted::Train(SimpleTT::new(cores)?))
}

/// Column-major `(left, site, right)` data of one core of a patch train, with
/// the boundary bonds reported as dimension 1.
fn core_as_col_major(
    tt: &ItTensorTrain,
    position: usize,
    site: &DynIndex,
) -> anyhow::Result<(Vec<f64>, usize, usize, usize)> {
    let tensor = tt
        .tensor(position)
        .map_err(|e| anyhow::anyhow!("patch train has no tensor at site {position}: {e}"))?;
    let left = position.checked_sub(1).and_then(|edge| tt.linkind(edge));
    let right = tt.linkind(position);
    let mut order = Vec::with_capacity(3);
    order.extend(left.clone());
    order.push(site.clone());
    order.extend(right.clone());
    anyhow::ensure!(
        order.len() == tensor.indices().len(),
        "patch core at site {position} has {} indices, expected {}",
        tensor.indices().len(),
        order.len()
    );
    let data = tensor.permute_indices(&order)?.to_vec::<f64>()?;
    Ok((
        data,
        left.map_or(1, |index| index.dim),
        site.dim,
        right.map_or(1, |index| index.dim),
    ))
}

fn identity(n: usize) -> Vec<f64> {
    let mut data = vec![0.0f64; n * n];
    for i in 0..n {
        data[i + n * i] = 1.0;
    }
    data
}

/// Put the fixed sites back, as one-hot cores, around a free-site train.
///
/// A fixed site between two free sites has to carry the bond across, which is
/// what a copy selector does: `delta(site, value)` times the identity on the
/// bond. A fixed site outside the free range sits on a bond of dimension one.
/// This mirrors what `adaptiveinterpolate` produces for its own patches, so the
/// output patches have the same shape as the input patches.
fn embed_free(
    product: &Restricted,
    sites: &[DynIndex],
    projector: &Projector,
    free: &[usize],
) -> anyhow::Result<ItTensorTrain> {
    let n = sites.len();
    let (link_dims, cores): (Vec<usize>, Vec<_>) = match product {
        Restricted::Scalar(_) => (Vec::new(), Vec::new()),
        Restricted::Train(tt) => (tt.link_dims(), tt.site_tensors().to_vec()),
    };
    // Bond dimensions of the embedded train: one per interior edge. An edge with
    // no free site on its left, or with all of them, carries nothing.
    let edges: Vec<DynIndex> = (0..n.saturating_sub(1))
        .map(|edge| {
            let left_free = free.iter().filter(|&&position| position <= edge).count();
            let dim = if left_free == 0 || left_free == free.len() {
                1
            } else {
                link_dims[left_free - 1]
            };
            DynIndex::new_dyn(dim)
        })
        .collect();

    let scalar = match product {
        Restricted::Scalar(value) => *value,
        Restricted::Train(_) => 1.0,
    };
    let mut tensors = Vec::with_capacity(n);
    let mut next_free = 0usize;
    for (position, site) in sites.iter().enumerate() {
        let left = position.checked_sub(1).map(|edge| &edges[edge]);
        let right = edges.get(position);
        if free.get(next_free) == Some(&position) {
            let core = &cores[next_free];
            let mut order = Vec::with_capacity(3);
            order.extend(left.cloned());
            order.push(site.clone());
            order.extend(right.cloned());
            tensors.push(IdxTensor::from_dense(order, core.to_col_major_vec())?);
            next_free += 1;
        } else {
            let value = projector
                .get(site)
                .ok_or_else(|| anyhow::anyhow!("site {position} is neither free nor projected"))?;
            // Only the first tensor carries the scalar of an all-fixed patch.
            let weight = if position == 0 { scalar } else { 1.0 };
            tensors.push(one_hot_tensor(left, site, right, value, weight)?);
        }
    }
    ItTensorTrain::new(tensors)
        .map_err(|e| anyhow::anyhow!("embedding produced an invalid tensor train: {e}"))
}

fn one_hot_tensor(
    left: Option<&DynIndex>,
    site: &DynIndex,
    right: Option<&DynIndex>,
    value: usize,
    scale: f64,
) -> anyhow::Result<IdxTensor> {
    match (left, right) {
        (Some(left), Some(right)) => {
            anyhow::ensure!(
                left.dim == right.dim,
                "a fixed site has to carry the bond across, got {} and {}",
                left.dim,
                right.dim
            );
            Ok(IdxTensor::from_copy_selector(
                left.clone(),
                site.clone(),
                right.clone(),
                value,
                scale,
            )?)
        }
        _ => {
            let mut order = Vec::with_capacity(3);
            order.extend(left.cloned());
            order.push(site.clone());
            order.extend(right.cloned());
            let outer: usize = order
                .iter()
                .filter(|index| index != &site)
                .map(|index| index.dim)
                .product();
            anyhow::ensure!(
                outer == 1,
                "a fixed site at the boundary needs a unit bond, got {outer}"
            );
            let mut data = vec![0.0f64; site.dim];
            data[value] = scale;
            Ok(IdxTensor::from_dense(order, data)?)
        }
    }
}

/// Evaluate a patched tensor train at one fused quantics multi-index.
///
/// Exactly one patch can match, since the patches are disjoint and cover the
/// domain. A point whose patch was dropped by the budgeting evaluates to zero,
/// which is the honest answer: the representation says the function is below its
/// budget there.
pub fn eval_patched(
    partitioned: &PartitionedTT,
    sites: &[DynIndex],
    fused: &[usize],
) -> anyhow::Result<f64> {
    anyhow::ensure!(
        fused.len() == sites.len(),
        "evaluation point has {} sites, expected {}",
        fused.len(),
        sites.len()
    );
    let mut found = None;
    for (projector, subdomain) in partitioned.iter() {
        let matches = sites
            .iter()
            .zip(fused)
            .all(|(site, &value)| projector.get(site).is_none_or(|fixed| fixed == value));
        if matches {
            anyhow::ensure!(
                found.is_none(),
                "two patches claim the same point, so the partition overlaps"
            );
            found = Some(subdomain);
        }
    }
    let Some(subdomain) = found else {
        return Ok(0.0);
    };
    // Fixing every site turns the restriction into an evaluation.
    let full = Projector::from_pairs(
        sites
            .iter()
            .cloned()
            .zip(fused.iter().copied())
            .collect::<Vec<_>>(),
    )?;
    match restrict_to_free(subdomain.data(), sites, &full, &[])? {
        Restricted::Scalar(value) => Ok(value),
        Restricted::Train(_) => anyhow::bail!("a fully fixed restriction kept free sites"),
    }
}

/// Number of stored parameters of a patched representation.
///
/// Only the free sites of a patch are counted. The cores at the fixed sites are
/// one-hot copy selectors: they are structure, not data, and an implementation
/// that stored a patch as a train over its free sites would not hold them at
/// all. Counting them would inflate the patched arms against the global ones.
pub fn total_params(partitioned: &PartitionedTT, sites: &[DynIndex]) -> anyhow::Result<usize> {
    let mut total = 0usize;
    for (projector, subdomain) in partitioned.iter() {
        let tt = subdomain.data();
        for (position, site) in sites.iter().enumerate() {
            if projector.is_projected_at(site) {
                continue;
            }
            let left = position
                .checked_sub(1)
                .and_then(|edge| tt.linkind(edge))
                .map_or(1, |index| index.dim);
            let right = tt.linkind(position).map_or(1, |index| index.dim);
            total += left * site.dim * right;
        }
    }
    Ok(total)
}

/// Largest bond dimension over all patches.
pub fn max_patch_bond(partitioned: &PartitionedTT) -> usize {
    partitioned
        .values()
        .map(|subdomain| subdomain.max_bond_dim())
        .max()
        .unwrap_or(0)
}

/// Case-5 error metric: sampled max relative error of a patched product against
/// the exact pointwise product of the two mixtures.
///
/// Same sampling, same normalization and same `error_metric` name as
/// [`crate::elementwise::max_rel_error_vs_mixture_product`], so a case-5 number
/// is directly comparable with a case-4 one.
#[allow(clippy::too_many_arguments)]
pub fn max_rel_error_patched<A: Field2D, B: Field2D>(
    h: &PartitionedTT,
    sites: &[DynIndex],
    f: &A,
    g: &B,
    box_l: f64,
    n_samples: usize,
    seed: u64,
) -> anyhow::Result<f64> {
    let r = sites.len();
    let xs = sample_grid_indices(r, n_samples, seed);
    let ys = sample_grid_indices(r, n_samples, seed.wrapping_add(1));
    let mut max_abs = 0.0f64;
    let mut max_ref = 0.0f64;
    for (&ix, &iy) in xs.iter().zip(&ys) {
        let x = grid_coord(ix, r, box_l);
        let y = grid_coord(iy, r, box_l);
        let xb = index_to_bits(ix, r);
        let yb = index_to_bits(iy, r);
        let fused: Vec<usize> = (0..r).map(|n| xb[n] + 2 * yb[n]).collect();
        let got = eval_patched(h, sites, &fused)?;
        let want = f.eval(x, y) * g.eval(x, y);
        max_abs = max_abs.max((got - want).abs());
        max_ref = max_ref.max(want.abs());
    }
    Ok(max_abs / max_ref.max(f64::MIN_POSITIVE))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elementwise::mixture_product_scales;
    use crate::gaussian::GaussianMixture2D;

    /// Two small mixtures at R = 6 and a per-patch cap tight enough that the
    /// construction really splits, so the test exercises the patch machinery
    /// rather than a one-patch degenerate case.
    fn setup(
        r: usize,
        chi_p: usize,
    ) -> (
        Vec<DynIndex>,
        GaussianMixture2D,
        GaussianMixture2D,
        PartitionedTT,
        PartitionedTT,
        f64,
        f64,
    ) {
        let box_l = 4.0;
        let f = GaussianMixture2D::random(3, box_l, (0.5, 8.0), 1);
        let g = GaussianMixture2D::random(3, box_l, (0.5, 8.0), 2);
        let sites = fused_site_indices(r);
        let scales = mixture_product_scales(&f, &g, r, box_l, 128, 99);
        let rtol = 1e-8;
        let fp = patched_input(
            &f,
            &sites,
            box_l,
            PatchedInputOptions {
                abs_tol: rtol * scales.input_scale_f,
                max_bond_dim: chi_p,
                max_iter: 200,
                seed: 1,
            },
        )
        .unwrap();
        let gp = patched_input(
            &g,
            &sites,
            box_l,
            PatchedInputOptions {
                abs_tol: rtol * scales.input_scale_g,
                max_bond_dim: chi_p,
                max_iter: 200,
                seed: 2,
            },
        )
        .unwrap();
        (sites, f, g, fp, gp, box_l, rtol)
    }

    /// The same two mixtures through the norm-driven path: one global train per
    /// input at the same tolerance, bridged and split by `add_with_patching`.
    #[allow(clippy::type_complexity)]
    fn setup_norm(
        r: usize,
        chi_p: usize,
    ) -> (
        Vec<DynIndex>,
        GaussianMixture2D,
        GaussianMixture2D,
        PartitionedTT,
        PartitionedTT,
        f64,
        f64,
    ) {
        use crate::gaussian::to_quantics_fused_tt;

        let box_l = 4.0;
        let f = GaussianMixture2D::random(3, box_l, (0.5, 8.0), 1);
        let g = GaussianMixture2D::random(3, box_l, (0.5, 8.0), 2);
        let sites = fused_site_indices(r);
        let rtol = 1e-8;
        let options = NormPatchedInputOptions {
            rtol,
            max_bond_dim: chi_p,
            strategy: PatchSplitStrategy::default(),
        };
        let (fa, _) = to_quantics_fused_tt(&f, r, box_l, rtol, 256).unwrap();
        let (gb, _) = to_quantics_fused_tt(&g, r, box_l, rtol, 256).unwrap();
        let fp = patched_input_from_global(&fa, &sites, options).unwrap();
        let gp = patched_input_from_global(&gb, &sites, options).unwrap();
        (sites, f, g, fp, gp, box_l, rtol)
    }

    fn product_options(engine: PatchedEngine, rtol: f64) -> PatchedProductOptions {
        PatchedProductOptions {
            engine,
            product_tol: rtol * 1e-2,
            product_max_bond_dim: 256,
            rtol,
            max_bond_dim: 256,
        }
    }

    /// All four engines must reproduce the pointwise product to an accuracy
    /// consistent with the tolerance they were built and truncated at.
    ///
    /// The bound is shared: unlike the fixed-budget cases, nothing here forces
    /// an arm to spend a budget it cannot afford, so every engine is expected to
    /// land near the tolerance. It is `1e3 * rtol` for the same reason case 1
    /// uses that shape, the pointwise error of a norm-relative truncation is not
    /// bounded by the tolerance itself.
    ///
    /// The four engines are expected to agree here, and in the case-5 sweep they
    /// agree to every reported digit: at a tolerance nothing binds against, each
    /// one computes the same product and the final budgeting truncates all of
    /// them identically. So this test cannot tell the engines apart, which is why
    /// it checks instead that they really ran, by requiring the patches to have
    /// enough free sites for the per-patch product to be a sweep rather than the
    /// single-site fallback. `elementwise::algorithms_are_distinguishable_under_
    /// forced_truncation` is what pins the engines apart, on the same code path.
    #[test]
    fn all_engines_reproduce_the_pointwise_product() {
        let (sites, f, g, fp, gp, box_l, rtol) = setup(6, 8);
        println!("patches: f {} g {}", fp.len(), gp.len());
        assert!(
            fp.len() > 1 || gp.len() > 1,
            "the cap did not force a split, so this instance tests nothing"
        );
        let deepest = fp
            .projectors()
            .chain(gp.projectors())
            .map(Projector::len)
            .max()
            .unwrap();
        assert!(
            sites.len() - deepest >= 2,
            "the deepest patch leaves {} free sites, so the per-patch products fall back \
             to the single-site path and no engine is exercised",
            sites.len() - deepest
        );
        for engine in [
            PatchedEngine::Naive,
            PatchedEngine::ZipupTreetn,
            PatchedEngine::FitTreetn,
            PatchedEngine::Aci,
        ] {
            let h = patched_elementwise(&fp, &gp, &sites, product_options(engine, rtol)).unwrap();
            let err = max_rel_error_patched(&h, &sites, &f, &g, box_l, 128, 99).unwrap();
            let params = total_params(&h, &sites).unwrap();
            println!(
                "{engine:?}: rel err {err:.3e}, {} patches, max bond {}, params {params}",
                h.len(),
                max_patch_bond(&h)
            );
            assert!(
                err.is_finite() && err < 1e3 * rtol,
                "{engine:?}: rel err {err:e} exceeds {:e}",
                1e3 * rtol
            );
            assert!(params > 0);
            assert!(!h.is_empty());
        }
    }

    /// The default construction of case 5, the norm-driven one, has to reproduce
    /// the pointwise product just as the TCI-driven one does.
    ///
    /// Same instance, same tolerance and same bound as
    /// `all_engines_reproduce_the_pointwise_product`, so the two paths are held to
    /// one standard and the difference between them stays visible in the printed
    /// patch counts rather than in what they are allowed to return. The cap is
    /// tight enough that the splitting really runs, and the free-site check is the
    /// same one: it is what says the per-patch products were sweeps rather than the
    /// single-site fallback, so the engines were exercised. Two engines are run
    /// here, one SVD-based and the interpolating one, which are the two kinds of
    /// per-patch product; the four-engine sweep on this code path is the test
    /// above.
    #[test]
    fn norm_driven_path_reproduces_the_pointwise_product() {
        let (sites, f, g, fp, gp, box_l, rtol) = setup_norm(6, 8);
        println!("patches: f {} g {}", fp.len(), gp.len());
        assert!(
            fp.len() > 1 || gp.len() > 1,
            "the cap did not force a split, so this instance tests nothing"
        );
        let deepest = fp
            .projectors()
            .chain(gp.projectors())
            .map(Projector::len)
            .max()
            .unwrap();
        assert!(
            sites.len() - deepest >= 2,
            "the deepest patch leaves {} free sites, so the per-patch products fall back \
             to the single-site path and no engine is exercised",
            sites.len() - deepest
        );
        for engine in [PatchedEngine::FitTreetn, PatchedEngine::Aci] {
            let h = patched_elementwise(&fp, &gp, &sites, product_options(engine, rtol)).unwrap();
            let err = max_rel_error_patched(&h, &sites, &f, &g, box_l, 128, 99).unwrap();
            let params = total_params(&h, &sites).unwrap();
            println!(
                "{engine:?}: rel err {err:.3e}, {} patches, max bond {}, params {params}",
                h.len(),
                max_patch_bond(&h)
            );
            assert!(
                err.is_finite() && err < 1e3 * rtol,
                "{engine:?}: rel err {err:e} exceeds {:e}",
                1e3 * rtol
            );
            assert!(params > 0);
            assert!(!h.is_empty());
        }
    }

    /// The anisotropic spike family, the default family of case 5, on the default
    /// input construction.
    ///
    /// Small and deliberately well conditioned: four spikes of minor width 0.25 on
    /// a box of half-width 1 at `R` = 6, so the grid resolves a spike to eight
    /// steps and the sampled reference is not a field of zeros, which it would be
    /// if this test copied the sweep's own spacing-to-width ratio at this bit
    /// count. What it checks is the same thing the smooth tests check on the same
    /// code path, that every engine reproduces the pointwise product at the
    /// tolerance, with the cap tight enough that the splitting really runs and the
    /// deepest patch still leaves the per-patch products a sweep to do.
    #[test]
    fn aniso_family_products_reproduce_the_pointwise_product() {
        use crate::gaussian::{to_quantics_fused_tt_field, AnisoMixture2D};

        let (r, box_l, chi_p) = (6usize, 1.0, 8usize);
        let f = AnisoMixture2D::random(4, box_l, 0.25, 8.0, 1);
        let g = AnisoMixture2D::random(4, box_l, 0.25, 8.0, 2);
        let sites = fused_site_indices(r);
        let rtol = 1e-8;
        let options = NormPatchedInputOptions {
            rtol,
            max_bond_dim: chi_p,
            strategy: PatchSplitStrategy::default(),
        };
        let (fa, _) = to_quantics_fused_tt_field(&f, r, box_l, rtol, 256).unwrap();
        let (gb, _) = to_quantics_fused_tt_field(&g, r, box_l, rtol, 256).unwrap();
        let fp = patched_input_from_global(&fa, &sites, options).unwrap();
        let gp = patched_input_from_global(&gb, &sites, options).unwrap();
        println!("patches: f {} g {}", fp.len(), gp.len());
        assert!(
            fp.len() > 1 || gp.len() > 1,
            "the cap did not force a split, so this instance tests nothing"
        );
        let deepest = fp
            .projectors()
            .chain(gp.projectors())
            .map(Projector::len)
            .max()
            .unwrap();
        assert!(
            sites.len() - deepest >= 2,
            "the deepest patch leaves {} free sites, so the per-patch products fall back \
             to the single-site path and no engine is exercised",
            sites.len() - deepest
        );
        for engine in [
            PatchedEngine::Naive,
            PatchedEngine::ZipupTreetn,
            PatchedEngine::FitTreetn,
            PatchedEngine::Aci,
        ] {
            let (h, stats) =
                patched_elementwise_with_stats(&fp, &gp, &sites, product_options(engine, rtol))
                    .unwrap();
            let err = max_rel_error_patched(&h, &sites, &f, &g, box_l, 128, 99).unwrap();
            println!(
                "{engine:?}: rel err {err:.3e}, {} patches, {} pairs, max bond {}",
                h.len(),
                stats.n_pairs,
                max_patch_bond(&h)
            );
            assert!(
                err.is_finite() && err < 1e3 * rtol,
                "{engine:?}: rel err {err:e} exceeds {:e}",
                1e3 * rtol
            );
            // The breakdown has to describe the run that produced this product.
            assert!(stats.n_pairs >= h.len());
            assert!(stats.pairs_secs > 0.0 && stats.truncate_secs > 0.0);
            assert!(total_params(&h, &sites).unwrap() > 0);
        }
    }

    /// The bridge that the norm-driven path goes through must copy the train
    /// rather than change it, and it must land on the site indices this module
    /// hands out: a bridge that minted its own indices would produce patches that
    /// can never be paired with another input's patches, since projector
    /// compatibility is decided by index identity.
    #[test]
    fn the_bridge_preserves_values_and_site_indices() {
        use crate::gaussian::to_quantics_fused_tt;

        let (r, box_l) = (5, 3.0);
        let mix = GaussianMixture2D::random(3, box_l, (0.5, 8.0), 4);
        let sites = fused_site_indices(r);
        let (tt, _) = to_quantics_fused_tt(&mix, r, box_l, 1e-10, 256).unwrap();
        let bridged = tt_to_patch_train(&tt, &sites).unwrap();

        // One subdomain over the whole domain, so a full projector turns the
        // restriction into an evaluation of the bridged train itself.
        let whole = PartitionedTT::from_subdomain(SubDomainTT::from_tt(bridged)).unwrap();
        for &(ix, iy) in &[(0u64, 0u64), (5, 27), (31, 1), (16, 16)] {
            let xb = index_to_bits(ix, r);
            let yb = index_to_bits(iy, r);
            let fused: Vec<usize> = (0..r).map(|n| xb[n] + 2 * yb[n]).collect();
            let got = eval_patched(&whole, &sites, &fused).unwrap();
            let want = tt.evaluate(&fused).unwrap();
            assert!(
                (got - want).abs() < 1e-12 * want.abs().max(1e-12),
                "bridge changed the value at ({ix},{iy}): got {got:e} want {want:e}"
            );
        }
    }

    /// The patched product must agree with the global product of the same
    /// instance, which is the statement that the patching is a representation
    /// change and not a different function.
    #[test]
    fn patched_product_agrees_with_the_global_product() {
        use crate::gaussian::to_quantics_fused_tt;

        let (sites, f, g, fp, gp, box_l, rtol) = setup(6, 8);
        let r = sites.len();
        let h = patched_elementwise(
            &fp,
            &gp,
            &sites,
            product_options(PatchedEngine::FitTreetn, rtol),
        )
        .unwrap();

        let (fa, _) = to_quantics_fused_tt(&f, r, box_l, rtol, 256).unwrap();
        let (gb, _) = to_quantics_fused_tt(&g, r, box_l, rtol, 256).unwrap();
        let global = elementwise_product(
            ElementwiseAlgo::Fit,
            &fa,
            &gb,
            rtol,
            256,
            AciTolerance::Absolute,
        )
        .unwrap();

        let mut worst = 0.0f64;
        let mut scale = 0.0f64;
        for &ix in &sample_grid_indices(r, 64, 5) {
            for &iy in &sample_grid_indices(r, 4, 6) {
                let xb = index_to_bits(ix, r);
                let yb = index_to_bits(iy, r);
                let fused: Vec<usize> = (0..r).map(|n| xb[n] + 2 * yb[n]).collect();
                let a = eval_patched(&h, &sites, &fused).unwrap();
                let b = global.evaluate(&fused).unwrap();
                worst = worst.max((a - b).abs());
                scale = scale.max(b.abs());
            }
        }
        println!("max |patched - global| = {worst:.3e} on a scale of {scale:.3e}");
        assert!(
            worst < 1e-4 * scale.max(1e-12),
            "patched and global disagree"
        );
    }

    /// A patch whose norm is negligible against its volume share of the budget
    /// is dropped, and the representation then evaluates to zero there, while the
    /// patches that carry the function survive.
    ///
    /// The instance is one Gaussian sitting in the lower-left quadrant, squared:
    /// the product is of order one there and around `1e-56` in the opposite
    /// corner, so the budgeting has both kinds of patch to decide about. Both
    /// halves of the behaviour are pinned here, patches disappearing where the
    /// function is negligible and `eval_patched` answering zero exactly there, and
    /// the surviving patches still reproducing the product where it is not.
    ///
    /// The width is chosen so that the Gaussian is still visible to a handful of
    /// random samples over the whole box. A much narrower one is a genuinely
    /// sparse function, and `adaptiveinterpolate` then declares the top patch zero
    /// without sampling further, as its documentation says it will unless it is
    /// given pivots in a nonzero region, which would make this a test of that
    /// behaviour instead.
    #[test]
    fn negligible_patches_are_dropped_and_evaluate_to_zero() {
        let (r, box_l) = (6, 4.0);
        let sites = fused_site_indices(r);
        let corner = GaussianMixture2D {
            weights: vec![1.0],
            alphas: vec![2.0],
            centers: vec![(-box_l / 2.0, -box_l / 2.0)],
        };
        let options = |seed| PatchedInputOptions {
            abs_tol: 1e-8,
            max_bond_dim: 8,
            max_iter: 200,
            seed,
        };
        let fp = patched_input(&corner, &sites, box_l, options(1)).unwrap();
        let gp = patched_input(&corner, &sites, box_l, options(2)).unwrap();

        let h = patched_elementwise(
            &fp,
            &gp,
            &sites,
            product_options(PatchedEngine::ZipupTreetn, 1e-8),
        )
        .unwrap();
        println!(
            "patches: f {} g {}, pairs kept after budgeting {}",
            fp.len(),
            gp.len(),
            h.len()
        );
        assert!(
            h.len() < fp.len() * gp.len(),
            "no patch was dropped from a product that vanishes over most of the box"
        );
        assert!(!h.is_empty(), "every patch was dropped, including the peak");

        // The far corner of the box is where the product is negligible, so its
        // patches are gone and the representation answers exactly zero. That is
        // also the right answer to about machine precision.
        let mut dropped = 0usize;
        for &ix in &sample_grid_indices(r, 32, 11) {
            let far = (1u64 << (r - 1)) | (ix >> 1);
            let xb = index_to_bits(far, r);
            let fused: Vec<usize> = (0..r).map(|n| xb[n] * 3).collect();
            let got = eval_patched(&h, &sites, &fused).unwrap();
            assert!(got.abs() < 1e-12, "expected a dropped patch, got {got:e}");
            if got == 0.0 {
                dropped += 1;
            }
        }
        assert!(dropped > 0, "no sampled point fell in a dropped patch");

        // The peak is in a surviving patch and is still reproduced there.
        let peak = 1u64 << (r - 2);
        let bits = index_to_bits(peak, r);
        let fused: Vec<usize> = (0..r).map(|n| bits[n] * 3).collect();
        let x = grid_coord(peak, r, box_l);
        let want = corner.eval(x, x).powi(2);
        let got = eval_patched(&h, &sites, &fused).unwrap();
        println!("at the peak: {got:.6e} against {want:.6e}");
        assert!(
            want > 1e-3,
            "the chosen point is not the peak of the product"
        );
        assert!((got - want).abs() < 1e-6 * want, "the peak patch was lost");
    }

    /// The fused layout of `patched_input` has to be the layout the rest of the
    /// benchmark uses, or the patched arms would silently interpolate a
    /// transposed function.
    #[test]
    fn fused_coordinates_match_the_benchmark_layout() {
        let (r, box_l) = (5, 3.0);
        for &(ix, iy) in &[(0u64, 0u64), (5, 27), (31, 1), (16, 16)] {
            let xb = index_to_bits(ix, r);
            let yb = index_to_bits(iy, r);
            let fused: Vec<usize> = (0..r).map(|n| xb[n] + 2 * yb[n]).collect();
            let (x, y) = fused_to_coords(&fused, r, box_l);
            assert_eq!((x, y), (grid_coord(ix, r, box_l), grid_coord(iy, r, box_l)));
        }
    }
}
