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
//!    projection, since a projected site carries a one-hot core), the product is
//!    formed over the remaining sites,
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
    add_with_patching, truncate_adaptive, PartitionedTT, PatchSplitStrategy, PatchingOptions,
    Projector, SubDomainTT, TensorTrain as ItTensorTrain,
};
use tensor4all_simplett::{
    tensor3_from_data, AbstractTensorTrain, SimpleTensorTrain as SimpleTT, Tensor3Ops,
};

use crate::elementwise::{elementwise_product, AciTolerance, ElementwiseAlgo};
use crate::gaussian::{grid_coord, Field2D};
use crate::harness::{index_to_bits, sample_grid_indices};

/// Which engine forms the product inside one patch pair.
///
/// The two variants match the global fit and ACI arms of case 2, but run on
/// projected patch trains.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PatchedEngine {
    /// `tensor4all_treetn::hadamard` with the variational fit contraction.
    FitTreetn,
    /// `tensor4all_aci::elementwise` cross interpolation of the product.
    Aci,
}

impl PatchedEngine {
    fn algo(self) -> ElementwiseAlgo {
        match self {
            PatchedEngine::FitTreetn => ElementwiseAlgo::Fit,
            PatchedEngine::Aci => ElementwiseAlgo::Aci,
        }
    }
}

/// One fused quantics site index per bit, dimension 4, in the layout
/// the interpolative Gaussian QTT uses: local index `x_bit + 2 * y_bit`,
/// most significant bit first.
///
/// The two inputs of a product MUST be built over the same vector: projector
/// compatibility is decided by index identity, so patches of `f` and patches of
/// `g` can only be paired when they name the same site objects.
pub fn fused_site_indices(r: usize) -> Vec<DynIndex> {
    (0..r).map(|_| DynIndex::new_dyn(4)).collect()
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
    /// digits of a quantity that contributes nothing.
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
    /// [`AciTolerance::Absolute`] because a patch in a near-empty region must not
    /// be held to a tolerance relative to its own magnitude.
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
/// budgeting has to gauge and truncate every collected patch once. The runner
/// records both parts because either can dominate.
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

/// Sampled maximum relative error of a patched product against the exact product.
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

/// Deterministic sampled relative-L2 error of a patched product.
#[allow(clippy::too_many_arguments)]
pub fn sampled_relative_l2_patched<A: Field2D, B: Field2D>(
    output: &PartitionedTT,
    sites: &[DynIndex],
    left: &A,
    right: &B,
    box_l: f64,
    samples: usize,
    seed: u64,
) -> anyhow::Result<f64> {
    let r = sites.len();
    let xs = sample_grid_indices(r, samples, seed);
    let ys = sample_grid_indices(r, samples, seed.wrapping_add(1));
    let mut squared_error = 0.0;
    let mut squared_reference = 0.0;
    for (&ix, &iy) in xs.iter().zip(&ys) {
        let x = grid_coord(ix, r, box_l);
        let y = grid_coord(iy, r, box_l);
        let fused: Vec<_> = index_to_bits(ix, r)
            .into_iter()
            .zip(index_to_bits(iy, r))
            .map(|(x, y)| x + 2 * y)
            .collect();
        let expected = left.eval(x, y) * right.eval(x, y);
        let error = eval_patched(output, sites, &fused)? - expected;
        squared_error += error * error;
        squared_reference += expected * expected;
    }
    anyhow::ensure!(squared_reference > 0.0, "zero elementwise reference norm");
    Ok((squared_error / squared_reference).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elementwise::{max_rel_error_vs_product, AciTolerance};
    use crate::gaussian::AnisoMixture2D;

    fn fixture() -> (
        Vec<DynIndex>,
        AnisoMixture2D,
        AnisoMixture2D,
        SimpleTT<f64>,
        SimpleTT<f64>,
        PartitionedTT,
        PartitionedTT,
    ) {
        let (r, box_l) = (6, 1.0);
        let left_mix = AnisoMixture2D::random(2, 0.8, 0.15, 2.0, 1);
        let right_mix = AnisoMixture2D::random(2, 0.8, 0.15, 2.0, 2);
        let left = left_mix
            .to_interpolative_qtt(r, box_l, 12, 1e-9, 1e-10)
            .unwrap();
        let right = right_mix
            .to_interpolative_qtt(r, box_l, 12, 1e-9, 1e-10)
            .unwrap();
        let sites = fused_site_indices(r);
        let options = NormPatchedInputOptions {
            rtol: 1e-8,
            max_bond_dim: crate::gaussian_input::PATCH_CAP,
            strategy: PatchSplitStrategy::Sequential,
        };
        let patched_left = patched_input_from_global(&left, &sites, options).unwrap();
        let patched_right = patched_input_from_global(&right, &sites, options).unwrap();
        (
            sites,
            left_mix,
            right_mix,
            left,
            right,
            patched_left,
            patched_right,
        )
    }

    #[test]
    fn bridge_preserves_values_and_site_indices() {
        let (sites, _, _, left, _, _, _) = fixture();
        let bridged = tt_to_patch_train(&left, &sites).unwrap();
        let patched = PartitionedTT::from_subdomains(vec![SubDomainTT::from_tt(bridged)]).unwrap();
        let (_, subdomain) = patched.iter().next().unwrap();
        for sample in sample_grid_indices(left.len(), 32, 9) {
            let bits = index_to_bits(sample, left.len());
            let fused: Vec<_> = bits.into_iter().map(|bit| bit * 3).collect();
            assert!(
                (eval_patched(&patched, &sites, &fused).unwrap() - left.evaluate(&fused).unwrap())
                    .abs()
                    < 1e-10
            );
        }
        for site in &sites {
            assert!(subdomain
                .data()
                .site_indices()
                .iter()
                .flatten()
                .any(|index| index == site));
        }
    }

    #[test]
    fn fit_and_aci_patched_products_match_the_reference() {
        let (sites, left_mix, right_mix, left, right, patched_left, patched_right) = fixture();
        for (engine, global_algo, aci_tolerance) in [
            (
                PatchedEngine::FitTreetn,
                ElementwiseAlgo::Fit,
                AciTolerance::Absolute,
            ),
            (
                PatchedEngine::Aci,
                ElementwiseAlgo::Aci,
                AciTolerance::ScaleRelative,
            ),
        ] {
            let global =
                elementwise_product(global_algo, &left, &right, 1e-8, 256, aci_tolerance).unwrap();
            let patched = patched_elementwise(
                &patched_left,
                &patched_right,
                &sites,
                PatchedProductOptions {
                    engine,
                    product_tol: 1e-10,
                    product_max_bond_dim: 256,
                    rtol: 1e-8,
                    max_bond_dim: crate::gaussian_input::PATCH_CAP,
                },
            )
            .unwrap();
            let global_error =
                max_rel_error_vs_product(&global, &left_mix, &right_mix, 6, 1.0, 64, 17);
            let patched_error =
                max_rel_error_patched(&patched, &sites, &left_mix, &right_mix, 1.0, 64, 17)
                    .unwrap();
            assert!(
                global_error < 1e-4,
                "{engine:?}: global error={global_error:.3e}"
            );
            assert!(
                patched_error < 1e-4,
                "{engine:?}: patched error={patched_error:.3e}"
            );
            assert!(max_patch_bond(&patched) <= crate::gaussian_input::PATCH_CAP);
        }
    }
}
