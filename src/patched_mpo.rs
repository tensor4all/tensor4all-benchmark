//! Patched MPO-MPO contraction on the legacy TT and chain TreeTN wrappers.

use std::collections::HashSet;
use tensor4all_core::{AnyScalar, DynIndex, SvdTruncationPolicy};
use tensor4all_itensorlike::{ContractOptions, TensorTrain, TruncateOptions};
use tensor4all_partitionedtreetn as tree;
use tensor4all_partitionedtt as tt;
use tensor4all_simplett::mpo::MPO;
use tensor4all_treetn::contraction::ContractionOptions as TreeContractOptions;

use crate::mpo_contract::{mpo_to_tensortrain, tensortrain_to_mpo};

/// Convert a whole-chain relative-L2 target to a conservative local sweep tolerance.
///
/// The squared target is divided over the two truncation-plan visits to every
/// chain edge. Zero- and one-node inputs use one effective edge visit pair.
///
/// # Examples
///
/// ```
/// let local = t4a_bench::patched_mpo::local_sweep_rtol(1.0e-6, 16);
/// assert!((local * local * 30.0 - 1.0e-12).abs() < 1.0e-27);
/// ```
pub fn local_sweep_rtol(global_rtol: f64, node_count: usize) -> f64 {
    let edge_visits = 2 * node_count.saturating_sub(1).max(1);
    global_rtol / (edge_visits as f64).sqrt()
}

fn interleave(left: &[DynIndex], right: &[DynIndex]) -> Vec<DynIndex> {
    left.iter()
        .zip(right)
        .flat_map(|(left, right)| [left.clone(), right.clone()])
        .collect()
}

fn regularize_binary_partition(
    partition: &tree::PartitionedTreeTN<usize>,
    axes: &[&[DynIndex]],
) -> anyhow::Result<tree::PartitionedTreeTN<usize>> {
    let depths: Vec<_> = axes
        .iter()
        .map(|axis| {
            partition
                .projectors()
                .map(|projector| {
                    axis.iter()
                        .take_while(|index| projector.is_projected_at(index))
                        .count()
                })
                .max()
                .unwrap_or(0)
        })
        .collect();
    let indices: Vec<_> = axes
        .iter()
        .zip(&depths)
        .flat_map(|(axis, &depth)| axis.iter().take(depth).cloned())
        .collect();
    anyhow::ensure!(
        indices.iter().all(|index| index.dim == 2),
        "balanced regularization requires binary physical indices"
    );
    let count = 1usize
        .checked_shl(indices.len() as u32)
        .ok_or_else(|| anyhow::anyhow!("regular patch count overflow"))?;
    let mut patches = Vec::with_capacity(count);
    let mut covered_sources = HashSet::new();
    for assignment in 0..count {
        let target = tree::Projector::from_pairs(
            indices
                .iter()
                .enumerate()
                .map(|(bit, index)| (index.clone(), (assignment >> bit) & 1)),
        )?;
        let mut sources = partition
            .values()
            .filter(|source| source.projector().is_compatible_with(&target));
        let Some(source) = sources.next() else {
            continue;
        };
        anyhow::ensure!(
            sources.next().is_none(),
            "regular projector has multiple sources"
        );
        covered_sources.insert(source.projector().clone());
        patches.push(
            source
                .project(&target)?
                .ok_or_else(|| anyhow::anyhow!("regular projection unexpectedly vanished"))?,
        );
    }
    anyhow::ensure!(
        covered_sources.len() == partition.len(),
        "regular refinement did not cover every nonzero adaptive source"
    );
    tree::PartitionedTreeTN::from_subdomains(patches).map_err(Into::into)
}

fn itensor_cutoff_policy(rtol: f64) -> SvdTruncationPolicy {
    SvdTruncationPolicy::new(rtol * rtol)
        .with_relative()
        .with_squared_values()
        .with_discarded_tail_sum()
}

/// Input axes eligible for adaptive patching.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MpoPatchLayout {
    /// Interleaved shared/output axes: left y/x and right y/z.
    BalancedXyz,
    /// Shared y indices only; x and z remain unprojected.
    SharedYOnly,
}

impl MpoPatchLayout {
    /// Stable record token.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BalancedXyz => "balanced_xyz",
            Self::SharedYOnly => "shared_y_only",
        }
    }
}

/// Prepared copies of one MPO pair for a fair TT versus chain TreeTN comparison.
pub struct PatchedMpoPair {
    x: Vec<DynIndex>,
    y: Vec<DynIndex>,
    z: Vec<DynIndex>,
    global_left: TensorTrain,
    global_right: TensorTrain,
    tt_left: tt::PartitionedTT,
    tt_right: tt::PartitionedTT,
    tree_left: tree::PartitionedTreeTN<usize>,
    tree_right: tree::PartitionedTreeTN<usize>,
    tt_options: tt::PatchingOptions,
    tree_options: tree::PatchingOptions,
    center: usize,
    input_max_bond: usize,
}

/// One contracted patched representation, summed back to an MPO for validation.
pub struct PatchedMpoOutput {
    /// Summed output MPO.
    pub mpo: MPO<f64>,
    /// Number of output patches before summing.
    pub n_patches: usize,
    /// Largest bond dimension among output patches.
    pub max_patch_bond: usize,
}

/// Convert a compatible MPO pair to tensor trains with one shared site index.
pub fn mpo_pair_to_tensortrains(
    left: &MPO<f64>,
    right: &MPO<f64>,
) -> anyhow::Result<(TensorTrain, TensorTrain)> {
    anyhow::ensure!(!left.is_empty(), "patched MPO input is empty");
    anyhow::ensure!(left.len() == right.len(), "MPO lengths differ");

    let n = left.len();
    let x: Vec<_> = (0..n)
        .map(|site| DynIndex::new_dyn(left.site_dim(site).0))
        .collect();
    let y: Vec<_> = (0..n)
        .map(|site| DynIndex::new_dyn(left.site_dim(site).1))
        .collect();
    let z: Vec<_> = (0..n)
        .map(|site| DynIndex::new_dyn(right.site_dim(site).1))
        .collect();
    for site in 0..n {
        anyhow::ensure!(
            left.site_dim(site).1 == right.site_dim(site).0,
            "contracted site dimensions differ at site {site}"
        );
    }

    Ok((
        mpo_to_tensortrain(left, &x, &y)?,
        mpo_to_tensortrain(right, &y, &z)?,
    ))
}

/// Truncate a tensor train to a relative L2 error tolerance with no rank cap.
pub fn truncate_tensortrain_l2(tt: &mut TensorTrain, rtol: f64) -> anyhow::Result<()> {
    anyhow::ensure!(rtol.is_finite() && rtol >= 0.0, "invalid L2 tolerance");
    let policy = SvdTruncationPolicy::new(rtol * rtol)
        .with_relative()
        .with_squared_values()
        .with_discarded_tail_sum();
    tt.truncate(&TruncateOptions::svd().with_svd_policy(policy))?;
    Ok(())
}

/// Number of stored scalar parameters in a tensor train.
pub fn tensortrain_n_params(tt: &TensorTrain) -> usize {
    tt.tensors()
        .into_iter()
        .map(|tensor| {
            tensor
                .indices()
                .iter()
                .map(|index| index.dim)
                .product::<usize>()
        })
        .sum()
}

fn subdomain_treetn_n_params(subdomain: &tree::SubDomainTreeTN<usize>) -> usize {
    let network = subdomain.data();
    network
        .node_names()
        .into_iter()
        .filter_map(|node| network.node_index(&node))
        .filter_map(|node| network.tensor(node))
        .map(|tensor| {
            tensor
                .indices()
                .iter()
                .map(|index| index.dim)
                .product::<usize>()
        })
        .sum()
}

/// Total stored scalar entries across a patched chain TreeTN.
pub fn partitioned_treetn_n_params(output: &tree::PartitionedTreeTN<usize>) -> usize {
    output.values().map(subdomain_treetn_n_params).sum()
}

impl PatchedMpoPair {
    /// Build both patched representations from exactly the same MPO cores and indices.
    pub fn new(
        left: &MPO<f64>,
        right: &MPO<f64>,
        rtol: f64,
        patch_max_bond: usize,
    ) -> anyhow::Result<Self> {
        Self::new_with_layout(
            left,
            right,
            rtol,
            patch_max_bond,
            MpoPatchLayout::BalancedXyz,
        )
    }

    /// Build patched representations with an explicit eligible-axis layout.
    pub fn new_with_layout(
        left: &MPO<f64>,
        right: &MPO<f64>,
        rtol: f64,
        patch_max_bond: usize,
        layout: MpoPatchLayout,
    ) -> anyhow::Result<Self> {
        let (left_train, right_train) = mpo_pair_to_tensortrains(left, right)?;
        Self::from_tensortrains_with_layout(
            left_train,
            right_train,
            rtol,
            rtol,
            patch_max_bond,
            layout,
        )
    }

    /// Build patched representations from cached global tensor trains.
    pub fn from_tensortrains(
        left_train: TensorTrain,
        right_train: TensorTrain,
        rtol: f64,
        patch_max_bond: usize,
    ) -> anyhow::Result<Self> {
        Self::from_tensortrains_with_input_rtol(left_train, right_train, rtol, rtol, patch_max_bond)
    }

    /// Build patched inputs with a tolerance independent of contraction truncation.
    pub fn from_tensortrains_with_input_rtol(
        left_train: TensorTrain,
        right_train: TensorTrain,
        input_rtol: f64,
        contract_rtol: f64,
        patch_max_bond: usize,
    ) -> anyhow::Result<Self> {
        Self::from_tensortrains_with_layout(
            left_train,
            right_train,
            input_rtol,
            contract_rtol,
            patch_max_bond,
            MpoPatchLayout::BalancedXyz,
        )
    }

    /// Build patched inputs with an explicit eligible-axis layout.
    pub fn from_tensortrains_with_layout(
        left_train: TensorTrain,
        right_train: TensorTrain,
        input_rtol: f64,
        contract_rtol: f64,
        patch_max_bond: usize,
        layout: MpoPatchLayout,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(!left_train.is_empty(), "cached MPO input is empty");
        anyhow::ensure!(
            left_train.len() == right_train.len(),
            "cached MPO lengths differ"
        );
        anyhow::ensure!(patch_max_bond > 0, "patch max bond must be positive");

        let left_sites = left_train.site_indices();
        let right_sites = right_train.site_indices();
        let mut x = Vec::with_capacity(left_train.len());
        let mut y = Vec::with_capacity(left_train.len());
        let mut z = Vec::with_capacity(left_train.len());
        for site in 0..left_train.len() {
            let common: Vec<_> = left_sites[site]
                .iter()
                .filter(|index| right_sites[site].contains(index))
                .cloned()
                .collect();
            anyhow::ensure!(
                common.len() == 1,
                "cached site {site} does not have one shared index"
            );
            let left_unique: Vec<_> = left_sites[site]
                .iter()
                .filter(|index| *index != &common[0])
                .cloned()
                .collect();
            let right_unique: Vec<_> = right_sites[site]
                .iter()
                .filter(|index| *index != &common[0])
                .cloned()
                .collect();
            anyhow::ensure!(
                left_unique.len() == 1 && right_unique.len() == 1,
                "cached site {site} does not have MPO-like site indices"
            );
            x.push(left_unique[0].clone());
            y.push(common[0].clone());
            z.push(right_unique[0].clone());
        }

        let n = left_train.len();
        let input_max_bond = left_train.max_bond_dim().max(right_train.max_bond_dim());
        let (left_order, right_order) = match layout {
            MpoPatchLayout::BalancedXyz => (interleave(&y, &x), interleave(&y, &z)),
            MpoPatchLayout::SharedYOnly => (y.clone(), y.clone()),
        };
        let output_order = interleave(&x, &z);
        // Partition patch cutoffs are local to each SVD. TreeTN's truncation
        // plan visits every edge twice, so distribute the requested squared
        // relative-L2 budget over those visits. Patch probes verify the exact
        // reconstructed-input residual independently.
        let local_input_rtol = local_sweep_rtol(input_rtol, n);
        let local_contract_rtol = local_sweep_rtol(contract_rtol, n);
        let tt_input_options = tt::PatchingOptions {
            rtol: local_input_rtol,
            max_bond_dim: Some(patch_max_bond),
            patch_order: left_order.clone(),
            split_strategy: tt::PatchSplitStrategy::Sequential,
        };
        let tt_right_input_options = tt::PatchingOptions {
            patch_order: right_order.clone(),
            ..tt_input_options.clone()
        };
        let tt_left = tt::add_with_patching(
            vec![tt::SubDomainTT::from_tt(left_train.clone())],
            &tt_input_options,
        )?;
        let tt_right = tt::add_with_patching(
            vec![tt::SubDomainTT::from_tt(right_train.clone())],
            &tt_right_input_options,
        )?;
        let global_left = left_train.clone();
        let global_right = right_train.clone();
        let tree_input_options = tree::PatchingOptions {
            cutoff: local_input_rtol * local_input_rtol,
            max_bond_dim: Some(patch_max_bond),
            patch_order: left_order.clone(),
            split_strategy: tree::PatchSplitStrategy::Sequential,
        };
        let tree_right_input_options = tree::PatchingOptions {
            patch_order: right_order.clone(),
            ..tree_input_options.clone()
        };
        let tt_options = tt::PatchingOptions {
            rtol: local_contract_rtol,
            max_bond_dim: Some(patch_max_bond),
            patch_order: output_order.clone(),
            split_strategy: tt::PatchSplitStrategy::Sequential,
        };
        let tree_options = tree::PatchingOptions {
            cutoff: local_contract_rtol * local_contract_rtol,
            max_bond_dim: None,
            patch_order: output_order,
            split_strategy: tree::PatchSplitStrategy::Sequential,
        };
        let center = n - 1;
        let adaptive_tree_left = tree::add_with_patching(
            vec![tree::SubDomainTreeTN::from_treetn(
                left_train.into_treetn(),
            )?],
            &center,
            &tree_input_options,
        )?;
        let adaptive_tree_right = tree::add_with_patching(
            vec![tree::SubDomainTreeTN::from_treetn(
                right_train.into_treetn(),
            )?],
            &center,
            &tree_right_input_options,
        )?;
        let tree_left = match layout {
            MpoPatchLayout::BalancedXyz => {
                regularize_binary_partition(&adaptive_tree_left, &[&x, &y])?
            }
            MpoPatchLayout::SharedYOnly => regularize_binary_partition(&adaptive_tree_left, &[&y])?,
        };
        let tree_right = match layout {
            MpoPatchLayout::BalancedXyz => {
                regularize_binary_partition(&adaptive_tree_right, &[&y, &z])?
            }
            MpoPatchLayout::SharedYOnly => {
                regularize_binary_partition(&adaptive_tree_right, &[&y])?
            }
        };
        Ok(Self {
            x,
            y,
            z,
            global_left,
            global_right,
            tt_left,
            tt_right,
            tree_left,
            tree_right,
            tt_options,
            tree_options,
            center,
            input_max_bond,
        })
    }

    /// Borrow the global tensor trains for input caching.
    pub fn global_inputs(&self) -> (&TensorTrain, &TensorTrain) {
        (&self.global_left, &self.global_right)
    }

    /// Maximum bond dimension before patching.
    pub fn input_max_bond(&self) -> usize {
        self.input_max_bond
    }

    /// Left and right input patch counts.
    pub fn input_patch_counts(&self) -> (usize, usize) {
        (self.tree_left.len(), self.tree_right.len())
    }

    /// Distinct x, left-y, right-y, and z projector regions.
    pub fn input_axis_patch_counts(&self) -> (usize, usize, usize, usize) {
        let count = |partition: &tree::PartitionedTreeTN<usize>, indices: &[DynIndex]| {
            partition
                .values()
                .map(|patch| patch.projector().filter_indices(indices))
                .collect::<HashSet<_>>()
                .len()
        };
        (
            count(&self.tree_left, &self.x),
            count(&self.tree_left, &self.y),
            count(&self.tree_right, &self.y),
            count(&self.tree_right, &self.z),
        )
    }

    /// Compatible input-pair and distinct x/z output-projector counts.
    pub fn input_contraction_layout_counts(&self) -> (usize, usize) {
        let output_indices: Vec<_> = self.x.iter().chain(&self.z).cloned().collect();
        let mut compatible_pairs = 0usize;
        let mut output_projectors = HashSet::new();
        for (left, _) in self.tree_left.iter() {
            for (right, _) in self.tree_right.iter() {
                if let Some(merged) = left.intersection(right) {
                    compatible_pairs += 1;
                    output_projectors.insert(merged.filter_indices(&output_indices));
                }
            }
        }
        (compatible_pairs, output_projectors.len())
    }

    /// Coarse work proxies summed over actual compatible input pairs.
    ///
    /// Returns `(parameter_product_sum, max_bond_product_sum,
    /// max_bond_product_cubed_sum)`. These are structural proxies, not measured
    /// floating-point operation counts.
    pub fn input_contraction_proxies(&self) -> anyhow::Result<(u64, u64, f64)> {
        let mut parameter_products = 0u64;
        let mut bond_products = 0u64;
        let mut cubed_bond_products = 0.0;
        for (left_projector, left) in self.tree_left.iter() {
            for (right_projector, right) in self.tree_right.iter() {
                if !left_projector.is_compatible_with(right_projector) {
                    continue;
                }
                let parameter_product = subdomain_treetn_n_params(left)
                    .checked_mul(subdomain_treetn_n_params(right))
                    .and_then(|value| u64::try_from(value).ok())
                    .ok_or_else(|| anyhow::anyhow!("parameter-product proxy overflow"))?;
                parameter_products = parameter_products
                    .checked_add(parameter_product)
                    .ok_or_else(|| anyhow::anyhow!("parameter-product proxy sum overflow"))?;
                let bond_product = left
                    .max_bond_dim()
                    .checked_mul(right.max_bond_dim())
                    .and_then(|value| u64::try_from(value).ok())
                    .ok_or_else(|| anyhow::anyhow!("bond-product proxy overflow"))?;
                bond_products = bond_products
                    .checked_add(bond_product)
                    .ok_or_else(|| anyhow::anyhow!("bond-product proxy sum overflow"))?;
                cubed_bond_products += (bond_product as f64).powi(3);
            }
        }
        anyhow::ensure!(cubed_bond_products.is_finite(), "cubed bond proxy overflow");
        Ok((parameter_products, bond_products, cubed_bond_products))
    }

    /// Sorted maximum bond dimensions of every left and right input patch.
    pub fn input_patch_bond_dims(&self) -> (Vec<usize>, Vec<usize>) {
        let sorted = |partition: &tree::PartitionedTreeTN<usize>| {
            let mut values: Vec<_> = partition
                .values()
                .map(|patch| patch.max_bond_dim())
                .collect();
            values.sort_unstable();
            values
        };
        (sorted(&self.tree_left), sorted(&self.tree_right))
    }

    /// Largest left and right input patch bond dimensions.
    pub fn input_patch_max_bonds(&self) -> (usize, usize) {
        (
            self.tree_left
                .values()
                .map(|patch| patch.max_bond_dim())
                .max()
                .unwrap_or(0),
            self.tree_right
                .values()
                .map(|patch| patch.max_bond_dim())
                .max()
                .unwrap_or(0),
        )
    }

    /// Sorted parameter counts of every left and right input patch.
    pub fn input_patch_param_counts(&self) -> (Vec<usize>, Vec<usize>) {
        let sorted = |partition: &tree::PartitionedTreeTN<usize>| {
            let mut values: Vec<_> = partition.values().map(subdomain_treetn_n_params).collect();
            values.sort_unstable();
            values
        };
        (sorted(&self.tree_left), sorted(&self.tree_right))
    }

    /// Total stored parameters in the left and right patched inputs.
    pub fn input_patch_n_params(&self) -> (usize, usize) {
        (
            partitioned_treetn_n_params(&self.tree_left),
            partitioned_treetn_n_params(&self.tree_right),
        )
    }

    /// Exact relative norm errors of the reconstructed patched inputs.
    pub fn input_patch_relative_errors(&self) -> anyhow::Result<(f64, f64)> {
        let relative_error = |global: &TensorTrain,
                              partition: &tree::PartitionedTreeTN<usize>|
         -> anyhow::Result<f64> {
            let reconstructed = TensorTrain::from_treetn(partition.to_treetn()?)?;
            let difference = global.axpby(
                AnyScalar::new_real(1.0),
                &reconstructed,
                AnyScalar::new_real(-1.0),
            )?;
            Ok(difference.norm()? / global.norm()?)
        };
        Ok((
            relative_error(&self.global_left, &self.tree_left)?,
            relative_error(&self.global_right, &self.tree_right)?,
        ))
    }

    /// Run one full-sweep global fit on the prepared unpatched inputs.
    pub fn contract_fit_global(
        &self,
        rtol: f64,
        output_max_bond: usize,
    ) -> anyhow::Result<TensorTrain> {
        let options = ContractOptions::fit()
            .with_max_bond_dim(output_max_bond)
            .with_svd_policy(itensor_cutoff_policy(rtol))
            .with_nsweeps(1);
        self.global_left
            .contract(&self.global_right, &options)
            .map_err(Into::into)
    }

    /// Convert a prepared global result to an MPO outside the timed region.
    pub fn finish_global_output(&self, output: &TensorTrain) -> anyhow::Result<MPO<f64>> {
        tensortrain_to_mpo(output, &self.x, &self.z)
    }

    /// Run one full-sweep fit through `tensor4all-partitionedtt`.
    pub fn contract_fit_tt_partitioned(
        &self,
        rtol: f64,
        output_max_bond: usize,
    ) -> anyhow::Result<tt::PartitionedTT> {
        let output_options = tt::PatchingOptions {
            rtol: local_sweep_rtol(rtol, self.global_left.len()),
            max_bond_dim: Some(output_max_bond),
            ..self.tt_options.clone()
        };
        let options = ContractOptions::fit()
            .with_max_bond_dim(output_max_bond)
            .with_svd_policy(itensor_cutoff_policy(rtol))
            .with_nsweeps(1);
        tt::contract_adaptive(&self.tt_left, &self.tt_right, &options, &output_options)
            .map_err(Into::into)
    }

    /// Sum and convert a legacy partitioned result outside the timed region.
    pub fn finish_tt_output(&self, output: tt::PartitionedTT) -> anyhow::Result<PatchedMpoOutput> {
        let max_patch_bond = output
            .values()
            .map(|patch| patch.max_bond_dim())
            .max()
            .unwrap_or(0);
        let mpo = tensortrain_to_mpo(&output.to_tensor_train()?, &self.x, &self.z)?;
        Ok(PatchedMpoOutput {
            mpo,
            n_patches: output.len(),
            max_patch_bond,
        })
    }

    /// Run and convert one legacy partitioned fit contraction.
    pub fn contract_fit_tt(
        &self,
        rtol: f64,
        output_max_bond: usize,
    ) -> anyhow::Result<PatchedMpoOutput> {
        let output = self.contract_fit_tt_partitioned(rtol, output_max_bond)?;
        self.finish_tt_output(output)
    }

    /// Run one full-sweep fit through `tensor4all-partitionedtreetn` on a chain.
    pub fn contract_fit_treetn_partitioned(
        &self,
        rtol: f64,
        contribution_max_bond: usize,
    ) -> anyhow::Result<tree::PartitionedTreeTN<usize>> {
        let local_rtol = local_sweep_rtol(rtol, self.global_left.len());
        let output_options = tree::PatchingOptions {
            cutoff: local_rtol * local_rtol,
            max_bond_dim: None,
            ..self.tree_options.clone()
        };
        // Balanced x/y/z input projectors already define disjoint x/z output
        // patches. Keep those groups uncapped, exact-add their y contributions,
        // then spend the output cutoff once in the final adaptive truncation.
        let options = TreeContractOptions::fit()
            .with_max_bond_dim(contribution_max_bond)
            .with_svd_policy(itensor_cutoff_policy(rtol))
            .with_nfullsweeps(1);
        tree::contract_adaptive(
            &self.tree_left,
            &self.tree_right,
            &self.center,
            &options,
            &output_options,
        )
        .map_err(Into::into)
    }

    /// Sum and convert a chain TreeTN partitioned result outside the timed region.
    pub fn finish_treetn_output(
        &self,
        output: tree::PartitionedTreeTN<usize>,
    ) -> anyhow::Result<PatchedMpoOutput> {
        let max_patch_bond = output
            .values()
            .map(|patch| patch.max_bond_dim())
            .max()
            .unwrap_or(0);
        let train = TensorTrain::from_treetn(output.to_treetn()?)?;
        let mpo = tensortrain_to_mpo(&train, &self.x, &self.z)?;
        Ok(PatchedMpoOutput {
            mpo,
            n_patches: output.len(),
            max_patch_bond,
        })
    }

    /// Run and convert one chain TreeTN partitioned fit contraction.
    pub fn contract_fit_treetn(
        &self,
        rtol: f64,
        output_max_bond: usize,
    ) -> anyhow::Result<PatchedMpoOutput> {
        let output = self.contract_fit_treetn_partitioned(rtol, output_max_bond)?;
        self.finish_treetn_output(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor4all_core::IdxTensor;
    use tensor4all_treetn::TreeTN;

    #[test]
    fn local_sweep_tolerance_distributes_squared_budget_over_edge_visits() {
        let local = local_sweep_rtol(1.0e-6, 16);
        assert!((local * local * 30.0 - 1.0e-12).abs() < 1.0e-27);
    }

    #[test]
    fn regular_refinement_splits_coarse_sources_without_overlap() -> anyhow::Result<()> {
        let x = DynIndex::new_dyn(2);
        let y = DynIndex::new_dyn(2);
        let network = TreeTN::from_tensors(
            vec![IdxTensor::from_dense(
                vec![x.clone(), y.clone()],
                vec![1.0_f64, 2.0, 3.0, 4.0],
            )?],
            vec![0usize],
        )?;
        let global = tree::SubDomainTreeTN::from_treetn(network)?;
        let coarse = tree::Projector::from_pairs([(x.clone(), 0)])?;
        let fine_zero = tree::Projector::from_pairs([(x.clone(), 1), (y.clone(), 0)])?;
        let fine_one = tree::Projector::from_pairs([(x.clone(), 1), (y.clone(), 1)])?;
        let adaptive = tree::PartitionedTreeTN::from_subdomains(vec![
            global.project(&coarse)?.unwrap(),
            global.project(&fine_zero)?.unwrap(),
            global.project(&fine_one)?.unwrap(),
        ])?;

        let regular = regularize_binary_partition(
            &adaptive,
            &[std::slice::from_ref(&x), std::slice::from_ref(&y)],
        )?;
        assert_eq!(regular.len(), 4);
        for x_value in 0..2 {
            for y_value in 0..2 {
                let projector =
                    tree::Projector::from_pairs([(x.clone(), x_value), (y.clone(), y_value)])?;
                assert!(regular.contains(&projector));
            }
        }
        let reconstructed = regular.to_treetn()?;
        assert_eq!(
            reconstructed
                .tensor(reconstructed.node_index(&0).unwrap())
                .unwrap()
                .to_vec::<f64>()?,
            vec![1.0, 2.0, 3.0, 4.0]
        );
        Ok(())
    }
}
