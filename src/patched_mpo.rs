//! Patched MPO-MPO contraction on the legacy TT and chain TreeTN wrappers.

use tensor4all_core::{DynIndex, SvdTruncationPolicy};
use tensor4all_itensorlike::{ContractOptions, TensorTrain, TruncateOptions};
use tensor4all_partitionedtreetn as tree;
use tensor4all_partitionedtt as tt;
use tensor4all_simplett::mpo::MPO;
use tensor4all_treetn::contraction::ContractionOptions as TreeContractOptions;

use crate::mpo_contract::{mpo_to_tensortrain, tensortrain_to_mpo};

fn itensor_cutoff_policy(rtol: f64) -> SvdTruncationPolicy {
    SvdTruncationPolicy::new(rtol * rtol)
        .with_relative()
        .with_squared_values()
        .with_discarded_tail_sum()
}

/// Prepared copies of one MPO pair for a fair TT versus chain TreeTN comparison.
pub struct PatchedMpoPair {
    x: Vec<DynIndex>,
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

/// Total stored scalar entries across a patched chain TreeTN.
pub fn partitioned_treetn_n_params(output: &tree::PartitionedTreeTN<usize>) -> usize {
    output
        .values()
        .map(|subdomain| {
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
                .sum::<usize>()
        })
        .sum()
}

impl PatchedMpoPair {
    /// Build both patched representations from exactly the same MPO cores and indices.
    pub fn new(
        left: &MPO<f64>,
        right: &MPO<f64>,
        rtol: f64,
        patch_max_bond: usize,
    ) -> anyhow::Result<Self> {
        let (left_train, right_train) = mpo_pair_to_tensortrains(left, right)?;
        Self::from_tensortrains(left_train, right_train, rtol, patch_max_bond)
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
        let left_order = y.iter().chain(&x).cloned().collect::<Vec<_>>();
        let right_order = y.iter().chain(&z).cloned().collect::<Vec<_>>();
        let tt_input_options = tt::PatchingOptions {
            rtol: input_rtol,
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
            rtol: input_rtol,
            max_bond_dim: Some(patch_max_bond),
            patch_order: left_order.clone(),
            split_strategy: tree::PatchSplitStrategy::Sequential,
        };
        let tree_right_input_options = tree::PatchingOptions {
            patch_order: right_order.clone(),
            ..tree_input_options.clone()
        };
        let tt_options = tt::PatchingOptions {
            rtol: contract_rtol,
            max_bond_dim: Some(patch_max_bond),
            patch_order: left_order.clone(),
            split_strategy: tt::PatchSplitStrategy::Sequential,
        };
        let tree_options = tree::PatchingOptions {
            rtol: contract_rtol,
            max_bond_dim: Some(patch_max_bond),
            patch_order: left_order,
            split_strategy: tree::PatchSplitStrategy::Sequential,
        };
        let center = n - 1;
        let tree_left = tree::add_with_patching(
            vec![tree::SubDomainTreeTN::from_treetn(
                left_train.into_treetn(),
            )?],
            &center,
            &tree_input_options,
        )?;
        let tree_right = tree::add_with_patching(
            vec![tree::SubDomainTreeTN::from_treetn(
                right_train.into_treetn(),
            )?],
            &center,
            &tree_right_input_options,
        )?;
        anyhow::ensure!(
            (tree_left.len(), tree_right.len()) == (tt_left.len(), tt_right.len()),
            "TT and TreeTN patch counts differ"
        );
        Ok(Self {
            x,
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
        (self.tt_left.len(), self.tt_right.len())
    }

    /// Largest left and right input patch bond dimensions.
    pub fn input_patch_max_bonds(&self) -> (usize, usize) {
        (
            self.tt_left
                .values()
                .map(|patch| patch.max_bond_dim())
                .max()
                .unwrap_or(0),
            self.tt_right
                .values()
                .map(|patch| patch.max_bond_dim())
                .max()
                .unwrap_or(0),
        )
    }

    /// Total stored parameters in the left and right patched inputs.
    pub fn input_patch_n_params(&self) -> (usize, usize) {
        (
            self.tt_left
                .values()
                .map(|subdomain| tensortrain_n_params(subdomain.data()))
                .sum(),
            self.tt_right
                .values()
                .map(|subdomain| tensortrain_n_params(subdomain.data()))
                .sum(),
        )
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
            rtol,
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
        output_max_bond: usize,
    ) -> anyhow::Result<tree::PartitionedTreeTN<usize>> {
        let output_options = tree::PatchingOptions {
            rtol,
            max_bond_dim: Some(output_max_bond),
            ..self.tree_options.clone()
        };
        let options = TreeContractOptions::fit()
            .with_max_bond_dim(output_max_bond)
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
