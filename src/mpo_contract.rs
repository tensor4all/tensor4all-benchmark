//! Global fit contraction of quantics MPOs and its sampled accuracy metric.

use tensor4all_core::{DynIndex, IdxTensor, IndexLike, SvdTruncationPolicy};
use tensor4all_itensorlike::{ContractOptions, TensorTrain};
use tensor4all_simplett::mpo::{tensor4_from_data, Tensor4Ops, MPO};

use crate::gaussian::{discrete_contraction_aniso_reference, grid_coord, AnisoMixture2D};
use crate::harness::{index_to_bits, sample_grid_indices};
use crate::integrated_gaussian::IntegratedGaussianField;

/// Full variational sweeps used by every fit contraction arm.
pub const FIT_NSWEEPS: usize = 1;

/// Number of stored scalar entries in an MPO.
pub fn mpo_n_params(mpo: &MPO<f64>) -> usize {
    (0..mpo.len())
        .map(|site| {
            let core = mpo.site_tensor(site);
            core.left_dim() * core.site_dim_1() * core.site_dim_2() * core.right_dim()
        })
        .sum()
}

/// Form the exact sitewise direct product of two equally long MPOs.
///
/// At each fused leg, the first operand is the low-position factor:
/// `combined = first + first_dimension * second`.
pub fn direct_product_mpo(
    first: &MPO<f64>,
    second: &MPO<f64>,
    max_bytes: usize,
) -> anyhow::Result<MPO<f64>> {
    anyhow::ensure!(
        first.len() == second.len(),
        "direct-product MPO lengths differ"
    );
    let mut total_entries = 0usize;
    for site in 0..first.len() {
        let a = first.site_tensor(site);
        let b = second.site_tensor(site);
        let entries = [
            a.left_dim(),
            b.left_dim(),
            a.site_dim_1(),
            b.site_dim_1(),
            a.site_dim_2(),
            b.site_dim_2(),
            a.right_dim(),
            b.right_dim(),
        ]
        .into_iter()
        .try_fold(1usize, |product, factor| product.checked_mul(factor))
        .ok_or_else(|| anyhow::anyhow!("direct-product parameter count overflow"))?;
        total_entries = total_entries
            .checked_add(entries)
            .ok_or_else(|| anyhow::anyhow!("direct-product parameter count overflow"))?;
    }
    let bytes = total_entries
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or_else(|| anyhow::anyhow!("direct-product byte count overflow"))?;
    anyhow::ensure!(
        bytes <= max_bytes,
        "direct-product MPO requires {bytes} bytes, exceeding limit {max_bytes}"
    );

    let tensors = (0..first.len())
        .map(|site| {
            let a = first.site_tensor(site);
            let b = second.site_tensor(site);
            let mut data = Vec::with_capacity(
                a.left_dim()
                    * b.left_dim()
                    * a.site_dim_1()
                    * b.site_dim_1()
                    * a.site_dim_2()
                    * b.site_dim_2()
                    * a.right_dim()
                    * b.right_dim(),
            );
            for b_right in 0..b.right_dim() {
                for a_right in 0..a.right_dim() {
                    for b_input in 0..b.site_dim_2() {
                        for a_input in 0..a.site_dim_2() {
                            for b_output in 0..b.site_dim_1() {
                                for a_output in 0..a.site_dim_1() {
                                    for b_left in 0..b.left_dim() {
                                        for a_left in 0..a.left_dim() {
                                            data.push(
                                                *a.get4(a_left, a_output, a_input, a_right)
                                                    * *b.get4(b_left, b_output, b_input, b_right),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            tensor4_from_data(
                data,
                a.left_dim() * b.left_dim(),
                a.site_dim_1() * b.site_dim_1(),
                a.site_dim_2() * b.site_dim_2(),
                a.right_dim() * b.right_dim(),
            )
            .map_err(anyhow::Error::from)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    MPO::new(tensors).map_err(anyhow::Error::from)
}

/// Contract `f(x,y)` and `g(y,z)` using the TreeTN variational fit engine.
pub fn fit_mpo_contract(
    left: &MPO<f64>,
    right: &MPO<f64>,
    rtol: f64,
    max_bond: usize,
) -> anyhow::Result<MPO<f64>> {
    let options = ContractOptions::fit()
        .with_max_bond_dim(max_bond)
        .with_svd_policy(
            SvdTruncationPolicy::new(rtol * rtol)
                .with_relative()
                .with_squared_values()
                .with_discarded_tail_sum(),
        )
        .with_nsweeps(FIT_NSWEEPS);
    contract_via_bridge(left, right, &options)
}

fn contract_via_bridge(
    left: &MPO<f64>,
    right: &MPO<f64>,
    options: &ContractOptions,
) -> anyhow::Result<MPO<f64>> {
    let n = left.len();
    anyhow::ensure!(n == right.len(), "MPO lengths differ");
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
            "contracted site dimension mismatch at {site}"
        );
    }
    let left = mpo_to_tensortrain(left, &x, &y)?;
    let right = mpo_to_tensortrain(right, &y, &z)?;
    let output = left
        .contract(&right, options)
        .map_err(|error| anyhow::anyhow!("fit contraction failed: {error}"))?;
    tensortrain_to_mpo(&output, &x, &z)
}

/// Bridge an MPO into an indexed tensor train using the supplied physical indices.
pub fn mpo_to_tensortrain(
    mpo: &MPO<f64>,
    first: &[DynIndex],
    second: &[DynIndex],
) -> anyhow::Result<TensorTrain> {
    let n = mpo.len();
    anyhow::ensure!(
        first.len() == n && second.len() == n,
        "site-index count mismatch"
    );
    let links: Vec<_> = (0..n.saturating_sub(1))
        .map(|site| DynIndex::new_dyn(mpo.link_dim(site)))
        .collect();
    let mut tensors = Vec::with_capacity(n);
    for site in 0..n {
        let core = mpo.site_tensor(site);
        anyhow::ensure!(
            site > 0 || core.left_dim() == 1,
            "leading MPO bond is not 1"
        );
        anyhow::ensure!(
            site + 1 < n || core.right_dim() == 1,
            "trailing MPO bond is not 1"
        );
        let mut indices = Vec::with_capacity(4);
        if site > 0 {
            indices.push(links[site - 1].clone());
        }
        indices.push(first[site].clone());
        indices.push(second[site].clone());
        if site + 1 < n {
            indices.push(links[site].clone());
        }
        tensors.push(IdxTensor::from_dense(indices, core.to_col_major_vec())?);
    }
    TensorTrain::new(tensors).map_err(|error| anyhow::anyhow!("invalid bridged MPO: {error}"))
}

/// Convert an indexed tensor train back to an MPO in the requested physical order.
pub fn tensortrain_to_mpo(
    tt: &TensorTrain,
    first: &[DynIndex],
    second: &[DynIndex],
) -> anyhow::Result<MPO<f64>> {
    let mut cores = Vec::with_capacity(tt.len());
    for site in 0..tt.len() {
        let tensor = tt.tensor(site)?;
        let left = (site > 0).then(|| tt.linkind(site - 1)).flatten();
        let right = (site + 1 < tt.len()).then(|| tt.linkind(site)).flatten();
        let mut order = Vec::with_capacity(4);
        order.extend(left.clone());
        order.push(first[site].clone());
        order.push(second[site].clone());
        order.extend(right.clone());
        anyhow::ensure!(
            order.len() == tensor.indices().len(),
            "result core index-count mismatch at {site}"
        );
        cores.push(tensor4_from_data(
            tensor.permute_indices(&order)?.to_vec::<f64>()?,
            left.map_or(1, |index| index.dim()),
            first[site].dim(),
            second[site].dim(),
            right.map_or(1, |index| index.dim()),
        )?);
    }
    Ok(MPO::new(cores)?)
}

/// Deterministic sampled relative-L2 error against the finite quantics grid sum.
#[allow(clippy::too_many_arguments)]
pub fn sampled_relative_l2_vs_aniso_grid(
    output: &MPO<f64>,
    grid_step: f64,
    left: &AnisoMixture2D,
    right: &AnisoMixture2D,
    r: usize,
    box_l: f64,
    samples: usize,
    seed: u64,
) -> anyhow::Result<f64> {
    let xs = sample_grid_indices(r, samples, seed);
    let zs = sample_grid_indices(r, samples, seed.wrapping_add(1));
    let mut squared_error = 0.0;
    let mut squared_reference = 0.0;
    for (&ix, &iz) in xs.iter().zip(&zs) {
        let bits: Vec<_> = index_to_bits(ix, r)
            .into_iter()
            .zip(index_to_bits(iz, r))
            .flat_map(|(x, z)| [x, z])
            .collect();
        let expected = discrete_contraction_aniso_reference(
            left,
            right,
            grid_coord(ix, r, box_l),
            grid_coord(iz, r, box_l),
            r,
            box_l,
        );
        let error = output.evaluate(&bits)? * grid_step - expected;
        squared_error += error * error;
        squared_reference += expected * expected;
    }
    anyhow::ensure!(squared_reference > 0.0, "zero contraction reference norm");
    Ok((squared_error / squared_reference).sqrt())
}

/// Relative-L2 and maximum RMS-scaled errors at retained output-Gaussian centers.
pub fn center_errors_vs_integrated_gaussians(
    output: &MPO<f64>,
    grid_step: f64,
    reference: &IntegratedGaussianField,
    r: usize,
    box_l: f64,
    samples: usize,
) -> anyhow::Result<(f64, f64)> {
    anyhow::ensure!(samples > 0, "center sample count must be positive");
    let centers = &reference.mixture().centers;
    anyhow::ensure!(!centers.is_empty(), "integrated reference has no centers");
    let count = samples.min(centers.len());
    let grid_size = 1u64 << r;
    let to_index = |coordinate: f64| {
        (((coordinate + box_l) * grid_size as f64 / (2.0 * box_l)).floor() as i64)
            .clamp(0, grid_size as i64 - 1) as u64
    };
    let mut squared_error = 0.0;
    let mut squared_reference = 0.0;
    let mut maximum_absolute_error = 0.0_f64;
    for sample in 0..count {
        let component = if count == 1 {
            0
        } else {
            sample * (centers.len() - 1) / (count - 1)
        };
        let (cx, cz) = centers[component];
        anyhow::ensure!(
            cx.abs() < box_l && cz.abs() < box_l,
            "retained output-Gaussian center lies outside the padded box"
        );
        let (ix, iz) = (to_index(cx), to_index(cz));
        let bits: Vec<_> = index_to_bits(ix, r)
            .into_iter()
            .zip(index_to_bits(iz, r))
            .flat_map(|(x, z)| [x, z])
            .collect();
        let expected = reference.eval(grid_coord(ix, r, box_l), grid_coord(iz, r, box_l));
        let error = output.evaluate(&bits)? * grid_step - expected;
        squared_error += error * error;
        squared_reference += expected * expected;
        maximum_absolute_error = maximum_absolute_error.max(error.abs());
    }
    anyhow::ensure!(squared_reference > 0.0, "zero center reference norm");
    let rms_reference = (squared_reference / count as f64).sqrt();
    Ok((
        (squared_error / squared_reference).sqrt(),
        maximum_absolute_error / rms_reference,
    ))
}

/// Sampled maximum relative difference between two MPOs.
pub fn max_sampled_mpo_relative_diff(
    left: &MPO<f64>,
    right: &MPO<f64>,
    r: usize,
    samples: usize,
    seed: u64,
) -> anyhow::Result<f64> {
    let xs = sample_grid_indices(r, samples, seed);
    let zs = sample_grid_indices(r, samples, seed.wrapping_add(1));
    let mut difference = 0.0_f64;
    let mut scale = 0.0_f64;
    for (&ix, &iz) in xs.iter().zip(&zs) {
        let bits: Vec<_> = index_to_bits(ix, r)
            .into_iter()
            .zip(index_to_bits(iz, r))
            .flat_map(|(x, z)| [x, z])
            .collect();
        let a = left.evaluate(&bits)?;
        let b = right.evaluate(&bits)?;
        difference = difference.max((a - b).abs());
        scale = scale.max(a.abs()).max(b.abs());
    }
    Ok(difference / scale.max(f64::MIN_POSITIVE))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian::fused_qtt_to_mpo;

    #[test]
    fn fit_contraction_matches_the_grid_reference() {
        let (r, box_l) = (6, 1.0);
        let left_mix = AnisoMixture2D::random(2, 0.8, 0.15, 2.0, 1);
        let right_mix = AnisoMixture2D::random(2, 0.8, 0.15, 2.0, 2);
        let left = fused_qtt_to_mpo(
            &left_mix
                .to_interpolative_qtt(r, box_l, 12, 1e-9, 1e-10)
                .unwrap(),
        )
        .unwrap();
        let right = fused_qtt_to_mpo(
            &right_mix
                .to_interpolative_qtt(r, box_l, 12, 1e-9, 1e-10)
                .unwrap(),
        )
        .unwrap();
        assert!(direct_product_mpo(&left, &right, 0).is_err());
        let direct = direct_product_mpo(&left, &right, usize::MAX).unwrap();
        assert_eq!(
            direct.link_dims(),
            left.link_dims()
                .into_iter()
                .zip(right.link_dims())
                .map(|(a, b)| a * b)
                .collect::<Vec<_>>()
        );
        let first = (0..r)
            .flat_map(|site| [site % 2, (site / 2) % 2])
            .collect::<Vec<_>>();
        let second = (0..r)
            .flat_map(|site| [(site / 2) % 2, (site + 1) % 2])
            .collect::<Vec<_>>();
        let combined = (0..r)
            .flat_map(|site| {
                [
                    first[2 * site] + 2 * second[2 * site],
                    first[2 * site + 1] + 2 * second[2 * site + 1],
                ]
            })
            .collect::<Vec<_>>();
        let expected = left.evaluate(&first).unwrap() * right.evaluate(&second).unwrap();
        assert!((direct.evaluate(&combined).unwrap() - expected).abs() < 1e-12);
        let output = fit_mpo_contract(&left, &right, 1e-8, 128).unwrap();
        let error = sampled_relative_l2_vs_aniso_grid(
            &output,
            2.0 * box_l / (1usize << r) as f64,
            &left_mix,
            &right_mix,
            r,
            box_l,
            64,
            7,
        )
        .unwrap();
        assert!(error < 1e-5, "relative L2 error={error:.3e}");
    }
}
