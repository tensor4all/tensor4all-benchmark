//! ITensorMPS.jl-compatible HDF5 export for benchmark inputs.
//!
//! The bridge is `simplett::SimpleTensorTrain` -> `treetn::TreeTN` ->
//! `itensorlike::TensorTrain` -> `tensor4all_hdf5::{save_mps, append_mps}`,
//! which writes the `MPS` v1 schema that ITensorMPS.jl reads.

use tensor4all_core::TensorElement;
use tensor4all_simplett::{SimpleTensorTrain, TTScalar};
use tensor4all_treetn::tensor_train_to_treetn;

/// Write `tt` into `path` under group `name` as an ITensorMPS.jl `MPS`.
///
/// With `append = false` the file is created (or truncated); with
/// `append = true` the group is added to an existing file, which is how
/// several named MPS objects end up in one instance file.
/// Both `f64` (written as `Dense{Float64}`) and `Complex64` (written as
/// `Dense{ComplexF64}`) element types are supported; the bounds are those of
/// `tensor_train_to_treetn` at the pinned upstream rev.
pub fn save_tt_as_mps<T>(
    path: &str,
    name: &str,
    tt: &SimpleTensorTrain<T>,
    append: bool,
) -> anyhow::Result<()>
where
    T: TTScalar + TensorElement + Clone,
{
    let (treetn, _indices) = tensor_train_to_treetn(tt)?;
    let itt = tensor4all_itensorlike::TensorTrain::from_treetn(treetn)?;
    if append {
        tensor4all_hdf5::append_mps(path, name, &itt)?;
    } else {
        tensor4all_hdf5::save_mps(path, name, &itt)?;
    }
    Ok(())
}

/// Load one HDF5 MPS group back into a simple tensor train.
pub fn load_tt_from_mps<T>(path: &str, name: &str) -> anyhow::Result<SimpleTensorTrain<T>>
where
    T: TTScalar + TensorElement + Clone,
{
    let tt = tensor4all_hdf5::load_mps(path, name)?;
    Ok(tensor4all_treetn::treetn_to_tensor_train::<T>(
        tt.into_treetn(),
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourier::FourierSeries;
    use crate::harness::{index_to_bits, sample_grid_indices};
    use num_complex::Complex64;
    use tensor4all_simplett::AbstractTensorTrain;

    /// The real fused-QTT cache bridge preserves values and rank.
    #[test]
    fn real_fused_qtt_round_trips_through_hdf5() {
        use crate::gaussian::AnisoMixture2D;

        let r = 6;
        let box_l = 1.0;
        let mix = AnisoMixture2D::random(3, 0.8, 0.1, 3.0, 7);
        let tt = mix
            .to_interpolative_qtt(r, box_l, 16, 1e-10, 1e-10)
            .unwrap();

        let path = std::env::temp_dir().join("t4a_bench_hdf5_export_roundtrip_f64.h5");
        let _ = std::fs::remove_file(&path);
        let path_str = path.to_str().unwrap();

        save_tt_as_mps(path_str, "f", &tt, false).unwrap();

        let loaded = tensor4all_hdf5::load_mps(path_str, "f").unwrap();
        assert_eq!(loaded.len(), r);
        assert_eq!(loaded.max_bond_dim(), tt.rank());

        let back = load_tt_from_mps::<f64>(path_str, "f").unwrap();
        for &(i, j) in &[(0u64, 0u64), (37, 20), (63, 1), (32, 32)] {
            let xb = crate::harness::index_to_bits(i, r);
            let yb = crate::harness::index_to_bits(j, r);
            let fused: Vec<usize> = (0..r).map(|n| xb[n] + 2 * yb[n]).collect();
            let got = back.evaluate(&fused).unwrap();
            let want = tt.evaluate(&fused).unwrap();
            assert!(
                (got - want).abs() < 1e-12 * want.abs().max(1.0),
                "f64 round-trip mismatch at ({i},{j}): got {got} want {want}"
            );
        }

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn qtt_round_trips_through_hdf5() {
        let r = 6;
        let f = FourierSeries::random(3, 1);
        let g = FourierSeries::random(3, 2);
        let a = f.to_qtt(r).unwrap();
        let b = g.to_qtt(r).unwrap();

        let path = std::env::temp_dir().join("t4a_bench_hdf5_export_roundtrip.h5");
        let _ = std::fs::remove_file(&path);
        let path_str = path.to_str().unwrap();

        save_tt_as_mps(path_str, "f", &a, false).unwrap();
        save_tt_as_mps(path_str, "g", &b, true).unwrap();

        let loaded_f = tensor4all_hdf5::load_mps(path_str, "f").unwrap();
        let loaded_g = tensor4all_hdf5::load_mps(path_str, "g").unwrap();

        assert_eq!(loaded_f.len(), r);
        assert_eq!(loaded_g.len(), r);
        assert_eq!(loaded_f.max_bond_dim(), a.rank());
        assert_eq!(loaded_g.max_bond_dim(), b.rank());

        // Structure is not enough: check the stored amplitudes against the
        // analytic series at sampled grid points.
        let scale = (1u64 << r) as f64;
        for (name, series) in [("f", &f), ("g", &g)] {
            let tt = load_tt_from_mps::<Complex64>(path_str, name).unwrap();
            for &i in &sample_grid_indices(r, 20, 11) {
                let got = tt.evaluate(&index_to_bits(i, r)).unwrap();
                let want = series.eval(i as f64 / scale);
                assert!(
                    (got - want).norm() < 1e-10,
                    "{name}: round-trip mismatch at i={i}: got {got} want {want}"
                );
            }
        }

        std::fs::remove_file(&path).unwrap();
    }
}
