//! ITensorMPS.jl-compatible HDF5 export for benchmark inputs.
//!
//! The bridge is `simplett::TensorTrain` -> `treetn::TreeTN` ->
//! `itensorlike::TensorTrain` -> `tensor4all_hdf5::{save_mps, append_mps}`,
//! which writes the `MPS` v1 schema that ITensorMPS.jl reads.

use num_complex::Complex64;
use tensor4all_simplett::TensorTrain;
use tensor4all_treetn::tensor_train_to_treetn;

/// Write `tt` into `path` under group `name` as an ITensorMPS.jl `MPS`.
///
/// With `append = false` the file is created (or truncated); with
/// `append = true` the group is added to an existing file, which is how
/// several named MPS objects end up in one instance file.
pub fn save_tt_as_mps(
    path: &str,
    name: &str,
    tt: &TensorTrain<Complex64>,
    append: bool,
) -> anyhow::Result<()> {
    let (treetn, _indices) = tensor_train_to_treetn(tt)?;
    let itt = tensor4all_itensorlike::TensorTrain::from_treetn(treetn)?;
    if append {
        tensor4all_hdf5::append_mps(path, name, &itt)?;
    } else {
        tensor4all_hdf5::save_mps(path, name, &itt)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourier::FourierSeries;
    use crate::harness::{index_to_bits, sample_grid_indices};
    use tensor4all_simplett::AbstractTensorTrain;

    /// HDF5 `MPS` -> `itensorlike` -> `treetn` -> `simplett`, the inverse of
    /// the export bridge, so the loaded tensors can be evaluated pointwise.
    fn load_tt(path: &str, name: &str) -> TensorTrain<Complex64> {
        let itt = tensor4all_hdf5::load_mps(path, name).unwrap();
        tensor4all_treetn::treetn_to_tensor_train::<Complex64>(itt.into_treetn()).unwrap()
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
        assert_eq!(loaded_f.maxbonddim(), a.rank());
        assert_eq!(loaded_g.maxbonddim(), b.rank());

        // Structure is not enough: check the stored amplitudes against the
        // analytic series at sampled grid points.
        let scale = (1u64 << r) as f64;
        for (name, series) in [("f", &f), ("g", &g)] {
            let tt = load_tt(path_str, name);
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
