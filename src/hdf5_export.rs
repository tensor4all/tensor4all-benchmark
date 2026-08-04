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
    use tensor4all_simplett::AbstractTensorTrain;

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

        std::fs::remove_file(&path).unwrap();
    }
}
