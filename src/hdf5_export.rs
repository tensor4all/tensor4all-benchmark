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
mod tests;
