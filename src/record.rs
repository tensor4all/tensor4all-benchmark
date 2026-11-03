use serde::Serialize;
use std::path::Path;

pub const SCHEMA_VERSION: u32 = 1;

/// One timed arm of one instance.
///
/// The trailing fields are optional and are omitted from the JSON when they do
/// not apply, which is why the schema version stays at 1: every record written
/// before they existed is still a valid record of this shape, and the report
/// generator reads them defensively. A field that every case fills belongs above
/// the optional block instead.
#[derive(Serialize)]
pub struct RunRecord {
    pub schema_version: u32,
    pub case: String,
    pub algorithm: String,
    pub params: serde_json::Value,
    pub seed: u64,
    pub tolerance: f64,
    pub wall_time_median_secs: f64,
    pub wall_times_secs: Vec<f64>,
    pub max_error: f64,
    pub input_max_bond_dim: usize,
    pub output_max_bond_dim: usize,
    pub output_bond_dims: Vec<usize>,

    /// Stored parameters of the output, the sum of its core sizes over every
    /// patch of a patched representation or over the single train of a global
    /// one. The size metric that compares across those two shapes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_params: Option<usize>,
    /// Number of patches of a patched output. Absent on the global arms.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_patches: Option<usize>,
    /// Largest bond dimension over the patches of a patched output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_patch_bond: Option<usize>,
    /// Global relative tolerance the output was truncated at, on the cases that
    /// are tolerance-driven rather than budget-driven.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rtol: Option<f64>,
    /// Wall time of building the two inputs of this instance, reported apart
    /// from the product time because it is a different measurement: the same
    /// inputs are shared by every arm of the instance.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_build_secs: Option<f64>,
    /// Heap allocation calls made by the timed operation, median over the timed
    /// runs. Present on every arm measured by the current harness; absent from
    /// records written before the counters existed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_allocations: Option<u64>,
    /// Bytes requested by those allocation calls, median over the timed runs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allocated_bytes: Option<u64>,
}

pub fn write_record(dir: &Path, name: &str, record: &RunRecord) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("{name}.json"));
    std::fs::write(&path, serde_json::to_string_pretty(record)?)?;
    Ok(())
}

#[cfg(test)]
mod tests;
