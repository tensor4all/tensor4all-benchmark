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
}

pub fn write_record(dir: &Path, name: &str, record: &RunRecord) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("{name}.json"));
    std::fs::write(&path, serde_json::to_string_pretty(record)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_roundtrips_to_json() {
        let rec = RunRecord {
            schema_version: SCHEMA_VERSION,
            case: "elementwise_fourier".into(),
            algorithm: "aci".into(),
            params: serde_json::json!({"k_max": 8, "r": 12}),
            seed: 0,
            tolerance: 1e-8,
            wall_time_median_secs: 0.1,
            wall_times_secs: vec![0.1],
            max_error: 1e-9,
            input_max_bond_dim: 5,
            output_max_bond_dim: 9,
            output_bond_dims: vec![2, 9, 2],
            n_params: Some(120),
            n_patches: None,
            max_patch_bond: None,
            rtol: None,
            input_build_secs: None,
        };
        let s = serde_json::to_string(&rec).unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["schema_version"], 1);
        assert_eq!(v["params"]["k_max"], 8);
        assert_eq!(v["n_params"], 120);
        // The unset optional fields are omitted rather than written as null, so
        // a record only ever carries the fields its case actually measured.
        assert!(
            v.get("n_patches").is_none(),
            "an unset optional field was written"
        );
        assert!(
            v.get("rtol").is_none(),
            "an unset optional field was written"
        );
    }
}
