use serde::Serialize;
use std::path::Path;

pub const SCHEMA_VERSION: u32 = 1;

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
        };
        let s = serde_json::to_string(&rec).unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["schema_version"], 1);
        assert_eq!(v["params"]["k_max"], 8);
    }
}
