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
        n_allocations: Some(1234),
        allocated_bytes: Some(5678),
    };
    let s = serde_json::to_string(&rec).unwrap();
    let v: serde_json::Value = serde_json::from_str(&s).unwrap();
    assert_eq!(v["schema_version"], 1);
    assert_eq!(v["params"]["k_max"], 8);
    assert_eq!(v["n_params"], 120);
    assert_eq!(v["n_allocations"], 1234);
    assert_eq!(v["allocated_bytes"], 5678);
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
