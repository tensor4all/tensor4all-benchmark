use super::*;

#[test]
fn cached_pair_round_trip_preserves_values_and_rank() {
    let cache_dir = std::env::temp_dir().join(format!("t4a-gaussian-cache-{}", std::process::id()));
    let config = GaussianInputConfig {
        n: 2,
        sigma_minor: 0.12,
        rho_max: 2.0,
        spacing: 3.0,
        polynomial_degree: 12,
        interpolation_tolerance: 1e-9,
        addition_tolerance: 1e-10,
        seed: 31,
        cache_dir: cache_dir.clone(),
        refresh: true,
    };
    let first = prepare_gaussian_pair(&config).unwrap();
    let second = prepare_gaussian_pair(&GaussianInputConfig {
        refresh: false,
        ..config
    })
    .unwrap();
    assert!(!first.cache_hit);
    assert!(second.cache_hit);
    assert_eq!(first.raw_left_chi, second.raw_left_chi);
    assert_eq!(first.raw_right_chi, second.raw_right_chi);
    for point in [vec![0; first.r], (0..first.r).map(|i| i % 4).collect()] {
        assert!(
            (first.left.evaluate(&point).unwrap() - second.left.evaluate(&point).unwrap()).abs()
                < 1e-12
        );
    }
    let left_mpo = crate::gaussian::fused_qtt_to_mpo(&second.left).unwrap();
    let right_mpo = crate::gaussian::fused_qtt_to_mpo(&second.right).unwrap();
    let (left_train, right_train) =
        crate::patched_mpo::mpo_pair_to_tensortrains(&left_mpo, &right_mpo).unwrap();
    for (left_sites, right_sites) in left_train
        .site_indices()
        .iter()
        .zip(right_train.site_indices())
    {
        assert_eq!(
            left_sites
                .iter()
                .filter(|index| right_sites.contains(index))
                .count(),
            1
        );
    }
    std::fs::remove_dir_all(cache_dir).unwrap();
}
