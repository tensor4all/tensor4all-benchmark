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
