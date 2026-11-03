use super::*;
use crate::elementwise::{max_rel_error_vs_product, AciTolerance};
use crate::gaussian::AnisoMixture2D;

fn fixture() -> (
    Vec<DynIndex>,
    AnisoMixture2D,
    AnisoMixture2D,
    SimpleTT<f64>,
    SimpleTT<f64>,
    PartitionedTT,
    PartitionedTT,
) {
    let (r, box_l) = (6, 1.0);
    let left_mix = AnisoMixture2D::random(2, 0.8, 0.15, 2.0, 1);
    let right_mix = AnisoMixture2D::random(2, 0.8, 0.15, 2.0, 2);
    let left = left_mix
        .to_interpolative_qtt(r, box_l, 12, 1e-9, 1e-10)
        .unwrap();
    let right = right_mix
        .to_interpolative_qtt(r, box_l, 12, 1e-9, 1e-10)
        .unwrap();
    let sites = fused_site_indices(r);
    let options = NormPatchedInputOptions {
        rtol: 1e-8,
        max_bond_dim: crate::gaussian_input::PATCH_CAP,
        strategy: PatchSplitStrategy::Sequential,
    };
    let patched_left = patched_input_from_global(&left, &sites, options).unwrap();
    let patched_right = patched_input_from_global(&right, &sites, options).unwrap();
    (
        sites,
        left_mix,
        right_mix,
        left,
        right,
        patched_left,
        patched_right,
    )
}

#[test]
fn bridge_preserves_values_and_site_indices() {
    let (sites, _, _, left, _, _, _) = fixture();
    let bridged = tt_to_patch_train(&left, &sites).unwrap();
    let patched = PartitionedTT::from_subdomains(vec![SubDomainTT::from_tt(bridged)]).unwrap();
    let (_, subdomain) = patched.iter().next().unwrap();
    for sample in sample_grid_indices(left.len(), 32, 9) {
        let bits = index_to_bits(sample, left.len());
        let fused: Vec<_> = bits.into_iter().map(|bit| bit * 3).collect();
        assert!(
            (eval_patched(&patched, &sites, &fused).unwrap() - left.evaluate(&fused).unwrap())
                .abs()
                < 1e-10
        );
    }
    for site in &sites {
        assert!(subdomain
            .data()
            .site_indices()
            .iter()
            .flatten()
            .any(|index| index == site));
    }
}

#[test]
fn fit_and_aci_patched_products_match_the_reference() {
    let (sites, left_mix, right_mix, left, right, patched_left, patched_right) = fixture();
    for (engine, global_algo, aci_tolerance) in [
        (
            PatchedEngine::FitTreetn,
            ElementwiseAlgo::Fit,
            AciTolerance::Absolute,
        ),
        (
            PatchedEngine::Aci,
            ElementwiseAlgo::Aci,
            AciTolerance::ScaleRelative,
        ),
    ] {
        let global =
            elementwise_product(global_algo, &left, &right, 1e-8, 256, aci_tolerance).unwrap();
        let patched = patched_elementwise(
            &patched_left,
            &patched_right,
            &sites,
            PatchedProductOptions {
                engine,
                product_tol: 1e-10,
                product_max_bond_dim: 256,
                rtol: 1e-8,
                max_bond_dim: crate::gaussian_input::PATCH_CAP,
            },
        )
        .unwrap();
        let global_error = max_rel_error_vs_product(&global, &left_mix, &right_mix, 6, 1.0, 64, 17);
        let patched_error =
            max_rel_error_patched(&patched, &sites, &left_mix, &right_mix, 1.0, 64, 17).unwrap();
        assert!(
            global_error < 1e-4,
            "{engine:?}: global error={global_error:.3e}"
        );
        assert!(
            patched_error < 1e-4,
            "{engine:?}: patched error={patched_error:.3e}"
        );
        assert!(max_patch_bond(&patched) <= crate::gaussian_input::PATCH_CAP);
    }
}
