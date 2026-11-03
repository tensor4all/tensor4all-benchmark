use super::*;
use crate::harness::index_to_bits;

fn fused_point(ix: u64, iy: u64, r: usize) -> Vec<usize> {
    index_to_bits(ix, r)
        .into_iter()
        .zip(index_to_bits(iy, r))
        .map(|(x, y)| x + 2 * y)
        .collect()
}

#[test]
fn rotated_gaussian_interpolation_resolves_a_narrow_ridge() {
    let (r, box_l) = (10, 1.0);
    let center = (0.07, -0.09);
    let (sigma, rho, angle) = (0.05_f64, 8.0_f64, std::f64::consts::FRAC_PI_4);
    let minor = 1.0 / (2.0 * sigma.powi(2));
    let major = minor / rho.powi(2);
    let (sin, cos) = angle.sin_cos();
    let quad = (
        major * cos * cos + minor * sin * sin,
        (major - minor) * sin * cos,
        major * sin * sin + minor * cos * cos,
    );
    let qtt = rotated_gaussian_qtt(r, box_l, center, quad, 1.3, 28, 1e-12).unwrap();
    let step = 2.0 * box_l / (1usize << r) as f64;
    let mut max_error = 0.0_f64;
    for sample in 0..256 {
        let ix = ((73 * sample + 11) % (1usize << r)) as u64;
        let iy = ((151 * sample + 29) % (1usize << r)) as u64;
        let x = grid_coord(ix, r, box_l);
        let y = grid_coord(iy, r, box_l);
        let (dx, dy) = (x - center.0, y - center.1);
        let expected =
            1.3 * (-(quad.0 * dx * dx + 2.0 * quad.1 * dx * dy + quad.2 * dy * dy)).exp();
        max_error =
            max_error.max((qtt.evaluate(&fused_point(ix, iy, r)).unwrap() - expected).abs());
    }
    assert!(
        max_error < 5e-8,
        "rank={}, error={max_error:.3e}",
        qtt.rank()
    );
    assert!(step <= sigma / 4.0);
}

#[test]
fn balanced_interpolative_sum_matches_the_mixture() {
    let (r, box_l) = (8, 1.0);
    let mixture = AnisoMixture2D::random(5, 0.7, 0.08, 3.0, 17);
    let qtt = mixture
        .to_interpolative_qtt(r, box_l, 24, 1e-12, 1e-10)
        .unwrap();
    let mut squared_error = 0.0;
    let mut squared_reference = 0.0;
    for sample in 0..128 {
        let ix = ((73 * sample + 11) % (1usize << r)) as u64;
        let iy = ((151 * sample + 29) % (1usize << r)) as u64;
        let expected = mixture.eval(grid_coord(ix, r, box_l), grid_coord(iy, r, box_l));
        let error = qtt.evaluate(&fused_point(ix, iy, r)).unwrap() - expected;
        squared_error += error * error;
        squared_reference += expected * expected;
    }
    assert!((squared_error / squared_reference).sqrt() < 1e-7);
}

#[test]
fn anisotropic_grid_reference_matches_the_explicit_sum() {
    let f = AnisoMixture2D::random(3, 1.0, 0.3, 4.0, 1);
    let g = AnisoMixture2D::random(2, 1.0, 0.3, 4.0, 2);
    let (x, z, r, box_l) = (0.3, -0.7, 10, 1.0);
    let step = 2.0 * box_l / (1u64 << r) as f64;
    let direct = (0..1u64 << r)
        .map(|i| f.eval(x, grid_coord(i, r, box_l)) * g.eval(grid_coord(i, r, box_l), z) * step)
        .sum::<f64>();
    let reference = discrete_contraction_aniso_reference(&f, &g, x, z, r, box_l);
    assert!((direct - reference).abs() < 1e-10 * direct.abs().max(1.0));
}

#[test]
fn random_draws_the_declared_anisotropic_family() {
    let (sigma, rho_max) = (0.05, 8.0);
    let mix = AnisoMixture2D::random(64, 1.0, sigma, rho_max, 7);
    assert_eq!(
        AnisoMixture2D::random(64, 1.0, sigma, rho_max, 7).quad,
        mix.quad
    );
    let expected_minor = 1.0 / (2.0 * sigma * sigma);
    for (i, &(a, b, c)) in mix.quad.iter().enumerate() {
        let mean = 0.5 * (a + c);
        let gap = (0.25 * (a - c).powi(2) + b * b).sqrt();
        let rho = (expected_minor / (mean - gap)).sqrt();
        assert!(
            (mean + gap - expected_minor).abs() < 1e-9 * expected_minor,
            "term {i}"
        );
        assert!((1.0..=rho_max).contains(&rho), "term {i}: rho={rho}");
        assert!((0.5..1.5).contains(&mix.weights[i]));
    }
}
