use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

pub struct Timing {
    pub median_secs: f64,
    pub runs_secs: Vec<f64>,
}

/// Run `f` `warmups` times untimed, then `repeats` times timed.
/// Returns the last result and the timing summary.
pub fn time_median<R>(warmups: usize, repeats: usize, mut f: impl FnMut() -> R) -> (R, Timing) {
    assert!(repeats >= 1);
    for _ in 0..warmups {
        std::hint::black_box(f());
    }
    let mut runs_secs = Vec::with_capacity(repeats);
    let mut last = None;
    for _ in 0..repeats {
        let t0 = Instant::now();
        last = Some(std::hint::black_box(f()));
        runs_secs.push(t0.elapsed().as_secs_f64());
    }
    let mut sorted = runs_secs.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_secs = sorted[sorted.len() / 2];
    (
        last.unwrap(),
        Timing {
            median_secs,
            runs_secs,
        },
    )
}

/// `n` random grid indices in [0, 2^r), deterministic in `seed`.
pub fn sample_grid_indices(r: usize, n: usize, seed: u64) -> Vec<u64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen_range(0..(1u64 << r))).collect()
}

/// MSB-first bit decomposition of a grid index into R local indices.
pub fn index_to_bits(i: u64, r: usize) -> Vec<usize> {
    (0..r).map(|n| ((i >> (r - 1 - n)) & 1) as usize).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_median_returns_result_and_positive_times() {
        let (val, timing) = time_median(1, 3, || 40 + 2);
        assert_eq!(val, 42);
        assert_eq!(timing.runs_secs.len(), 3);
        assert!(timing.median_secs >= 0.0);
    }

    #[test]
    fn sample_grid_indices_is_deterministic_and_in_range() {
        let a = sample_grid_indices(10, 64, 7);
        let b = sample_grid_indices(10, 64, 7);
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
        assert!(a.iter().all(|&i| i < 1u64 << 10));
    }

    #[test]
    fn index_to_bits_msb_first() {
        assert_eq!(index_to_bits(0b100, 3), vec![1, 0, 0]);
        assert_eq!(index_to_bits(0b011, 3), vec![0, 1, 1]);
    }
}
