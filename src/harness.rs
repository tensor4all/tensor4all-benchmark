use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

pub struct Timing {
    pub median_secs: f64,
    pub runs_secs: Vec<f64>,
    /// Heap allocation calls made by the timed operation, median over the runs.
    pub alloc_calls_median: u64,
    /// Bytes requested by those calls, median over the runs.
    pub alloc_bytes_median: u64,
}

/// Run `f` `warmups` times untimed, then `repeats` times timed.
/// Returns the last result and the timing summary.
pub fn time_median<R>(warmups: usize, repeats: usize, mut f: impl FnMut() -> R) -> (R, Timing) {
    assert!(repeats >= 1);
    for _ in 0..warmups {
        std::hint::black_box(f());
    }
    let mut runs_secs = Vec::with_capacity(repeats);
    let mut alloc_calls = Vec::with_capacity(repeats);
    let mut alloc_bytes = Vec::with_capacity(repeats);
    let mut last = None;
    for _ in 0..repeats {
        let before = crate::alloc::snapshot();
        let t0 = Instant::now();
        last = Some(std::hint::black_box(f()));
        runs_secs.push(t0.elapsed().as_secs_f64());
        let allocated = crate::alloc::delta(before);
        alloc_calls.push(allocated.calls);
        alloc_bytes.push(allocated.bytes);
    }
    let mut sorted = runs_secs.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_secs = sorted[sorted.len() / 2];
    (
        last.unwrap(),
        Timing {
            median_secs,
            runs_secs,
            alloc_calls_median: median_u64(&alloc_calls),
            alloc_bytes_median: median_u64(&alloc_bytes),
        },
    )
}

fn median_u64(values: &[u64]) -> u64 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

/// `n` random grid indices in [0, 2^r), deterministic in `seed`.
pub fn sample_grid_indices(r: usize, n: usize, seed: u64) -> Vec<u64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n).map(|_| rng.random_range(0..(1u64 << r))).collect()
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
        let (val, timing) = time_median(1, 3, || vec![40u8, 2]);
        assert_eq!(val, vec![40u8, 2]);
        assert_eq!(timing.runs_secs.len(), 3);
        assert!(timing.median_secs >= 0.0);
        assert!(
            timing.alloc_calls_median >= 1,
            "the timed closure allocates"
        );
        assert!(timing.alloc_bytes_median >= 2);
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
