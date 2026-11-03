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
