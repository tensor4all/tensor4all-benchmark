use super::*;

#[test]
fn delta_counts_a_known_allocation() {
    let before = snapshot();
    let payload: Vec<u8> = vec![7u8; 4096];
    let after = delta(before);
    assert!(
        after.calls >= 1,
        "allocating a 4 KiB vector must be counted"
    );
    assert!(
        after.bytes >= 4096,
        "the requested bytes must cover the vector payload, got {}",
        after.bytes
    );
    assert_eq!(payload.iter().filter(|&&b| b == 7).count(), 4096);
}
