//! Process-wide allocation counters for the timed arms.
//!
//! The maintained cases run small tensor networks, where per-call heap traffic
//! can outweigh the floating-point work: a variational fit at twenty sites with
//! bond dimension below thirty allocates two orders of magnitude more than the
//! network holds, and the count depends on the number of local updates rather
//! than on the tensor size. Recording allocations next to the wall time keeps
//! that overhead visible, so an upstream change to the contraction engine shows
//! up as a number instead of as an unexplained timing shift.
//!
//! The counters are only read around the timed region, so the two relaxed atomic
//! adds per allocation stay far below the timing noise.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

pub struct CountingAllocator;

static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

// Only `alloc` and `dealloc` are overridden. The trait's default `realloc` and
// `alloc_zeroed` both route through `self.alloc`, so every allocation event is
// counted exactly once without duplicating the system calls here.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Allocation counters at one instant, in allocation calls and requested bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllocSnapshot {
    pub calls: u64,
    pub bytes: u64,
}

pub fn snapshot() -> AllocSnapshot {
    AllocSnapshot {
        calls: ALLOC_CALLS.load(Ordering::Relaxed),
        bytes: ALLOC_BYTES.load(Ordering::Relaxed),
    }
}

/// Counters accumulated since `earlier`.
pub fn delta(earlier: AllocSnapshot) -> AllocSnapshot {
    let now = snapshot();
    AllocSnapshot {
        calls: now.calls - earlier.calls,
        bytes: now.bytes - earlier.bytes,
    }
}

#[cfg(test)]
mod tests {
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
}
