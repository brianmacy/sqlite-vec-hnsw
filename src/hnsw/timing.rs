//! Timing utilities for performance analysis
//!
//! This module is conditionally compiled based on the `timing` feature flag.
//! When disabled (default), all timing operations are no-ops with zero overhead.

#[cfg(feature = "timing")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "timing")]
use std::time::Instant;

/// Timer for scoped timing measurements (only active with `timing` feature)
#[cfg(feature = "timing")]
pub struct Timer {
    name: &'static str,
    start: Instant,
}

#[cfg(feature = "timing")]
impl Timer {
    pub fn new(name: &'static str) -> Self {
        Timer {
            name,
            start: Instant::now(),
        }
    }
}

#[cfg(feature = "timing")]
impl Drop for Timer {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed().as_micros();
        eprintln!("[TIMING] {}: {}μs", self.name, elapsed);
    }
}

/// No-op timer when timing feature is disabled
#[cfg(not(feature = "timing"))]
pub struct Timer;

#[cfg(not(feature = "timing"))]
impl Timer {
    #[inline(always)]
    pub fn new(_name: &'static str) -> Self {
        Timer
    }
}

// Global counters for operations (only when timing is enabled)
#[cfg(feature = "timing")]
static SEARCH_LAYER_TIME: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "timing")]
static PRUNE_TIME: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "timing")]
static FETCH_NODE_TIME: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "timing")]
static FETCH_NEIGHBORS_TIME: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "timing")]
static INSERT_EDGE_TIME: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "timing")]
static DISTANCE_CALC_TIME: AtomicU64 = AtomicU64::new(0);

/// Macro for conditional timing addition
/// When timing feature is enabled, adds to the atomic counter
/// When disabled, this is a complete no-op
#[macro_export]
macro_rules! timing_add {
    ($counter:expr, $value:expr) => {
        #[cfg(feature = "timing")]
        {
            $counter.fetch_add($value, std::sync::atomic::Ordering::Relaxed);
        }
        #[cfg(not(feature = "timing"))]
        {
            let _ = &$counter;
            let _ = $value;
        }
    };
}

/// Macro for conditional timing measurement
/// Returns elapsed microseconds when timing is enabled, 0 when disabled
#[macro_export]
macro_rules! timing_elapsed_micros {
    ($start:expr) => {{
        #[cfg(feature = "timing")]
        {
            $start.elapsed().as_micros() as u64
        }
        #[cfg(not(feature = "timing"))]
        {
            let _ = &$start;
            0u64
        }
    }};
}

/// Macro for conditional timing start
/// Returns Instant::now() when timing is enabled, dummy value when disabled
#[macro_export]
macro_rules! timing_start {
    () => {{
        #[cfg(feature = "timing")]
        {
            std::time::Instant::now()
        }
        #[cfg(not(feature = "timing"))]
        {
            ()
        }
    }};
}

#[cfg(feature = "timing")]
pub fn add_search_layer_time(micros: u64) {
    SEARCH_LAYER_TIME.fetch_add(micros, Ordering::Relaxed);
}

#[cfg(not(feature = "timing"))]
#[inline(always)]
pub fn add_search_layer_time(_micros: u64) {}

#[cfg(feature = "timing")]
pub fn add_prune_time(micros: u64) {
    PRUNE_TIME.fetch_add(micros, Ordering::Relaxed);
}

#[cfg(not(feature = "timing"))]
#[inline(always)]
pub fn add_prune_time(_micros: u64) {}

#[cfg(feature = "timing")]
pub fn add_fetch_node_time(micros: u64) {
    FETCH_NODE_TIME.fetch_add(micros, Ordering::Relaxed);
}

#[cfg(not(feature = "timing"))]
#[inline(always)]
pub fn add_fetch_node_time(_micros: u64) {}

#[cfg(feature = "timing")]
pub fn add_fetch_neighbors_time(micros: u64) {
    FETCH_NEIGHBORS_TIME.fetch_add(micros, Ordering::Relaxed);
}

#[cfg(not(feature = "timing"))]
#[inline(always)]
pub fn add_fetch_neighbors_time(_micros: u64) {}

#[cfg(feature = "timing")]
pub fn add_insert_edge_time(micros: u64) {
    INSERT_EDGE_TIME.fetch_add(micros, Ordering::Relaxed);
}

#[cfg(not(feature = "timing"))]
#[inline(always)]
pub fn add_insert_edge_time(_micros: u64) {}

#[cfg(feature = "timing")]
pub fn add_distance_calc_time(micros: u64) {
    DISTANCE_CALC_TIME.fetch_add(micros, Ordering::Relaxed);
}

#[cfg(not(feature = "timing"))]
#[inline(always)]
pub fn add_distance_calc_time(_micros: u64) {}

#[cfg(feature = "timing")]
pub fn print_timing_summary() {
    eprintln!("\n=== Timing Summary ===");
    eprintln!("search_layer:     {}ms", SEARCH_LAYER_TIME.load(Ordering::Relaxed) / 1000);
    eprintln!("prune:            {}ms", PRUNE_TIME.load(Ordering::Relaxed) / 1000);
    eprintln!("fetch_node:       {}ms", FETCH_NODE_TIME.load(Ordering::Relaxed) / 1000);
    eprintln!("fetch_neighbors:  {}ms", FETCH_NEIGHBORS_TIME.load(Ordering::Relaxed) / 1000);
    eprintln!("insert_edge:      {}ms", INSERT_EDGE_TIME.load(Ordering::Relaxed) / 1000);
    eprintln!("distance_calc:    {}ms", DISTANCE_CALC_TIME.load(Ordering::Relaxed) / 1000);
}

#[cfg(not(feature = "timing"))]
#[inline(always)]
pub fn print_timing_summary() {}

#[cfg(feature = "timing")]
pub fn reset_timers() {
    SEARCH_LAYER_TIME.store(0, Ordering::Relaxed);
    PRUNE_TIME.store(0, Ordering::Relaxed);
    FETCH_NODE_TIME.store(0, Ordering::Relaxed);
    FETCH_NEIGHBORS_TIME.store(0, Ordering::Relaxed);
    INSERT_EDGE_TIME.store(0, Ordering::Relaxed);
    DISTANCE_CALC_TIME.store(0, Ordering::Relaxed);
}

#[cfg(not(feature = "timing"))]
#[inline(always)]
pub fn reset_timers() {}
