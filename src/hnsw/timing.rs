//! Timing utilities for performance analysis

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub struct Timer {
    name: &'static str,
    start: Instant,
}

impl Timer {
    pub fn new(name: &'static str) -> Self {
        Timer {
            name,
            start: Instant::now(),
        }
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed().as_micros();
        eprintln!("[TIMING] {}: {}μs", self.name, elapsed);
    }
}

// Global counters for operations
static SEARCH_LAYER_TIME: AtomicU64 = AtomicU64::new(0);
static PRUNE_TIME: AtomicU64 = AtomicU64::new(0);
static FETCH_NODE_TIME: AtomicU64 = AtomicU64::new(0);
static FETCH_NEIGHBORS_TIME: AtomicU64 = AtomicU64::new(0);
static INSERT_EDGE_TIME: AtomicU64 = AtomicU64::new(0);
static DISTANCE_CALC_TIME: AtomicU64 = AtomicU64::new(0);

pub fn add_search_layer_time(micros: u64) {
    SEARCH_LAYER_TIME.fetch_add(micros, Ordering::Relaxed);
}

pub fn add_prune_time(micros: u64) {
    PRUNE_TIME.fetch_add(micros, Ordering::Relaxed);
}

pub fn add_fetch_node_time(micros: u64) {
    FETCH_NODE_TIME.fetch_add(micros, Ordering::Relaxed);
}

pub fn add_fetch_neighbors_time(micros: u64) {
    FETCH_NEIGHBORS_TIME.fetch_add(micros, Ordering::Relaxed);
}

pub fn add_insert_edge_time(micros: u64) {
    INSERT_EDGE_TIME.fetch_add(micros, Ordering::Relaxed);
}

pub fn add_distance_calc_time(micros: u64) {
    DISTANCE_CALC_TIME.fetch_add(micros, Ordering::Relaxed);
}

pub fn print_timing_summary() {
    eprintln!("\n=== Timing Summary ===");
    eprintln!("search_layer:     {}ms", SEARCH_LAYER_TIME.load(Ordering::Relaxed) / 1000);
    eprintln!("prune:            {}ms", PRUNE_TIME.load(Ordering::Relaxed) / 1000);
    eprintln!("fetch_node:       {}ms", FETCH_NODE_TIME.load(Ordering::Relaxed) / 1000);
    eprintln!("fetch_neighbors:  {}ms", FETCH_NEIGHBORS_TIME.load(Ordering::Relaxed) / 1000);
    eprintln!("insert_edge:      {}ms", INSERT_EDGE_TIME.load(Ordering::Relaxed) / 1000);
    eprintln!("distance_calc:    {}ms", DISTANCE_CALC_TIME.load(Ordering::Relaxed) / 1000);
}

pub fn reset_timers() {
    SEARCH_LAYER_TIME.store(0, Ordering::Relaxed);
    PRUNE_TIME.store(0, Ordering::Relaxed);
    FETCH_NODE_TIME.store(0, Ordering::Relaxed);
    FETCH_NEIGHBORS_TIME.store(0, Ordering::Relaxed);
    INSERT_EDGE_TIME.store(0, Ordering::Relaxed);
    DISTANCE_CALC_TIME.store(0, Ordering::Relaxed);
}
