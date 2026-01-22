//! HNSW search algorithm implementation
//!
//! Implements the page-cache based search algorithm that queries nodes/edges
//! from shadow tables on demand instead of keeping the entire graph in memory.

use crate::distance;
use crate::error::{Error, Result};
use crate::hnsw::HnswMetadata;
use crate::hnsw::storage;
use crate::vector::{IndexQuantization, Vector, VectorType};
use fixedbitset::FixedBitSet;
use rusqlite::{Connection, ffi};
use std::collections::{BinaryHeap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

/// Hybrid visited set: FixedBitSet for dense rowids [1, capacity), HashSet for outliers
/// Provides ~100x faster lookups for typical HNSW usage with sequential rowids
struct HybridVisited {
    bits: FixedBitSet,
    overflow: HashSet<i64>,
}

impl HybridVisited {
    fn new(capacity: usize) -> Self {
        Self {
            bits: FixedBitSet::with_capacity(capacity),
            overflow: HashSet::new(),
        }
    }

    #[inline]
    fn contains(&self, rowid: &i64) -> bool {
        let rowid = *rowid;
        if rowid > 0 && (rowid as usize) < self.bits.len() {
            self.bits.contains(rowid as usize)
        } else {
            self.overflow.contains(&rowid)
        }
    }

    #[inline]
    fn insert(&mut self, rowid: i64) -> bool {
        if rowid > 0 && (rowid as usize) < self.bits.len() {
            let was_set = self.bits.contains(rowid as usize);
            self.bits.insert(rowid as usize);
            !was_set
        } else {
            self.overflow.insert(rowid)
        }
    }
}

// Timing counters for search_layer breakdown
static SEARCH_FETCH_EDGES_TIME: AtomicU64 = AtomicU64::new(0);
static SEARCH_FETCH_NODES_TIME: AtomicU64 = AtomicU64::new(0);
static SEARCH_DISTANCE_TIME: AtomicU64 = AtomicU64::new(0);
static SEARCH_HEAP_TIME: AtomicU64 = AtomicU64::new(0);
static SEARCH_VISITED_TIME: AtomicU64 = AtomicU64::new(0);
static SEARCH_FROM_BLOB_TIME: AtomicU64 = AtomicU64::new(0);
static SEARCH_LOOP_ITERATIONS: AtomicU64 = AtomicU64::new(0);
static SEARCH_NEIGHBORS_FETCHED: AtomicU64 = AtomicU64::new(0);
static SEARCH_DISTANCES_COMPUTED: AtomicU64 = AtomicU64::new(0);
// Batch size distribution buckets: 1-4, 5-16, 17-32, 33-64, 65+
static BATCH_SIZE_1_4: AtomicU64 = AtomicU64::new(0);
static BATCH_SIZE_5_16: AtomicU64 = AtomicU64::new(0);
static BATCH_SIZE_17_32: AtomicU64 = AtomicU64::new(0);
static BATCH_SIZE_33_64: AtomicU64 = AtomicU64::new(0);
static BATCH_SIZE_65_PLUS: AtomicU64 = AtomicU64::new(0);
static BATCH_FETCH_CALLS: AtomicU64 = AtomicU64::new(0);

pub fn print_search_timing_stats() {
    eprintln!("\n=== SEARCH_LAYER BREAKDOWN ===");
    eprintln!(
        "  fetch_edges:    {}ms",
        SEARCH_FETCH_EDGES_TIME.load(Ordering::Relaxed) / 1000
    );
    eprintln!(
        "  fetch_nodes:    {}ms",
        SEARCH_FETCH_NODES_TIME.load(Ordering::Relaxed) / 1000
    );
    eprintln!(
        "  visited_check:  {}ms",
        SEARCH_VISITED_TIME.load(Ordering::Relaxed) / 1000
    );
    eprintln!(
        "  loop_iters:     {}",
        SEARCH_LOOP_ITERATIONS.load(Ordering::Relaxed)
    );
    eprintln!(
        "  neighbors:      {}",
        SEARCH_NEIGHBORS_FETCHED.load(Ordering::Relaxed)
    );
    eprintln!(
        "  distances:      {}",
        SEARCH_DISTANCES_COMPUTED.load(Ordering::Relaxed)
    );

    let total_batches = BATCH_FETCH_CALLS.load(Ordering::Relaxed);
    if total_batches > 0 {
        eprintln!("\n  Batch size distribution ({} fetches):", total_batches);
        eprintln!(
            "    1-4:    {} ({:.1}%)",
            BATCH_SIZE_1_4.load(Ordering::Relaxed),
            BATCH_SIZE_1_4.load(Ordering::Relaxed) as f64 / total_batches as f64 * 100.0
        );
        eprintln!(
            "    5-16:   {} ({:.1}%)",
            BATCH_SIZE_5_16.load(Ordering::Relaxed),
            BATCH_SIZE_5_16.load(Ordering::Relaxed) as f64 / total_batches as f64 * 100.0
        );
        eprintln!(
            "    17-32:  {} ({:.1}%)",
            BATCH_SIZE_17_32.load(Ordering::Relaxed),
            BATCH_SIZE_17_32.load(Ordering::Relaxed) as f64 / total_batches as f64 * 100.0
        );
        eprintln!(
            "    33-64:  {} ({:.1}%)",
            BATCH_SIZE_33_64.load(Ordering::Relaxed),
            BATCH_SIZE_33_64.load(Ordering::Relaxed) as f64 / total_batches as f64 * 100.0
        );
        eprintln!(
            "    65+:    {} ({:.1}%)",
            BATCH_SIZE_65_PLUS.load(Ordering::Relaxed),
            BATCH_SIZE_65_PLUS.load(Ordering::Relaxed) as f64 / total_batches as f64 * 100.0
        );
    }
}

pub fn reset_search_timing_stats() {
    SEARCH_FETCH_EDGES_TIME.store(0, Ordering::Relaxed);
    SEARCH_FETCH_NODES_TIME.store(0, Ordering::Relaxed);
    SEARCH_DISTANCE_TIME.store(0, Ordering::Relaxed);
    SEARCH_HEAP_TIME.store(0, Ordering::Relaxed);
    SEARCH_VISITED_TIME.store(0, Ordering::Relaxed);
    SEARCH_FROM_BLOB_TIME.store(0, Ordering::Relaxed);
    SEARCH_LOOP_ITERATIONS.store(0, Ordering::Relaxed);
    SEARCH_NEIGHBORS_FETCHED.store(0, Ordering::Relaxed);
    SEARCH_DISTANCES_COMPUTED.store(0, Ordering::Relaxed);
    BATCH_SIZE_1_4.store(0, Ordering::Relaxed);
    BATCH_SIZE_5_16.store(0, Ordering::Relaxed);
    BATCH_SIZE_17_32.store(0, Ordering::Relaxed);
    BATCH_SIZE_33_64.store(0, Ordering::Relaxed);
    BATCH_SIZE_65_PLUS.store(0, Ordering::Relaxed);
    BATCH_FETCH_CALLS.store(0, Ordering::Relaxed);
}

/// Cached statement pointers for search operations
pub struct SearchStmtCache {
    pub get_node_data: Option<*mut ffi::sqlite3_stmt>,
    pub get_edges: Option<*mut ffi::sqlite3_stmt>,
    /// Single batch fetch statement with 64 placeholders (pad unused with -1)
    pub batch_fetch_nodes: Option<*mut ffi::sqlite3_stmt>,
}

/// Search context to reduce parameter count
pub struct SearchContext<'a> {
    pub db: &'a Connection,
    pub metadata: &'a HnswMetadata,
    pub table_name: &'a str,
    pub column_name: &'a str,
    pub query_vec: &'a Vector,
    pub stmt_cache: Option<&'a SearchStmtCache>,
}

/// Candidate for exploration - min-heap ordering (closest first)
#[derive(Debug, Clone)]
struct MinCandidate {
    rowid: i64,
    distance: f32,
}

impl PartialEq for MinCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.rowid == other.rowid
    }
}

impl Eq for MinCandidate {}

impl PartialOrd for MinCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering for min-heap (closest first when popped)
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Result candidate - max-heap ordering (farthest first)
#[derive(Debug, Clone)]
struct MaxCandidate {
    rowid: i64,
    distance: f32,
}

impl PartialEq for MaxCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.rowid == other.rowid
    }
}

impl Eq for MaxCandidate {}

impl PartialOrd for MaxCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MaxCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Normal ordering for max-heap (farthest first when peeked/popped)
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Search for k nearest neighbors using HNSW algorithm
///
/// # Arguments
/// * `db` - Database connection
/// * `metadata` - HNSW index metadata
/// * `table_name` - Virtual table name
/// * `column_name` - Vector column name
/// * `query_vector` - Query vector as bytes
/// * `k` - Number of nearest neighbors to return
/// * `ef_search` - Dynamic candidate list size (defaults to metadata.params.ef_search)
/// * `stmt_cache` - Optional cached statements for faster execution
///
/// # Returns
/// List of (rowid, distance) pairs, sorted by distance
#[allow(clippy::too_many_arguments)]
pub fn search_hnsw(
    db: &Connection,
    metadata: &HnswMetadata,
    table_name: &str,
    column_name: &str,
    query_vector: &[u8],
    k: usize,
    ef_search: Option<i32>,
    stmt_cache: Option<&SearchStmtCache>,
) -> Result<Vec<(i64, f32)>> {
    // Check if index is empty
    if metadata.entry_point_rowid == -1 {
        return Ok(Vec::new());
    }

    let ef = ef_search.unwrap_or(metadata.params.ef_search).max(k as i32);

    // Parse query vector
    let mut query_vec = Vector::from_blob(
        query_vector,
        metadata.element_type,
        metadata.dimensions as usize,
    )?;

    // Normalize query vector if using Cosine distance (to match normalized stored vectors)
    if metadata.normalize_vectors && metadata.element_type == VectorType::Float32 {
        query_vec = query_vec.normalize()?;
    }

    // Optionally quantize query vector to match index storage
    let query_vec = match metadata.index_quantization {
        IndexQuantization::Int8 if metadata.element_type == VectorType::Float32 => {
            query_vec.quantize_int8_for_index()?
        }
        _ => query_vec,
    };

    // Create search context
    let ctx = SearchContext {
        db,
        metadata,
        table_name,
        column_name,
        query_vec: &query_vec,
        stmt_cache,
    };

    // Start from entry point, search from top level down to level 0
    let mut current_nearest = metadata.entry_point_rowid;

    // Greedy search from top level down to level 1
    for level in (1..=metadata.entry_point_level).rev() {
        current_nearest = search_layer(&ctx, current_nearest, 1, level)? // ef=1 for greedy search
            .first()
            .map(|(rowid, _)| *rowid)
            .unwrap_or(current_nearest);
    }

    // Search at level 0 with full ef_search
    let results = search_layer(&ctx, current_nearest, ef as usize, 0)?;

    // Return top-k results with distances converted to user-requested metric
    // For Cosine with normalize_vectors, internal L2 distances are converted to cosine distances
    Ok(results
        .into_iter()
        .take(k)
        .map(|(rowid, dist)| (rowid, metadata.convert_distance_for_output(dist)))
        .collect())
}

/// Search a single layer for nearest neighbors
///
/// This is the core HNSW search algorithm at one layer
pub fn search_layer(
    ctx: &SearchContext,
    entry_rowid: i64,
    ef: usize,
    level: i32,
) -> Result<Vec<(i64, f32)>> {
    // Use HybridVisited for ~100x faster lookups with dense rowids
    // Capacity is num_nodes + 1 to handle rowids 1..num_nodes
    let visited_capacity = (ctx.metadata.num_nodes as usize)
        .saturating_add(1)
        .max(ef * 10);
    let mut visited = HybridVisited::new(visited_capacity);
    // candidates: min-heap - we explore closest candidates first
    let mut candidates: BinaryHeap<MinCandidate> = BinaryHeap::with_capacity(ef * 2);
    // results: max-heap - peek() gives us the worst (farthest) result for termination check
    let mut results: BinaryHeap<MaxCandidate> = BinaryHeap::with_capacity(ef + 1);

    // Get entry point node
    let cached_node_stmt = ctx.stmt_cache.and_then(|c| c.get_node_data);
    let entry_node = storage::fetch_node_data(
        ctx.db,
        ctx.table_name,
        ctx.column_name,
        entry_rowid,
        cached_node_stmt,
    )?
    .ok_or_else(|| {
        Error::InvalidParameter(format!("Entry point node {} not found", entry_rowid))
    })?;

    // Determine the element type for stored vectors (may be quantized)
    let stored_element_type = match ctx.metadata.index_quantization {
        IndexQuantization::Int8 if ctx.metadata.element_type == VectorType::Float32 => {
            VectorType::Int8
        }
        _ => ctx.metadata.element_type,
    };

    let entry_vec = Vector::from_blob(
        &entry_node.vector,
        stored_element_type,
        ctx.metadata.dimensions as usize,
    )?;

    let entry_dist = distance::distance(
        ctx.query_vec,
        &entry_vec,
        ctx.metadata.internal_distance_metric(),
    )?;

    candidates.push(MinCandidate {
        rowid: entry_rowid,
        distance: entry_dist,
    });

    results.push(MaxCandidate {
        rowid: entry_rowid,
        distance: entry_dist,
    });

    visited.insert(entry_rowid);

    // Main search loop
    while let Some(candidate) = candidates.pop() {
        // If closest unexplored candidate is farther than our worst result, we're done
        if let Some(worst) = results.peek()
            && candidate.distance > worst.distance
        {
            break;
        }

        // Get neighbors of this candidate at the current level (rowids only - cheap)
        let cached_edges_stmt = ctx.stmt_cache.and_then(|c| c.get_edges);
        let neighbors = storage::fetch_neighbors_cached(
            ctx.db,
            ctx.table_name,
            ctx.column_name,
            candidate.rowid,
            level,
            cached_edges_stmt,
        )?;

        // Filter to unvisited neighbors and mark them visited BEFORE expensive fetch
        let unvisited_neighbors: Vec<i64> = neighbors
            .into_iter()
            .filter(|&rowid| {
                if visited.contains(&rowid) {
                    false
                } else {
                    visited.insert(rowid);
                    true
                }
            })
            .collect();

        if unvisited_neighbors.is_empty() {
            continue;
        }

        SEARCH_NEIGHBORS_FETCHED.fetch_add(unvisited_neighbors.len() as u64, Ordering::Relaxed);

        // Track batch size distribution
        BATCH_FETCH_CALLS.fetch_add(1, Ordering::Relaxed);
        let batch_size = unvisited_neighbors.len();
        match batch_size {
            1..=4 => BATCH_SIZE_1_4.fetch_add(1, Ordering::Relaxed),
            5..=16 => BATCH_SIZE_5_16.fetch_add(1, Ordering::Relaxed),
            17..=32 => BATCH_SIZE_17_32.fetch_add(1, Ordering::Relaxed),
            33..=64 => BATCH_SIZE_33_64.fetch_add(1, Ordering::Relaxed),
            _ => BATCH_SIZE_65_PLUS.fetch_add(1, Ordering::Relaxed),
        };

        // BATCH FETCH: Get ONLY unvisited neighbor nodes
        // Use cached statement (16 placeholders) for fast path, fallback to dynamic SQL
        let neighbor_nodes = if let Some(stmt) = ctx
            .stmt_cache
            .and_then(|c| c.batch_fetch_nodes)
            .filter(|s| !s.is_null())
        {
            // Fast path: use cached prepared statement with 16 placeholders
            let mut all_nodes = Vec::with_capacity(unvisited_neighbors.len());
            for chunk in unvisited_neighbors.chunks(16) {
                unsafe {
                    let nodes = storage::fetch_nodes_batch_cached(stmt, 16, chunk)?;
                    all_nodes.extend(nodes);
                }
            }
            all_nodes
        } else {
            // Slow path: dynamic SQL (fallback for tests without cache)
            storage::fetch_nodes_batch(
                ctx.db,
                ctx.table_name,
                ctx.column_name,
                &unvisited_neighbors,
            )?
        };

        for neighbor_node in neighbor_nodes {
            let neighbor_vec = Vector::from_blob(
                &neighbor_node.vector,
                stored_element_type,
                ctx.metadata.dimensions as usize,
            )?;

            let neighbor_dist = distance::distance(
                ctx.query_vec,
                &neighbor_vec,
                ctx.metadata.internal_distance_metric(),
            )?;

            // Check if this neighbor is better than our worst result (or we have room)
            if results.len() < ef || neighbor_dist < results.peek().unwrap().distance {
                candidates.push(MinCandidate {
                    rowid: neighbor_node.rowid,
                    distance: neighbor_dist,
                });

                results.push(MaxCandidate {
                    rowid: neighbor_node.rowid,
                    distance: neighbor_dist,
                });

                // Trim results to ef (removes farthest)
                while results.len() > ef {
                    results.pop();
                }
            }
        }
    }

    // Convert results heap to sorted vector (closest first)
    let mut final_results: Vec<(i64, f32)> =
        results.into_iter().map(|c| (c.rowid, c.distance)).collect();

    // Sort by distance ascending (closest first)
    final_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    Ok(final_results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::DistanceMetric;
    use crate::shadow;
    use crate::vector::VectorType;

    #[test]
    fn test_search_empty_index() {
        let db = Connection::open_in_memory().unwrap();
        shadow::create_hnsw_shadow_tables(&db, "test_table", "embedding").unwrap();

        let metadata = HnswMetadata::new(3, VectorType::Float32, DistanceMetric::L2);

        let query = vec![1u8, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]; // [1.0, 2.0, 3.0]
        let results = search_hnsw(
            &db,
            &metadata,
            "test_table",
            "embedding",
            &query,
            5,
            None,
            None,
        )
        .unwrap();

        assert_eq!(results.len(), 0, "Empty index should return no results");
    }

    #[test]
    fn test_search_single_node() {
        let db = Connection::open_in_memory().unwrap();
        shadow::create_hnsw_shadow_tables(&db, "test_table", "embedding").unwrap();

        // Create metadata with one node as entry point
        let mut metadata = HnswMetadata::new(3, VectorType::Float32, DistanceMetric::L2);
        metadata.entry_point_rowid = 1;
        metadata.entry_point_level = 0;
        metadata.num_nodes = 1;

        // Insert the node
        let vector = vec![1u8, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]; // [1.0, 2.0, 3.0]
        storage::insert_node(&db, "test_table", "embedding", 1, 0, &vector, None).unwrap();

        // Search for it
        let query = vec![1u8, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]; // Same vector
        let results = search_hnsw(
            &db,
            &metadata,
            "test_table",
            "embedding",
            &query,
            1,
            None,
            None,
        )
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
        assert!(
            results[0].1 < 0.001,
            "Distance to itself should be near zero"
        );
    }

    #[test]
    fn test_search_candidate_ordering() {
        // Test MinCandidate (min-heap - closest first)
        let mut min_heap: BinaryHeap<MinCandidate> = BinaryHeap::new();

        min_heap.push(MinCandidate {
            rowid: 1,
            distance: 0.5,
        });
        min_heap.push(MinCandidate {
            rowid: 2,
            distance: 0.3,
        });
        min_heap.push(MinCandidate {
            rowid: 3,
            distance: 0.7,
        });

        // Should pop in order: 0.3, 0.5, 0.7 (min-heap)
        let first = min_heap.pop().unwrap();
        assert_eq!(first.rowid, 2);
        assert!((first.distance - 0.3).abs() < 0.001);

        let second = min_heap.pop().unwrap();
        assert_eq!(second.rowid, 1);
        assert!((second.distance - 0.5).abs() < 0.001);

        // Test MaxCandidate (max-heap - farthest first)
        let mut max_heap: BinaryHeap<MaxCandidate> = BinaryHeap::new();

        max_heap.push(MaxCandidate {
            rowid: 1,
            distance: 0.5,
        });
        max_heap.push(MaxCandidate {
            rowid: 2,
            distance: 0.3,
        });
        max_heap.push(MaxCandidate {
            rowid: 3,
            distance: 0.7,
        });

        // Should pop in order: 0.7, 0.5, 0.3 (max-heap)
        let first = max_heap.pop().unwrap();
        assert_eq!(first.rowid, 3);
        assert!((first.distance - 0.7).abs() < 0.001);

        let second = max_heap.pop().unwrap();
        assert_eq!(second.rowid, 1);
        assert!((second.distance - 0.5).abs() < 0.001);
    }
}
