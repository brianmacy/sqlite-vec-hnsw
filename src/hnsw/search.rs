//! HNSW search algorithm implementation
//!
//! Implements the page-cache based search algorithm that queries nodes/edges
//! from shadow tables on demand instead of keeping the entire graph in memory.

use crate::distance;
use crate::error::{Error, Result};
use crate::hnsw::storage;
use crate::hnsw::HnswMetadata;
use crate::vector::Vector;
use rusqlite::Connection;
use std::collections::{BinaryHeap, HashSet};

/// Search context to reduce parameter count
pub struct SearchContext<'a> {
    pub db: &'a Connection,
    pub metadata: &'a HnswMetadata,
    pub table_name: &'a str,
    pub column_name: &'a str,
    pub query_vec: &'a Vector,
}

/// Search candidate with priority ordering
#[derive(Debug, Clone)]
struct SearchCandidate {
    rowid: i64,
    distance: f32,
}

impl PartialEq for SearchCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.rowid == other.rowid
    }
}

impl Eq for SearchCandidate {}

impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering for min-heap (closest first)
        other
            .distance
            .partial_cmp(&self.distance)
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
///
/// # Returns
/// List of (rowid, distance) pairs, sorted by distance
pub fn search_hnsw(
    db: &Connection,
    metadata: &HnswMetadata,
    table_name: &str,
    column_name: &str,
    query_vector: &[u8],
    k: usize,
    ef_search: Option<i32>,
) -> Result<Vec<(i64, f32)>> {
    // Check if index is empty
    if metadata.entry_point_rowid == -1 {
        return Ok(Vec::new());
    }

    let ef = ef_search.unwrap_or(metadata.params.ef_search).max(k as i32);

    // Parse query vector
    let query_vec = Vector::from_blob(query_vector, metadata.element_type, metadata.dimensions as usize)?;

    // Create search context
    let ctx = SearchContext {
        db,
        metadata,
        table_name,
        column_name,
        query_vec: &query_vec,
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

    // Return top-k results
    Ok(results.into_iter().take(k).collect())
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
    let mut visited = HashSet::new();
    let mut candidates = BinaryHeap::new();
    let mut results = BinaryHeap::new();

    // Get entry point node
    let entry_node = storage::fetch_node_data(ctx.db, ctx.table_name, ctx.column_name, entry_rowid)?
        .ok_or_else(|| Error::InvalidParameter(format!("Entry point node {} not found", entry_rowid)))?;

    let entry_vec = Vector::from_blob(&entry_node.vector, ctx.metadata.element_type, ctx.metadata.dimensions as usize)?;
    let entry_dist = distance::distance(ctx.query_vec, &entry_vec, ctx.metadata.distance_metric)?;

    candidates.push(SearchCandidate {
        rowid: entry_rowid,
        distance: -entry_dist, // Negative for max-heap (farthest first)
    });

    results.push(SearchCandidate {
        rowid: entry_rowid,
        distance: entry_dist,
    });

    visited.insert(entry_rowid);

    // Main search loop
    while let Some(candidate) = candidates.pop() {
        let candidate_dist = -candidate.distance; // Convert back to positive

        // If this candidate is farther than the worst result, we're done
        if let Some(worst) = results.peek()
            && candidate_dist > worst.distance
        {
            break;
        }

        // Get neighbors of this candidate at the current level
        let neighbors = storage::fetch_neighbors(ctx.db, ctx.table_name, ctx.column_name, candidate.rowid, level)?;

        for neighbor_rowid in neighbors {
            if visited.contains(&neighbor_rowid) {
                continue;
            }
            visited.insert(neighbor_rowid);

            // Fetch neighbor vector and calculate distance
            let neighbor_node = storage::fetch_node_data(ctx.db, ctx.table_name, ctx.column_name, neighbor_rowid)?;
            let neighbor_node = match neighbor_node {
                Some(n) => n,
                None => continue, // Node deleted
            };

            let neighbor_vec = Vector::from_blob(
                &neighbor_node.vector,
                ctx.metadata.element_type,
                ctx.metadata.dimensions as usize,
            )?;
            let neighbor_dist = distance::distance(ctx.query_vec, &neighbor_vec, ctx.metadata.distance_metric)?;

            // Check if this neighbor is better than our worst result
            if results.len() < ef || neighbor_dist < results.peek().unwrap().distance {
                candidates.push(SearchCandidate {
                    rowid: neighbor_rowid,
                    distance: -neighbor_dist,
                });

                results.push(SearchCandidate {
                    rowid: neighbor_rowid,
                    distance: neighbor_dist,
                });

                // Trim results to ef
                while results.len() > ef {
                    results.pop();
                }
            }
        }
    }

    // Convert results to sorted vector
    let mut final_results: Vec<(i64, f32)> = results
        .into_sorted_vec()
        .into_iter()
        .map(|c| (c.rowid, c.distance))
        .collect();

    final_results.reverse(); // Sort ascending by distance

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
        let results = search_hnsw(&db, &metadata, "test_table", "embedding", &query, 5, None).unwrap();

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
        storage::insert_node(&db, "test_table", "embedding", 1, 0, &vector).unwrap();

        // Search for it
        let query = vec![1u8, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]; // Same vector
        let results = search_hnsw(&db, &metadata, "test_table", "embedding", &query, 1, None).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
        assert!(results[0].1 < 0.001, "Distance to itself should be near zero");
    }

    #[test]
    fn test_search_candidate_ordering() {
        let mut heap = BinaryHeap::new();

        heap.push(SearchCandidate {
            rowid: 1,
            distance: 0.5,
        });
        heap.push(SearchCandidate {
            rowid: 2,
            distance: 0.3,
        });
        heap.push(SearchCandidate {
            rowid: 3,
            distance: 0.7,
        });

        // Should pop in order: 0.3, 0.5, 0.7 (min-heap)
        let first = heap.pop().unwrap();
        assert_eq!(first.rowid, 2);
        assert!((first.distance - 0.3).abs() < 0.001);

        let second = heap.pop().unwrap();
        assert_eq!(second.rowid, 1);
        assert!((second.distance - 0.5).abs() < 0.001);
    }
}
