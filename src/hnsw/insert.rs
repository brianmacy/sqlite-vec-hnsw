//! HNSW insert algorithm implementation
//!
//! Implements the page-cache based insert algorithm that persists nodes and edges
//! to shadow tables while maintaining HNSW graph invariants.

use crate::distance;
use crate::error::{Error, Result};
use crate::hnsw::{HnswMetadata, search, storage};
use crate::vector::Vector;
use rusqlite::Connection;

/// Generate a random level for a new node
///
/// Uses exponential decay: level = floor(-ln(uniform_random()) * level_factor)
fn generate_level(metadata: &HnswMetadata) -> i32 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    // Generate pseudo-random number based on current metadata state
    let random_state = RandomState::new();
    let mut hasher = random_state.build_hasher();
    hasher.write_u32(metadata.rng_seed);
    hasher.write_i32(metadata.num_nodes);
    hasher.write_u64(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64,
    );

    let random_val = (hasher.finish() % 1_000_000) as f64 / 1_000_000.0; // [0, 1)

    // Avoid log(0)
    let random_val = random_val.max(1e-9);

    let level = (-random_val.ln() * metadata.params.level_factor).floor() as i32;
    level.min(metadata.params.max_level - 1).max(0)
}

/// Prune edges using RNG (Relative Neighborhood Graph) heuristic
///
/// Based on HNSWlib's getNeighborsByHeuristic2() algorithm.
/// Promotes diversity by rejecting candidates that are closer to already-selected
/// neighbors than to the center point. This prevents dense graphs and hub nodes.
///
/// # Arguments
/// * `candidates` - Candidate neighbors with distances to center
/// * `max_connections` - Maximum number of neighbors to select
/// * `center_vector` - Optional center vector for distance calculations (if None, falls back to simple pruning)
/// * `candidate_vectors` - Optional map of candidate vectors (avoids re-fetching)
/// * `metadata` - HNSW metadata for distance calculations
///
/// # Returns
/// Selected neighbors (may be fewer than max_connections)
fn prune_edges_with_heuristic(
    candidates: &[(i64, f32)],
    max_connections: usize,
    center_vector: Option<&Vector>,
    candidate_vectors: Option<&std::collections::HashMap<i64, Vector>>,
    metadata: Option<&HnswMetadata>,
) -> Vec<(i64, f32)> {
    // If no center vector or metadata provided, fall back to simple pruning
    if center_vector.is_none() || candidate_vectors.is_none() || metadata.is_none() {
        let mut sorted = candidates.to_vec();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(max_connections);
        return sorted;
    }

    let _center = center_vector.unwrap();
    let vectors = candidate_vectors.unwrap();
    let meta = metadata.unwrap();

    // If we have fewer candidates than max, return all
    if candidates.len() <= max_connections {
        return candidates.to_vec();
    }

    // Sort candidates by distance to center (closest first)
    let mut sorted = candidates.to_vec();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut selected = Vec::new();

    // RNG heuristic: process candidates from closest to farthest
    for (candidate_rowid, dist_to_center) in sorted.iter() {
        if selected.len() >= max_connections {
            break;
        }

        // Get candidate vector
        let candidate_vec = match vectors.get(candidate_rowid) {
            Some(v) => v,
            None => continue, // Skip if vector not available
        };

        // Check distance to all already-selected neighbors
        let mut accept = true;
        for (selected_rowid, _) in selected.iter() {
            if let Some(selected_vec) = vectors.get(selected_rowid) {
                // Calculate distance from candidate to this selected neighbor
                if let Ok(dist_to_selected) =
                    distance::distance(candidate_vec, selected_vec, meta.distance_metric)
                {
                    // RNG heuristic: reject if closer to selected neighbor than to center
                    if dist_to_selected < *dist_to_center {
                        accept = false;
                        break;
                    }
                }
            }
        }

        if accept {
            selected.push((*candidate_rowid, *dist_to_center));
        }
    }

    selected
}


/// Insert a vector into the HNSW index
///
/// # Arguments
/// * `db` - Database connection
/// * `metadata` - HNSW index metadata (will be updated)
/// * `table_name` - Virtual table name
/// * `column_name` - Vector column name
/// * `rowid` - Rowid for the new vector
/// * `vector` - Vector data as bytes
///
/// # Algorithm
/// 1. Generate random level for new node
/// 2. Insert node into _hnsw_nodes shadow table
/// 3. If first node, set as entry point
/// 4. Otherwise, find nearest neighbors at each level
/// 5. Create bidirectional edges
/// 6. Prune edges to maintain M connections
/// 7. Update entry point if new node has higher level
/// Cached prepared statements for HNSW operations
pub struct HnswStmtCache {
    pub get_node_data: *mut rusqlite::ffi::sqlite3_stmt,
    pub get_edges_with_dist: *mut rusqlite::ffi::sqlite3_stmt,
    pub insert_node: *mut rusqlite::ffi::sqlite3_stmt,
    pub insert_edge: *mut rusqlite::ffi::sqlite3_stmt,
    pub delete_edges_from: *mut rusqlite::ffi::sqlite3_stmt,
}

pub fn insert_hnsw(
    db: &Connection,
    metadata: &mut HnswMetadata,
    table_name: &str,
    column_name: &str,
    rowid: i64,
    vector: &[u8],
    stmt_cache: Option<&HnswStmtCache>,
) -> Result<()> {
    // Generate level for new node
    let level = generate_level(metadata);

    // Insert node into shadow table
    let insert_node_stmt = stmt_cache.map(|c| c.insert_node);
    storage::insert_node(db, table_name, column_name, rowid, level, vector, insert_node_stmt)?;

    // Handle first node case
    if metadata.entry_point_rowid == -1 {
        metadata.entry_point_rowid = rowid;
        metadata.entry_point_level = level;
        metadata.num_nodes = 1;
        metadata.hnsw_version += 1;
        metadata.save_to_db(db, table_name, column_name)?;
        return Ok(());
    }

    // Parse the vector for distance calculations
    let new_vec = Vector::from_blob(vector, metadata.element_type, metadata.dimensions as usize)?;

    // Find insertion points by searching from entry point down
    let mut current_nearest = metadata.entry_point_rowid;

    // Traverse from top level down to insertion level + 1 (greedy search)
    for lv in (level + 1..=metadata.entry_point_level).rev() {
        let nearest = find_closest_at_level(
            db,
            metadata,
            table_name,
            column_name,
            &new_vec,
            current_nearest,
            lv,
            stmt_cache,
        )?;
        current_nearest = nearest;
    }

    // Create search context for neighbor finding
    let ctx = search::SearchContext {
        db,
        metadata,
        table_name,
        column_name,
        query_vec: &new_vec,
    };

    // Insert at each level from insertion level down to 0
    for lv in (0..=level).rev() {
        // Find M nearest neighbors at this level
        let neighbors = search::search_layer(
            &ctx,
            current_nearest,
            metadata.params.ef_construction as usize,
            lv,
        )?;

        // Determine max connections for this level
        let max_connections = if lv == 0 {
            metadata.params.max_m0 as usize
        } else {
            metadata.params.m as usize
        };

        // Fetch candidate vectors for RNG heuristic pruning
        let get_node_stmt = stmt_cache.map(|c| c.get_node_data);
        let mut candidate_vectors = std::collections::HashMap::new();
        for (candidate_rowid, _) in neighbors.iter() {
            if let Some(node) = storage::fetch_node_data(db, table_name, column_name, *candidate_rowid, get_node_stmt)? {
                if let Ok(vec) = Vector::from_blob(&node.vector, metadata.element_type, metadata.dimensions as usize) {
                    candidate_vectors.insert(*candidate_rowid, vec);
                }
            }
        }

        // Prune using RNG heuristic (promotes diversity)
        let pruned = prune_edges_with_heuristic(
            &neighbors,
            max_connections,
            Some(&new_vec),
            Some(&candidate_vectors),
            Some(metadata),
        );

        // Create bidirectional edges
        let insert_edge_stmt = stmt_cache.map(|c| c.insert_edge);
        for (neighbor_rowid, dist) in pruned.iter() {
            // Edge from new node to neighbor
            storage::insert_edge(
                db,
                table_name,
                column_name,
                rowid,
                *neighbor_rowid,
                lv,
                *dist,
                insert_edge_stmt,
            )?;

            // Edge from neighbor to new node
            storage::insert_edge(
                db,
                table_name,
                column_name,
                *neighbor_rowid,
                rowid,
                lv,
                *dist,
                insert_edge_stmt,
            )?;

            // Check if neighbor now has too many connections, prune if needed
            // Fetch edges WITH distances to avoid recalculating
            let get_edges_stmt = stmt_cache.map(|c| c.get_edges_with_dist);
            let mut neighbor_edges_with_dist = storage::fetch_neighbors_with_distances(
                db,
                table_name,
                column_name,
                *neighbor_rowid,
                lv,
                get_edges_stmt,
            )?;

            if neighbor_edges_with_dist.len() >= max_connections {
                // Need to prune neighbor's edges using RNG heuristic
                // Fetch neighbor's vector (the center for pruning)
                if let Some(neighbor_node) = storage::fetch_node_data(db, table_name, column_name, *neighbor_rowid, get_node_stmt)? {
                    if let Ok(neighbor_vec) = Vector::from_blob(&neighbor_node.vector, metadata.element_type, metadata.dimensions as usize) {
                        // Add the new edge if not already present
                        if !neighbor_edges_with_dist.iter().any(|(r, _)| *r == rowid) {
                            neighbor_edges_with_dist.push((rowid, *dist));
                        }

                        // Fetch vectors for all candidates
                        let mut neighbor_candidate_vectors = std::collections::HashMap::new();
                        neighbor_candidate_vectors.insert(rowid, new_vec.clone()); // New node vector

                        for (cand_rowid, _) in neighbor_edges_with_dist.iter() {
                            if *cand_rowid != rowid {
                                if let Some(cand_node) = storage::fetch_node_data(db, table_name, column_name, *cand_rowid, get_node_stmt)? {
                                    if let Ok(cand_vec) = Vector::from_blob(&cand_node.vector, metadata.element_type, metadata.dimensions as usize) {
                                        neighbor_candidate_vectors.insert(*cand_rowid, cand_vec);
                                    }
                                }
                            }
                        }

                        // Prune using RNG heuristic with neighbor as center
                        let pruned_neighbor = prune_edges_with_heuristic(
                            &neighbor_edges_with_dist,
                            max_connections,
                            Some(&neighbor_vec),
                            Some(&neighbor_candidate_vectors),
                            Some(metadata),
                        );

                        // Rebuild neighbor's edges
                        let delete_edges_stmt = stmt_cache.map(|c| c.delete_edges_from);
                        storage::delete_edges_from_level(db, table_name, column_name, *neighbor_rowid, lv, delete_edges_stmt)?;
                        for (ne_rowid, ne_dist) in pruned_neighbor {
                            storage::insert_edge(
                                db,
                                table_name,
                                column_name,
                                *neighbor_rowid,
                                ne_rowid,
                                lv,
                                ne_dist,
                                insert_edge_stmt,
                            )?;
                        }
                    }
                }
            }
        }

        // Update current_nearest for next level
        if let Some((nearest, _)) = pruned.first() {
            current_nearest = *nearest;
        }
    }

    // Update entry point if new node has higher level
    if level > metadata.entry_point_level {
        metadata.entry_point_rowid = rowid;
        metadata.entry_point_level = level;
    }

    // Update metadata
    metadata.num_nodes += 1;
    metadata.hnsw_version += 1;
    metadata.save_to_db(db, table_name, column_name)?;

    Ok(())
}

/// Find the closest node to the query at a specific level (greedy search)
fn find_closest_at_level(
    db: &Connection,
    metadata: &HnswMetadata,
    table_name: &str,
    column_name: &str,
    query_vec: &Vector,
    start_rowid: i64,
    level: i32,
    stmt_cache: Option<&HnswStmtCache>,
) -> Result<i64> {
    let mut current = start_rowid;
    let mut changed = true;

    let get_node_stmt = stmt_cache.map(|c| c.get_node_data);

    while changed {
        changed = false;

        // Get current node's distance
        let current_node = storage::fetch_node_data(db, table_name, column_name, current, get_node_stmt)?
            .ok_or_else(|| Error::InvalidParameter(format!("Node {} not found", current)))?;

        let current_vec = Vector::from_blob(
            &current_node.vector,
            metadata.element_type,
            metadata.dimensions as usize,
        )?;
        let current_dist = distance::distance(query_vec, &current_vec, metadata.distance_metric)?;

        // Check all neighbors
        let neighbors = storage::fetch_neighbors(db, table_name, column_name, current, level)?;

        for neighbor_rowid in neighbors {
            let neighbor_node =
                storage::fetch_node_data(db, table_name, column_name, neighbor_rowid, get_node_stmt)?;
            let neighbor_node = match neighbor_node {
                Some(n) => n,
                None => continue,
            };

            let neighbor_vec = Vector::from_blob(
                &neighbor_node.vector,
                metadata.element_type,
                metadata.dimensions as usize,
            )?;
            let neighbor_dist =
                distance::distance(query_vec, &neighbor_vec, metadata.distance_metric)?;

            if neighbor_dist < current_dist {
                current = neighbor_rowid;
                changed = true;
                break;
            }
        }
    }

    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::DistanceMetric;
    use crate::shadow;
    use crate::vector::VectorType;

    #[test]
    fn test_generate_level() {
        let metadata = HnswMetadata::new(128, VectorType::Float32, DistanceMetric::L2);

        // Generate multiple levels to test distribution
        let mut levels = Vec::new();
        for i in 0..100 {
            let mut meta = metadata.clone();
            meta.num_nodes = i; // Vary state for different random values
            let level = generate_level(&meta);
            levels.push(level);
        }

        // Most levels should be 0 (exponential decay)
        let level_0_count = levels.iter().filter(|&&l| l == 0).count();
        assert!(level_0_count > 50, "Most nodes should be at level 0");

        // All levels should be valid
        for level in levels {
            assert!((0..16).contains(&level));
        }
    }

    #[test]
    fn test_prune_edges_keeps_closest() {
        let candidates = vec![(1, 0.5), (2, 0.3), (3, 0.7), (4, 0.2), (5, 0.9)];

        // Test with None (falls back to simple pruning)
        let pruned = prune_edges_with_heuristic(&candidates, 3, None, None, None);

        assert_eq!(pruned.len(), 3);
        // Should keep the 3 closest: (4, 0.2), (2, 0.3), (1, 0.5)
        assert_eq!(pruned[0].0, 4);
        assert_eq!(pruned[1].0, 2);
        assert_eq!(pruned[2].0, 1);
    }

    #[test]
    fn test_insert_first_node() {
        let db = Connection::open_in_memory().unwrap();
        shadow::create_hnsw_shadow_tables(&db, "test_table", "embedding").unwrap();

        let mut metadata = HnswMetadata::new(3, VectorType::Float32, DistanceMetric::L2);

        // Insert first vector
        let vector = vec![1u8, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]; // [1.0, 2.0, 3.0]
        insert_hnsw(&db, &mut metadata, "test_table", "embedding", 1, &vector, None).unwrap();

        // Verify metadata updated
        assert_eq!(metadata.entry_point_rowid, 1);
        assert_eq!(metadata.num_nodes, 1);
        assert!(metadata.entry_point_level >= 0);

        // Verify node was persisted
        let node = storage::fetch_node_data(&db, "test_table", "embedding", 1, None)
            .unwrap()
            .expect("Node should exist");

        assert_eq!(node.rowid, 1);
        assert_eq!(node.vector, vector);
    }

    #[test]
    fn test_insert_multiple_nodes() {
        let db = Connection::open_in_memory().unwrap();
        shadow::create_hnsw_shadow_tables(&db, "test_table", "embedding").unwrap();

        let mut metadata = HnswMetadata::new(3, VectorType::Float32, DistanceMetric::L2);

        // Insert multiple vectors
        for i in 1..=5 {
            let vector = vec![
                i as u8,
                0,
                0,
                0,
                (i + 1) as u8,
                0,
                0,
                0,
                (i + 2) as u8,
                0,
                0,
                0,
            ];
            insert_hnsw(&db, &mut metadata, "test_table", "embedding", i, &vector, None).unwrap();
        }

        // Verify all nodes inserted
        assert_eq!(metadata.num_nodes, 5);

        let count = storage::count_nodes(&db, "test_table", "embedding").unwrap();
        assert_eq!(count, 5);

        // Verify entry point is set
        assert!(metadata.entry_point_rowid > 0);
    }

    #[test]
    fn test_insert_creates_edges() {
        let db = Connection::open_in_memory().unwrap();
        shadow::create_hnsw_shadow_tables(&db, "test_table", "embedding").unwrap();

        let mut metadata = HnswMetadata::new(3, VectorType::Float32, DistanceMetric::L2);

        // Insert a few vectors
        for i in 1..=3 {
            let vector = vec![i as u8, 0, 0, 0, i as u8, 0, 0, 0, i as u8, 0, 0, 0];
            insert_hnsw(&db, &mut metadata, "test_table", "embedding", i, &vector, None).unwrap();
        }

        // Check that edges were created (at least for some nodes)
        let edges_count: i64 = db
            .query_row(
                "SELECT COUNT(*) FROM test_table_embedding_hnsw_edges",
                [],
                |row| row.get(0),
            )
            .unwrap();

        println!("Edges created: {}", edges_count);
        // After 3 inserts, should have some edges (bidirectional)
        assert!(edges_count > 0, "Should have created some edges");
    }
}
