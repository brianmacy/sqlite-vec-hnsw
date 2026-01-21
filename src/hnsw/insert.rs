//! HNSW insert algorithm implementation
//!
//! Implements the page-cache based insert algorithm that persists nodes and edges
//! to shadow tables while maintaining HNSW graph invariants.

use crate::error::Result;
use crate::hnsw::{HnswMetadata, search, storage};
use crate::vector::Vector;
use rusqlite::Connection;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// Global timing counters
static VALIDATE_TIME: AtomicU64 = AtomicU64::new(0);
static SEARCH_LAYER_TIME: AtomicU64 = AtomicU64::new(0);
static INSERT_EDGE_TIME: AtomicU64 = AtomicU64::new(0);
static PRUNE_TIME: AtomicU64 = AtomicU64::new(0);
static PRUNE_FETCH_TIME: AtomicU64 = AtomicU64::new(0);
static PRUNE_DELETE_TIME: AtomicU64 = AtomicU64::new(0);
static PRUNE_REINSERT_TIME: AtomicU64 = AtomicU64::new(0);
static METADATA_TIME: AtomicU64 = AtomicU64::new(0);
static PRUNE_CALLS: AtomicU64 = AtomicU64::new(0);
static PRUNE_EARLY_RETURNS: AtomicU64 = AtomicU64::new(0);
static PRUNE_ACTUAL_PRUNES: AtomicU64 = AtomicU64::new(0);
static PRUNE_EDGES_DELETED: AtomicU64 = AtomicU64::new(0);
static PRUNE_EDGES_REINSERTED: AtomicU64 = AtomicU64::new(0);

pub fn print_timing_stats() {
    let prune_actual = PRUNE_ACTUAL_PRUNES.load(Ordering::Relaxed);
    let edges_deleted = PRUNE_EDGES_DELETED.load(Ordering::Relaxed);
    let edges_reinserted = PRUNE_EDGES_REINSERTED.load(Ordering::Relaxed);

    eprintln!("\n=== INSERT TIMING BREAKDOWN ===");
    eprintln!(
        "  validate:     {}ms",
        VALIDATE_TIME.load(Ordering::Relaxed) / 1000
    );
    eprintln!(
        "  search_layer: {}ms",
        SEARCH_LAYER_TIME.load(Ordering::Relaxed) / 1000
    );
    eprintln!(
        "  insert_edge:  {}ms",
        INSERT_EDGE_TIME.load(Ordering::Relaxed) / 1000
    );
    eprintln!(
        "  prune:        {}ms",
        PRUNE_TIME.load(Ordering::Relaxed) / 1000
    );
    eprintln!("    - calls:    {}", PRUNE_CALLS.load(Ordering::Relaxed));
    eprintln!(
        "    - early:    {}",
        PRUNE_EARLY_RETURNS.load(Ordering::Relaxed)
    );
    eprintln!("    - actual:   {} prunes", prune_actual);
    eprintln!("    - deleted:  {} edges total", edges_deleted);
    eprintln!("    - reinsert: {} edges total", edges_reinserted);
    if prune_actual > 0 {
        eprintln!(
            "    - avg edges/prune: {:.1}",
            edges_deleted as f64 / prune_actual as f64
        );
    }
    eprintln!(
        "    - fetch:    {}ms",
        PRUNE_FETCH_TIME.load(Ordering::Relaxed) / 1000
    );
    eprintln!(
        "    - delete:   {}ms",
        PRUNE_DELETE_TIME.load(Ordering::Relaxed) / 1000
    );
    eprintln!(
        "    - reinsert: {}ms",
        PRUNE_REINSERT_TIME.load(Ordering::Relaxed) / 1000
    );
    eprintln!(
        "  metadata:     {}ms",
        METADATA_TIME.load(Ordering::Relaxed) / 1000
    );
}

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

/// Prune neighbor's connections if they exceed max_connections using RNG heuristic
///
/// Implements the HNSWlib RNG (Relative Neighborhood Graph) heuristic:
/// - Sort candidates by distance to center (the neighbor being pruned)
/// - For each candidate, reject if it's closer to any already-selected neighbor than to center
/// - This ensures diverse, spread-out connections for better recall
#[allow(clippy::too_many_arguments)]
fn prune_neighbor_if_needed(
    db: &Connection,
    table_name: &str,
    column_name: &str,
    neighbor_rowid: i64,
    _new_node_rowid: i64,
    _new_edge_dist: f32,
    level: i32,
    max_connections: usize,
    metadata: &HnswMetadata,
    _new_vec: &Vector,
    stmt_cache: Option<&HnswStmtCache>,
    get_edges_stmt: Option<*mut rusqlite::ffi::sqlite3_stmt>,
    insert_edge_stmt: Option<*mut rusqlite::ffi::sqlite3_stmt>,
) -> Result<()> {
    PRUNE_CALLS.fetch_add(1, Ordering::Relaxed);

    // Fetch current neighbor edges (just rowids, no distances)
    let t = Instant::now();
    let get_edges_cached = get_edges_stmt;
    let neighbor_edges = storage::fetch_neighbors_cached(
        db,
        table_name,
        column_name,
        neighbor_rowid,
        level,
        get_edges_cached,
    )?;
    PRUNE_FETCH_TIME.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);

    // Early return if under limit - no pruning needed (matches C behavior)
    // C uses < (strictly less than), not <=
    if neighbor_edges.len() < max_connections {
        PRUNE_EARLY_RETURNS.fetch_add(1, Ordering::Relaxed);
        return Ok(());
    }

    PRUNE_ACTUAL_PRUNES.fetch_add(1, Ordering::Relaxed);
    PRUNE_EDGES_DELETED.fetch_add(neighbor_edges.len() as u64, Ordering::Relaxed);

    // Fetch the center vector (the neighbor being pruned)
    let get_node_stmt = stmt_cache.map(|c| c.get_node_data);
    let center_node =
        storage::fetch_node_data(db, table_name, column_name, neighbor_rowid, get_node_stmt)?
            .ok_or_else(|| {
                crate::error::Error::Hnsw(format!("Node {} not found during prune", neighbor_rowid))
            })?;
    let center_vec = Vector::from_blob(
        &center_node.vector,
        metadata.element_type,
        metadata.dimensions as usize,
    )?;

    // Fetch vectors for all candidate edges and compute distances to center
    let mut candidates: Vec<(i64, f32, Vec<u8>)> = Vec::with_capacity(neighbor_edges.len());
    for &edge_rowid in &neighbor_edges {
        if let Ok(Some(node)) =
            storage::fetch_node_data(db, table_name, column_name, edge_rowid, get_node_stmt)
        {
            let candidate_vec = Vector::from_blob(
                &node.vector,
                metadata.element_type,
                metadata.dimensions as usize,
            )?;
            let dist =
                crate::distance::distance(&center_vec, &candidate_vec, metadata.distance_metric)?;
            candidates.push((edge_rowid, dist, node.vector));
        }
    }

    // Sort by distance to center (closest first)
    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Apply RNG heuristic: select diverse neighbors
    let mut selected: Vec<i64> = Vec::with_capacity(max_connections);
    let mut selected_vecs: Vec<Vector> = Vec::with_capacity(max_connections);

    for (rowid, dist_to_center, blob) in candidates {
        if selected.len() >= max_connections {
            break;
        }

        let candidate_vec =
            Vector::from_blob(&blob, metadata.element_type, metadata.dimensions as usize)?;

        // RNG heuristic: reject if candidate is closer to any already-selected than to center
        let mut good = true;
        for selected_vec in &selected_vecs {
            let dist_to_selected =
                crate::distance::distance(&candidate_vec, selected_vec, metadata.distance_metric)?;
            if dist_to_selected < dist_to_center {
                good = false;
                break;
            }
        }

        if good {
            selected.push(rowid);
            selected_vecs.push(candidate_vec);
        }
    }

    PRUNE_EDGES_REINSERTED.fetch_add(selected.len() as u64, Ordering::Relaxed);

    // Delete all edges at this level
    let t = Instant::now();
    let delete_stmt = stmt_cache.map(|c| c.delete_edges_from);
    storage::delete_edges_from_level(
        db,
        table_name,
        column_name,
        neighbor_rowid,
        level,
        delete_stmt,
    )?;
    PRUNE_DELETE_TIME.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);

    // Re-insert only the selected diverse neighbors
    let t = Instant::now();
    for to_rowid in &selected {
        storage::insert_edge(
            db,
            table_name,
            column_name,
            neighbor_rowid,
            *to_rowid,
            level,
            insert_edge_stmt,
        )?;
    }
    PRUNE_REINSERT_TIME.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);

    Ok(())
}

/// Cached prepared statements for HNSW operations
pub struct HnswStmtCache {
    pub get_node_data: *mut rusqlite::ffi::sqlite3_stmt,
    pub get_edges: *mut rusqlite::ffi::sqlite3_stmt,
    pub insert_node: *mut rusqlite::ffi::sqlite3_stmt,
    pub insert_edge: *mut rusqlite::ffi::sqlite3_stmt,
    pub delete_edges_from: *mut rusqlite::ffi::sqlite3_stmt,
    pub update_meta: *mut rusqlite::ffi::sqlite3_stmt,
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
#[allow(clippy::collapsible_if)]
pub fn insert_hnsw(
    db: &Connection,
    metadata: &mut HnswMetadata,
    table_name: &str,
    column_name: &str,
    rowid: i64,
    vector: &[u8],
    stmt_cache: Option<&HnswStmtCache>,
) -> Result<()> {
    // CRITICAL: Validate metadata before insert (multi-connection safety)
    // This ensures our metadata is current and reloads if another connection modified the index
    let t = Instant::now();
    metadata.validate_and_refresh(db, table_name, column_name)?;
    VALIDATE_TIME.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);

    // Generate level for new node
    let level = generate_level(metadata);

    // Insert node into shadow table
    let insert_node_stmt = stmt_cache.map(|c| c.insert_node);
    storage::insert_node(
        db,
        table_name,
        column_name,
        rowid,
        level,
        vector,
        insert_node_stmt,
    )?;

    // Handle first node case
    if metadata.entry_point_rowid == -1 {
        metadata.entry_point_rowid = rowid;
        metadata.entry_point_level = level;
        metadata.num_nodes = 1;
        metadata.hnsw_version += 1;

        // Use cached statement if available, otherwise fallback
        if let Some(cache) = stmt_cache {
            unsafe {
                metadata.save_dynamic_to_db_cached(cache.update_meta, true)?;
            }
        } else {
            metadata.save_dynamic_to_db(db, table_name, column_name)?;
        }
        return Ok(());
    }

    // Parse the vector for distance calculations
    let new_vec = Vector::from_blob(vector, metadata.element_type, metadata.dimensions as usize)?;

    // Find insertion points by searching from entry point down
    let mut current_nearest = metadata.entry_point_rowid;

    // Create search statement cache for all search operations
    let search_stmt_cache = stmt_cache.map(|c| search::SearchStmtCache {
        get_node_data: Some(c.get_node_data),
        get_edges: Some(c.get_edges),
    });
    let search_stmt_cache_ref = search_stmt_cache.as_ref();

    // Create search context for all search operations
    let ctx = search::SearchContext {
        db,
        metadata,
        table_name,
        column_name,
        query_vec: &new_vec,
        stmt_cache: search_stmt_cache_ref,
    };

    // Traverse from top level down to insertion level + 1 (greedy search with ef=1)
    // This matches C implementation's hnsw_search_layer_query with ef=1
    for lv in (level + 1..=metadata.entry_point_level).rev() {
        let t = Instant::now();
        let results = search::search_layer(&ctx, current_nearest, 1, lv)?;
        SEARCH_LAYER_TIME.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        if let Some((nearest_rowid, _dist)) = results.first() {
            current_nearest = *nearest_rowid;
        }
    }

    // Insert at each level from insertion level down to 0
    for lv in (0..=level).rev() {
        // Find M nearest neighbors at this level
        let t = Instant::now();
        let neighbors = search::search_layer(
            &ctx,
            current_nearest,
            metadata.params.ef_construction as usize,
            lv,
        )?;
        SEARCH_LAYER_TIME.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);

        // Determine max connections for this level
        let max_connections = if lv == 0 {
            metadata.params.max_m0 as usize
        } else {
            metadata.params.m as usize
        };

        // Select M closest neighbors from search results (already sorted by distance)
        // No need to re-fetch vectors - distances are computed during search
        let selected: Vec<(i64, f32)> = neighbors.into_iter().take(max_connections).collect();

        // Insert bidirectional edges and prune (matches C: single loop)
        let insert_edge_stmt = stmt_cache.map(|c| c.insert_edge);
        let get_edges_stmt = stmt_cache.map(|c| c.get_edges);
        let _delete_edges_stmt = stmt_cache.map(|c| c.delete_edges_from);

        for (neighbor_rowid, _dist) in selected.iter() {
            // Edge: new_node -> neighbor
            let t = Instant::now();
            storage::insert_edge(
                db,
                table_name,
                column_name,
                rowid,
                *neighbor_rowid,
                lv,
                insert_edge_stmt,
            )?;

            // Edge: neighbor -> new_node (bidirectional)
            storage::insert_edge(
                db,
                table_name,
                column_name,
                *neighbor_rowid,
                rowid,
                lv,
                insert_edge_stmt,
            )?;
            INSERT_EDGE_TIME.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);

            // Prune neighbor immediately after inserting (matches C)
            let t = Instant::now();
            prune_neighbor_if_needed(
                db,
                table_name,
                column_name,
                *neighbor_rowid,
                rowid,
                0.0,
                lv,
                max_connections,
                metadata,
                &new_vec,
                stmt_cache,
                get_edges_stmt,
                insert_edge_stmt,
            )?;
            PRUNE_TIME.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        }

        // Update current_nearest for next level
        if let Some((nearest, _)) = selected.first() {
            current_nearest = *nearest;
        }
    }

    // Update entry point if new node has higher level
    let entry_point_changed = level > metadata.entry_point_level;
    if entry_point_changed {
        metadata.entry_point_rowid = rowid;
        metadata.entry_point_level = level;
    }

    // Update metadata (always increment num_nodes and version)
    metadata.num_nodes += 1;
    metadata.hnsw_version += 1;

    // Use cached statement if available (FAST PATH - like C)
    let t = Instant::now();
    if let Some(cache) = stmt_cache {
        unsafe {
            metadata.save_dynamic_to_db_cached(cache.update_meta, entry_point_changed)?;
        }
    } else {
        // Fallback: slower path for tests without cache
        metadata.save_dynamic_to_db(db, table_name, column_name)?;
    }
    METADATA_TIME.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);

    Ok(())
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
        // Test that simple pruning keeps closest neighbors
        let mut candidates = vec![(1, 0.5), (2, 0.3), (3, 0.7), (4, 0.2), (5, 0.9)];
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let pruned: Vec<_> = candidates.into_iter().take(3).collect();

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
        insert_hnsw(
            &db,
            &mut metadata,
            "test_table",
            "embedding",
            1,
            &vector,
            None,
        )
        .unwrap();

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
            insert_hnsw(
                &db,
                &mut metadata,
                "test_table",
                "embedding",
                i,
                &vector,
                None,
            )
            .unwrap();
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
            insert_hnsw(
                &db,
                &mut metadata,
                "test_table",
                "embedding",
                i,
                &vector,
                None,
            )
            .unwrap();
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
