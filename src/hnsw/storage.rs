//! Page-cache based HNSW storage operations
//!
//! This module implements HNSW node and edge operations that query shadow tables
//! instead of keeping the entire graph in memory. This enables scaling to millions
//! of vectors while keeping memory usage minimal (~64 bytes per index).
//!
//! All functions that use cached statements employ `StmtHandleGuard` for RAII-based
//! statement reset. This ensures shared-cache locks are released even on error paths.

use crate::error::{Error, Result};
use crate::vtab::StmtHandleGuard;
use rusqlite::{Connection, OptionalExtension, ffi};

/// HNSW node data
#[derive(Debug, Clone)]
pub struct HnswNode {
    pub rowid: i64,
    pub level: i32,
    pub vector: Vec<u8>,
}

/// Fetch node data from shadow table
///
/// # Arguments
/// * `db` - Database connection
/// * `table_name` - Virtual table name
/// * `column_name` - Vector column name
/// * `rowid` - Node rowid to fetch
///
/// # Returns
/// Node data if found, None otherwise
pub fn fetch_node_data(
    db: &Connection,
    table_name: &str,
    column_name: &str,
    rowid: i64,
    cached_stmt: Option<*mut ffi::sqlite3_stmt>,
) -> Result<Option<HnswNode>> {
    // Fast path: use cached statement (avoids SQL parsing)
    if let Some(stmt) = cached_stmt {
        unsafe {
            // RAII guard: resets on create (clears previous state) and on drop (releases locks)
            let guard = StmtHandleGuard::new(stmt)
                .ok_or_else(|| Error::InvalidParameter("null statement".to_string()))?;

            ffi::sqlite3_bind_int64(guard.as_ptr(), 1, rowid);

            let rc = ffi::sqlite3_step(guard.as_ptr());
            if rc == ffi::SQLITE_ROW {
                let node_rowid = ffi::sqlite3_column_int64(guard.as_ptr(), 0);
                let level = ffi::sqlite3_column_int(guard.as_ptr(), 1);

                let blob_ptr = ffi::sqlite3_column_blob(guard.as_ptr(), 2);
                let blob_len = ffi::sqlite3_column_bytes(guard.as_ptr(), 2);
                // CRITICAL: Copy data BEFORE guard drops (blob_ptr becomes invalid after reset)
                let vector =
                    std::slice::from_raw_parts(blob_ptr as *const u8, blob_len as usize).to_vec();

                // Guard drops here, automatically resets statement
                return Ok(Some(HnswNode {
                    rowid: node_rowid,
                    level,
                    vector,
                }));
            } else if rc == ffi::SQLITE_DONE {
                // Guard drops here, automatically resets statement
                return Ok(None);
            } else {
                // Guard drops here, automatically resets statement even on error
                return Err(Error::Sqlite(rusqlite::Error::SqliteFailure(
                    rusqlite::ffi::Error::new(rc),
                    None,
                )));
            }
        }
    }

    // Slow path fallback: parse SQL every time
    let nodes_table = format!("{}_{}_hnsw_nodes", table_name, column_name);
    let query = format!(
        "SELECT rowid, level, vector FROM \"{}\" WHERE rowid = ?",
        nodes_table
    );

    db.query_row(&query, [rowid], |row| {
        Ok(HnswNode {
            rowid: row.get(0)?,
            level: row.get(1)?,
            vector: row.get(2)?,
        })
    })
    .optional()
    .map_err(Error::Sqlite)
}

/// Fetch node level only (lighter query than full node data)
pub fn fetch_node_level(
    db: &Connection,
    table_name: &str,
    column_name: &str,
    rowid: i64,
) -> Result<Option<i32>> {
    let nodes_table = format!("{}_{}_hnsw_nodes", table_name, column_name);
    let query = format!("SELECT level FROM \"{}\" WHERE rowid = ?", nodes_table);

    db.query_row(&query, [rowid], |row| row.get::<_, i32>(0))
        .optional()
        .map_err(Error::Sqlite)
}

/// Fetch neighbors of a node at a specific level
///
/// # Arguments
/// * `cached_stmt` - Optional cached prepared statement (10x faster if provided)
///
/// # Returns
/// List of neighbor rowids at the given level
pub fn fetch_neighbors_cached(
    db: &Connection,
    table_name: &str,
    column_name: &str,
    from_rowid: i64,
    level: i32,
    cached_stmt: Option<*mut ffi::sqlite3_stmt>,
) -> Result<Vec<i64>> {
    // Fast path: use cached statement (like C implementation)
    if let Some(stmt) = cached_stmt {
        unsafe {
            // RAII guard: resets on create (clears previous state) and on drop (releases locks)
            let guard = StmtHandleGuard::new(stmt)
                .ok_or_else(|| Error::InvalidParameter("null statement".to_string()))?;

            ffi::sqlite3_bind_int64(guard.as_ptr(), 1, from_rowid);
            ffi::sqlite3_bind_int(guard.as_ptr(), 2, level);

            let mut neighbors = Vec::new();
            loop {
                let rc = ffi::sqlite3_step(guard.as_ptr());
                if rc == ffi::SQLITE_ROW {
                    let to_rowid = ffi::sqlite3_column_int64(guard.as_ptr(), 0);
                    neighbors.push(to_rowid);
                } else if rc == ffi::SQLITE_DONE {
                    break;
                } else {
                    // Guard drops here, automatically resets statement
                    return Err(Error::Sqlite(rusqlite::Error::SqliteFailure(
                        rusqlite::ffi::Error::new(rc),
                        None,
                    )));
                }
            }

            // Guard drops here, automatically resets statement
            return Ok(neighbors);
        }
    }

    // Slow path fallback: prepare statement each time (only for testing/fallback)
    let edges_table = format!("{}_{}_hnsw_edges", table_name, column_name);
    let query = format!(
        "SELECT to_rowid FROM \"{}\" WHERE from_rowid = ? AND level = ?",
        edges_table
    );

    let mut stmt = db.prepare(&query).map_err(Error::Sqlite)?;
    let neighbors = stmt
        .query_map([from_rowid, level as i64], |row| row.get::<_, i64>(0))
        .map_err(Error::Sqlite)?
        .collect::<std::result::Result<Vec<i64>, _>>()
        .map_err(Error::Sqlite)?;

    Ok(neighbors)
}

/// Fetch neighbors of a node WITH distances at a specific level
///
/// # Arguments
/// * `cached_stmt` - Optional cached prepared statement (10x faster if provided)
///
/// # Returns
/// List of (neighbor_rowid, distance) tuples at the given level
pub fn fetch_neighbors_with_distances(
    db: &Connection,
    table_name: &str,
    column_name: &str,
    from_rowid: i64,
    level: i32,
    cached_stmt: Option<*mut ffi::sqlite3_stmt>,
) -> Result<Vec<(i64, f32)>> {
    // Fast path: use cached statement
    if let Some(stmt) = cached_stmt {
        unsafe {
            // RAII guard: resets on create and on drop
            let guard = StmtHandleGuard::new(stmt)
                .ok_or_else(|| Error::InvalidParameter("null statement".to_string()))?;

            ffi::sqlite3_bind_int64(guard.as_ptr(), 1, from_rowid);
            ffi::sqlite3_bind_int(guard.as_ptr(), 2, level);

            let mut neighbors = Vec::new();
            loop {
                let rc = ffi::sqlite3_step(guard.as_ptr());
                if rc == ffi::SQLITE_ROW {
                    let to_rowid = ffi::sqlite3_column_int64(guard.as_ptr(), 0);
                    let distance = ffi::sqlite3_column_double(guard.as_ptr(), 1) as f32;
                    neighbors.push((to_rowid, distance));
                } else if rc == ffi::SQLITE_DONE {
                    break;
                } else {
                    // Guard drops here, automatically resets statement
                    return Err(Error::Sqlite(rusqlite::Error::SqliteFailure(
                        rusqlite::ffi::Error::new(rc),
                        None,
                    )));
                }
            }

            // Guard drops here, automatically resets statement
            return Ok(neighbors);
        }
    }

    // Slow path fallback
    let edges_table = format!("{}_{}_hnsw_edges", table_name, column_name);
    let query = format!(
        "SELECT to_rowid, distance FROM \"{}\" WHERE from_rowid = ? AND level = ?",
        edges_table
    );

    let mut stmt = db.prepare(&query).map_err(Error::Sqlite)?;
    let neighbors = stmt
        .query_map([from_rowid, level as i64], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)? as f32))
        })
        .map_err(Error::Sqlite)?
        .collect::<std::result::Result<Vec<(i64, f32)>, _>>()
        .map_err(Error::Sqlite)?;

    Ok(neighbors)
}

/// Insert a new node into the HNSW index
pub fn insert_node(
    db: &Connection,
    table_name: &str,
    column_name: &str,
    rowid: i64,
    level: i32,
    vector: &[u8],
    cached_stmt: Option<*mut ffi::sqlite3_stmt>,
) -> Result<()> {
    // Fast path: use cached statement
    if let Some(stmt) = cached_stmt {
        unsafe {
            // RAII guard: resets on create and on drop
            let guard = StmtHandleGuard::new(stmt)
                .ok_or_else(|| Error::InvalidParameter("null statement".to_string()))?;

            ffi::sqlite3_bind_int64(guard.as_ptr(), 1, rowid);
            ffi::sqlite3_bind_int(guard.as_ptr(), 2, level);
            ffi::sqlite3_bind_blob(
                guard.as_ptr(),
                3,
                vector.as_ptr() as *const _,
                vector.len() as i32,
                ffi::SQLITE_TRANSIENT(),
            );

            let rc = ffi::sqlite3_step(guard.as_ptr());
            if rc != ffi::SQLITE_DONE {
                // Guard drops here, automatically resets statement
                return Err(Error::Sqlite(rusqlite::Error::SqliteFailure(
                    rusqlite::ffi::Error::new(rc),
                    None,
                )));
            }
            // Guard drops here, automatically resets statement
        }
    } else {
        // Slow path fallback
        let nodes_table = format!("{}_{}_hnsw_nodes", table_name, column_name);
        let insert_sql = format!(
            "INSERT OR REPLACE INTO \"{}\" (rowid, level, vector) VALUES (?, ?, ?)",
            nodes_table
        );

        db.execute(&insert_sql, rusqlite::params![rowid, level, vector])
            .map_err(Error::Sqlite)?;
    }

    Ok(())
}

/// Insert an edge between two nodes at a specific level
pub fn insert_edge(
    db: &Connection,
    table_name: &str,
    column_name: &str,
    from_rowid: i64,
    to_rowid: i64,
    level: i32,
    cached_stmt: Option<*mut ffi::sqlite3_stmt>,
) -> Result<()> {
    // Fast path: use cached statement
    if let Some(stmt) = cached_stmt {
        unsafe {
            // RAII guard: resets on create and on drop
            let guard = StmtHandleGuard::new(stmt)
                .ok_or_else(|| Error::InvalidParameter("null statement".to_string()))?;

            ffi::sqlite3_bind_int64(guard.as_ptr(), 1, from_rowid);
            ffi::sqlite3_bind_int64(guard.as_ptr(), 2, to_rowid);
            ffi::sqlite3_bind_int(guard.as_ptr(), 3, level);

            let rc = ffi::sqlite3_step(guard.as_ptr());
            if rc != ffi::SQLITE_DONE {
                // Guard drops here, automatically resets statement
                return Err(Error::Sqlite(rusqlite::Error::SqliteFailure(
                    rusqlite::ffi::Error::new(rc),
                    None,
                )));
            }
            // Guard drops here, automatically resets statement
        }
    } else {
        // Slow path fallback
        let edges_table = format!("{}_{}_hnsw_edges", table_name, column_name);
        let insert_sql = format!(
            "INSERT OR IGNORE INTO \"{}\" (from_rowid, to_rowid, level) VALUES (?, ?, ?)",
            edges_table
        );

        db.execute(&insert_sql, rusqlite::params![from_rowid, to_rowid, level])
            .map_err(Error::Sqlite)?;
    }

    Ok(())
}

/// Insert multiple edges in a single batch operation
/// Uses multi-row INSERT for efficiency: INSERT INTO edges VALUES (?,?,?,?),(?,?,?,?),...
/// Now includes distance for O(1) prune operations
pub fn insert_edges_batch(
    db: &Connection,
    table_name: &str,
    column_name: &str,
    edges: &[(i64, i64, i32, f32)], // (from_rowid, to_rowid, level, distance)
) -> Result<()> {
    if edges.is_empty() {
        return Ok(());
    }

    let edges_table = format!("{}_{}_hnsw_edges", table_name, column_name);

    // Build multi-row INSERT: INSERT OR IGNORE INTO edges (from_rowid, to_rowid, level, distance) VALUES (?,?,?,?),(?,?,?,?),...
    // SQLite supports up to 999 parameters, so batch in chunks of 249 edges (4 params each)
    const MAX_EDGES_PER_BATCH: usize = 249;

    for chunk in edges.chunks(MAX_EDGES_PER_BATCH) {
        let placeholders: Vec<&str> = (0..chunk.len()).map(|_| "(?,?,?,?)").collect();
        let sql = format!(
            "INSERT OR IGNORE INTO \"{}\" (from_rowid, to_rowid, level, distance) VALUES {}",
            edges_table,
            placeholders.join(",")
        );

        let mut params: Vec<rusqlite::types::Value> = Vec::with_capacity(chunk.len() * 4);
        for (from_rowid, to_rowid, level, distance) in chunk {
            params.push((*from_rowid).into());
            params.push((*to_rowid).into());
            params.push((*level as i64).into());
            params.push((*distance as f64).into());
        }

        db.execute(&sql, rusqlite::params_from_iter(params))
            .map_err(Error::Sqlite)?;
    }

    Ok(())
}

/// Fetch multiple nodes in a single batch operation
/// Uses SELECT ... WHERE rowid IN (?, ?, ...) for efficiency
pub fn fetch_nodes_batch(
    db: &Connection,
    table_name: &str,
    column_name: &str,
    rowids: &[i64],
) -> Result<Vec<HnswNode>> {
    if rowids.is_empty() {
        return Ok(Vec::new());
    }

    let nodes_table = format!("{}_{}_hnsw_nodes", table_name, column_name);

    // SQLite supports up to 999 parameters, batch if needed
    const MAX_ROWIDS_PER_BATCH: usize = 999;
    let mut all_nodes = Vec::with_capacity(rowids.len());

    for chunk in rowids.chunks(MAX_ROWIDS_PER_BATCH) {
        let placeholders: Vec<&str> = (0..chunk.len()).map(|_| "?").collect();
        let sql = format!(
            "SELECT rowid, level, vector FROM \"{}\" WHERE rowid IN ({})",
            nodes_table,
            placeholders.join(",")
        );

        let params: Vec<rusqlite::types::Value> = chunk.iter().map(|&r| r.into()).collect();

        let mut stmt = db.prepare(&sql).map_err(Error::Sqlite)?;
        let nodes = stmt
            .query_map(rusqlite::params_from_iter(params), |row| {
                Ok(HnswNode {
                    rowid: row.get(0)?,
                    level: row.get(1)?,
                    vector: row.get(2)?,
                })
            })
            .map_err(Error::Sqlite)?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Error::Sqlite)?;

        all_nodes.extend(nodes);
    }

    Ok(all_nodes)
}

/// Fetch multiple nodes using cached prepared statements (FAST PATH)
///
/// Uses pre-compiled statements for power-of-2 batch sizes (1, 2, 4, 8, 16, 32, 64).
/// For larger batches, processes in chunks of 64.
///
/// # Arguments
/// * `stmt` - Pre-compiled statement for the batch size
/// * `padded_size` - Number of parameters to bind (power of 2)
/// * `rowids` - Actual rowids to fetch (may be fewer than padded_size)
///
/// # Safety
/// Caller must ensure stmt is valid and has the correct number of parameters.
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn fetch_nodes_batch_cached(
    stmt: *mut ffi::sqlite3_stmt,
    padded_size: usize,
    rowids: &[i64],
) -> Result<Vec<HnswNode>> {
    if rowids.is_empty() {
        return Ok(Vec::new());
    }

    // RAII guard: resets on create and on drop
    let guard = StmtHandleGuard::new(stmt)
        .ok_or_else(|| Error::InvalidParameter("null statement".to_string()))?;

    // Bind actual rowids
    for (i, &rowid) in rowids.iter().enumerate() {
        ffi::sqlite3_bind_int64(guard.as_ptr(), (i + 1) as i32, rowid);
    }
    // Pad with -1 for unused slots (won't match any rowid)
    for i in rowids.len()..padded_size {
        ffi::sqlite3_bind_int64(guard.as_ptr(), (i + 1) as i32, -1);
    }

    // Collect results
    let mut nodes = Vec::with_capacity(rowids.len());
    loop {
        let rc = ffi::sqlite3_step(guard.as_ptr());
        if rc == ffi::SQLITE_ROW {
            let rowid = ffi::sqlite3_column_int64(guard.as_ptr(), 0);
            let level = ffi::sqlite3_column_int(guard.as_ptr(), 1);

            let blob_ptr = ffi::sqlite3_column_blob(guard.as_ptr(), 2);
            let blob_len = ffi::sqlite3_column_bytes(guard.as_ptr(), 2);
            // Copy blob data before guard drops
            let vector = if blob_len > 0 && !blob_ptr.is_null() {
                std::slice::from_raw_parts(blob_ptr as *const u8, blob_len as usize).to_vec()
            } else {
                Vec::new()
            };

            nodes.push(HnswNode {
                rowid,
                level,
                vector,
            });
        } else if rc == ffi::SQLITE_DONE {
            break;
        } else {
            // Guard drops here, automatically resets statement
            return Err(Error::Sqlite(rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rc),
                None,
            )));
        }
    }

    // Guard drops here, automatically resets statement
    Ok(nodes)
}

/// Delete specific edges from a node at a specific level (delta update)
/// Uses DELETE ... WHERE to_rowid IN (?, ?, ...) for efficiency
pub fn delete_edges_batch(
    db: &Connection,
    table_name: &str,
    column_name: &str,
    from_rowid: i64,
    level: i32,
    to_rowids: &[i64],
) -> Result<()> {
    if to_rowids.is_empty() {
        return Ok(());
    }

    let edges_table = format!("{}_{}_hnsw_edges", table_name, column_name);

    // SQLite supports up to 999 parameters, batch if needed
    // We use 2 params for from_rowid and level, leaving 997 for to_rowids
    const MAX_ROWIDS_PER_BATCH: usize = 997;

    for chunk in to_rowids.chunks(MAX_ROWIDS_PER_BATCH) {
        let placeholders: Vec<&str> = (0..chunk.len()).map(|_| "?").collect();
        let sql = format!(
            "DELETE FROM \"{}\" WHERE from_rowid = ? AND level = ? AND to_rowid IN ({})",
            edges_table,
            placeholders.join(",")
        );

        let mut params: Vec<rusqlite::types::Value> = Vec::with_capacity(chunk.len() + 2);
        params.push(from_rowid.into());
        params.push((level as i64).into());
        for &to_rowid in chunk {
            params.push(to_rowid.into());
        }

        db.execute(&sql, rusqlite::params_from_iter(params))
            .map_err(Error::Sqlite)?;
    }

    Ok(())
}

/// Delete all outgoing edges from a node at a specific level
pub fn delete_edges_from_level(
    db: &Connection,
    table_name: &str,
    column_name: &str,
    from_rowid: i64,
    level: i32,
    cached_stmt: Option<*mut ffi::sqlite3_stmt>,
) -> Result<()> {
    // Fast path: use cached statement
    if let Some(stmt) = cached_stmt {
        unsafe {
            // RAII guard: resets on create and on drop
            let guard = StmtHandleGuard::new(stmt)
                .ok_or_else(|| Error::InvalidParameter("null statement".to_string()))?;

            ffi::sqlite3_bind_int64(guard.as_ptr(), 1, from_rowid);
            ffi::sqlite3_bind_int(guard.as_ptr(), 2, level);

            let rc = ffi::sqlite3_step(guard.as_ptr());
            if rc != ffi::SQLITE_DONE {
                // Guard drops here, automatically resets statement
                return Err(Error::Sqlite(rusqlite::Error::SqliteFailure(
                    rusqlite::ffi::Error::new(rc),
                    None,
                )));
            }
            // Guard drops here, automatically resets statement
        }
    } else {
        // Slow path fallback
        let edges_table = format!("{}_{}_hnsw_edges", table_name, column_name);
        let delete_sql = format!(
            "DELETE FROM \"{}\" WHERE from_rowid = ? AND level = ?",
            edges_table
        );

        db.execute(&delete_sql, [from_rowid, level as i64])
            .map_err(Error::Sqlite)?;
    }

    Ok(())
}

/// Get all nodes at a specific level (for rebuild operations)
/// Note: levels table removed for performance - queries nodes table directly
pub fn get_nodes_at_level(
    db: &Connection,
    table_name: &str,
    column_name: &str,
    level: i32,
) -> Result<Vec<i64>> {
    let nodes_table = format!("{}_{}_hnsw_nodes", table_name, column_name);
    let query = format!(
        "SELECT rowid FROM \"{}\" WHERE level >= ? ORDER BY rowid",
        nodes_table
    );

    let mut stmt = db.prepare(&query).map_err(Error::Sqlite)?;
    let rowids = stmt
        .query_map([level as i64], |row| row.get::<_, i64>(0))
        .map_err(Error::Sqlite)?
        .collect::<std::result::Result<Vec<i64>, _>>()
        .map_err(Error::Sqlite)?;

    Ok(rowids)
}

/// Fetch neighbors with their node data in a single JOIN query
/// Combines fetch_neighbors_cached + fetch_nodes_batch into one query
/// Query: SELECT n.rowid, n.level, n.vector FROM edges e JOIN nodes n ON e.to_rowid = n.rowid WHERE e.from_rowid = ? AND e.level = ?
pub fn fetch_neighbor_nodes_joined(
    db: &Connection,
    table_name: &str,
    column_name: &str,
    from_rowid: i64,
    level: i32,
) -> Result<Vec<HnswNode>> {
    let edges_table = format!("{}_{}_hnsw_edges", table_name, column_name);
    let nodes_table = format!("{}_{}_hnsw_nodes", table_name, column_name);

    let query = format!(
        "SELECT n.rowid, n.level, n.vector FROM \"{}\" e \
         JOIN \"{}\" n ON e.to_rowid = n.rowid \
         WHERE e.from_rowid = ? AND e.level = ?",
        edges_table, nodes_table
    );

    let mut stmt = db.prepare(&query).map_err(Error::Sqlite)?;
    let nodes = stmt
        .query_map([from_rowid, level as i64], |row| {
            Ok(HnswNode {
                rowid: row.get(0)?,
                level: row.get(1)?,
                vector: row.get(2)?,
            })
        })
        .map_err(Error::Sqlite)?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Error::Sqlite)?;

    Ok(nodes)
}

/// Count total nodes in the index
pub fn count_nodes(db: &Connection, table_name: &str, column_name: &str) -> Result<i32> {
    let nodes_table = format!("{}_{}_hnsw_nodes", table_name, column_name);
    let query = format!("SELECT COUNT(*) FROM \"{}\"", nodes_table);

    db.query_row(&query, [], |row| row.get::<_, i32>(0))
        .map_err(Error::Sqlite)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shadow;

    #[test]
    fn test_insert_and_fetch_node() {
        let db = Connection::open_in_memory().unwrap();
        shadow::create_hnsw_shadow_tables(&db, "test_table", "embedding").unwrap();

        // Insert a node
        let vector = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]; // 3 floats
        insert_node(&db, "test_table", "embedding", 1, 2, &vector, None).unwrap();

        // Fetch it back
        let node = fetch_node_data(&db, "test_table", "embedding", 1, None)
            .unwrap()
            .expect("Node should exist");

        assert_eq!(node.rowid, 1);
        assert_eq!(node.level, 2);
        assert_eq!(node.vector, vector);
    }

    #[test]
    fn test_fetch_node_level() {
        let db = Connection::open_in_memory().unwrap();
        shadow::create_hnsw_shadow_tables(&db, "test_table", "embedding").unwrap();

        let vector = vec![1u8; 12];
        insert_node(&db, "test_table", "embedding", 5, 3, &vector, None).unwrap();

        let level = fetch_node_level(&db, "test_table", "embedding", 5)
            .unwrap()
            .expect("Level should exist");

        assert_eq!(level, 3);
    }

    #[test]
    fn test_insert_and_fetch_edges() {
        let db = Connection::open_in_memory().unwrap();
        shadow::create_hnsw_shadow_tables(&db, "test_table", "embedding").unwrap();

        // Insert some nodes first
        let vector = vec![1u8; 12];
        insert_node(&db, "test_table", "embedding", 1, 2, &vector, None).unwrap();
        insert_node(&db, "test_table", "embedding", 2, 2, &vector, None).unwrap();
        insert_node(&db, "test_table", "embedding", 3, 2, &vector, None).unwrap();

        // Insert edges from node 1 to nodes 2 and 3 at level 1
        insert_edge(&db, "test_table", "embedding", 1, 2, 1, None).unwrap();
        insert_edge(&db, "test_table", "embedding", 1, 3, 1, None).unwrap();

        // Fetch neighbors
        let neighbors = fetch_neighbors_cached(&db, "test_table", "embedding", 1, 1, None).unwrap();

        assert_eq!(neighbors.len(), 2);
        assert_eq!(neighbors[0], 2); // Should be sorted by distance
        assert_eq!(neighbors[1], 3);
    }

    #[test]
    fn test_delete_edges() {
        let db = Connection::open_in_memory().unwrap();
        shadow::create_hnsw_shadow_tables(&db, "test_table", "embedding").unwrap();

        let vector = vec![1u8; 12];
        insert_node(&db, "test_table", "embedding", 1, 2, &vector, None).unwrap();
        insert_node(&db, "test_table", "embedding", 2, 2, &vector, None).unwrap();

        // Insert edges
        insert_edge(&db, "test_table", "embedding", 1, 2, 0, None).unwrap();
        insert_edge(&db, "test_table", "embedding", 1, 2, 1, None).unwrap();

        // Delete edges at level 0
        delete_edges_from_level(&db, "test_table", "embedding", 1, 0, None).unwrap();

        // Level 0 should be empty
        let neighbors0 =
            fetch_neighbors_cached(&db, "test_table", "embedding", 1, 0, None).unwrap();
        assert_eq!(neighbors0.len(), 0);

        // Level 1 should still have the edge
        let neighbors1 =
            fetch_neighbors_cached(&db, "test_table", "embedding", 1, 1, None).unwrap();
        assert_eq!(neighbors1.len(), 1);
    }

    #[test]
    fn test_get_nodes_at_level() {
        let db = Connection::open_in_memory().unwrap();
        shadow::create_hnsw_shadow_tables(&db, "test_table", "embedding").unwrap();

        let vector = vec![1u8; 12];
        // Insert nodes with different levels
        insert_node(&db, "test_table", "embedding", 1, 0, &vector, None).unwrap(); // Level 0 only
        insert_node(&db, "test_table", "embedding", 2, 1, &vector, None).unwrap(); // Levels 0-1
        insert_node(&db, "test_table", "embedding", 3, 2, &vector, None).unwrap(); // Levels 0-2

        // Get nodes at level 0 (all nodes)
        let level0 = get_nodes_at_level(&db, "test_table", "embedding", 0).unwrap();
        assert_eq!(level0.len(), 3);

        // Get nodes at level 1
        let level1 = get_nodes_at_level(&db, "test_table", "embedding", 1).unwrap();
        assert_eq!(level1.len(), 2); // Nodes 2 and 3

        // Get nodes at level 2
        let level2 = get_nodes_at_level(&db, "test_table", "embedding", 2).unwrap();
        assert_eq!(level2.len(), 1); // Only node 3
    }

    #[test]
    fn test_count_nodes() {
        let db = Connection::open_in_memory().unwrap();
        shadow::create_hnsw_shadow_tables(&db, "test_table", "embedding").unwrap();

        // Initially zero
        let count = count_nodes(&db, "test_table", "embedding").unwrap();
        assert_eq!(count, 0);

        // Insert some nodes
        let vector = vec![1u8; 12];
        for i in 1..=5 {
            insert_node(&db, "test_table", "embedding", i, 1, &vector, None).unwrap();
        }

        // Should have 5 nodes
        let count = count_nodes(&db, "test_table", "embedding").unwrap();
        assert_eq!(count, 5);
    }

    #[test]
    fn test_bidirectional_edges() {
        let db = Connection::open_in_memory().unwrap();
        shadow::create_hnsw_shadow_tables(&db, "test_table", "embedding").unwrap();

        let vector = vec![1u8; 12];
        insert_node(&db, "test_table", "embedding", 1, 1, &vector, None).unwrap();
        insert_node(&db, "test_table", "embedding", 2, 1, &vector, None).unwrap();

        // Insert bidirectional edges
        insert_edge(&db, "test_table", "embedding", 1, 2, 0, None).unwrap();
        insert_edge(&db, "test_table", "embedding", 2, 1, 0, None).unwrap();

        // Both directions should work
        let neighbors1 =
            fetch_neighbors_cached(&db, "test_table", "embedding", 1, 0, None).unwrap();
        assert_eq!(neighbors1, vec![2]);

        let neighbors2 =
            fetch_neighbors_cached(&db, "test_table", "embedding", 2, 0, None).unwrap();
        assert_eq!(neighbors2, vec![1]);
    }
}
