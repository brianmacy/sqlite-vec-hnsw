//! Page-cache based HNSW storage operations
//!
//! This module implements HNSW node and edge operations that query shadow tables
//! instead of keeping the entire graph in memory. This enables scaling to millions
//! of vectors while keeping memory usage minimal (~64 bytes per index).

use crate::error::{Error, Result};
use rusqlite::{Connection, OptionalExtension};

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
) -> Result<Option<HnswNode>> {
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
/// # Returns
/// List of neighbor rowids at the given level
pub fn fetch_neighbors(
    db: &Connection,
    table_name: &str,
    column_name: &str,
    from_rowid: i64,
    level: i32,
) -> Result<Vec<i64>> {
    let edges_table = format!("{}_{}_hnsw_edges", table_name, column_name);
    let query = format!(
        "SELECT to_rowid FROM \"{}\" WHERE from_rowid = ? AND level = ? ORDER BY distance",
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
/// # Returns
/// List of (neighbor_rowid, distance) tuples at the given level
pub fn fetch_neighbors_with_distances(
    db: &Connection,
    table_name: &str,
    column_name: &str,
    from_rowid: i64,
    level: i32,
) -> Result<Vec<(i64, f32)>> {
    let edges_table = format!("{}_{}_hnsw_edges", table_name, column_name);
    let query = format!(
        "SELECT to_rowid, distance FROM \"{}\" WHERE from_rowid = ? AND level = ? ORDER BY distance",
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
) -> Result<()> {
    let nodes_table = format!("{}_{}_hnsw_nodes", table_name, column_name);
    let insert_sql = format!(
        "INSERT OR REPLACE INTO \"{}\" (rowid, level, vector) VALUES (?, ?, ?)",
        nodes_table
    );

    db.execute(&insert_sql, rusqlite::params![rowid, level, vector])
        .map_err(Error::Sqlite)?;

    // Also insert into levels table for efficient level queries
    let levels_table = format!("{}_{}_hnsw_levels", table_name, column_name);
    for lv in 0..=level {
        let insert_level_sql = format!(
            "INSERT OR IGNORE INTO \"{}\" (level, rowid) VALUES (?, ?)",
            levels_table
        );
        db.execute(&insert_level_sql, [lv as i64, rowid])
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
    distance: f32,
) -> Result<()> {
    let edges_table = format!("{}_{}_hnsw_edges", table_name, column_name);
    let insert_sql = format!(
        "INSERT OR IGNORE INTO \"{}\" (from_rowid, to_rowid, level, distance) VALUES (?, ?, ?, ?)",
        edges_table
    );

    db.execute(
        &insert_sql,
        rusqlite::params![from_rowid, to_rowid, level, distance],
    )
    .map_err(Error::Sqlite)?;

    Ok(())
}

/// Delete all outgoing edges from a node at a specific level
pub fn delete_edges_from_level(
    db: &Connection,
    table_name: &str,
    column_name: &str,
    from_rowid: i64,
    level: i32,
) -> Result<()> {
    let edges_table = format!("{}_{}_hnsw_edges", table_name, column_name);
    let delete_sql = format!(
        "DELETE FROM \"{}\" WHERE from_rowid = ? AND level = ?",
        edges_table
    );

    db.execute(&delete_sql, [from_rowid, level as i64])
        .map_err(Error::Sqlite)?;

    Ok(())
}

/// Get all nodes at a specific level (for rebuild operations)
pub fn get_nodes_at_level(
    db: &Connection,
    table_name: &str,
    column_name: &str,
    level: i32,
) -> Result<Vec<i64>> {
    let levels_table = format!("{}_{}_hnsw_levels", table_name, column_name);
    let query = format!(
        "SELECT rowid FROM \"{}\" WHERE level = ? ORDER BY rowid",
        levels_table
    );

    let mut stmt = db.prepare(&query).map_err(Error::Sqlite)?;
    let rowids = stmt
        .query_map([level as i64], |row| row.get::<_, i64>(0))
        .map_err(Error::Sqlite)?
        .collect::<std::result::Result<Vec<i64>, _>>()
        .map_err(Error::Sqlite)?;

    Ok(rowids)
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
        insert_node(&db, "test_table", "embedding", 1, 2, &vector).unwrap();

        // Fetch it back
        let node = fetch_node_data(&db, "test_table", "embedding", 1)
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
        insert_node(&db, "test_table", "embedding", 5, 3, &vector).unwrap();

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
        insert_node(&db, "test_table", "embedding", 1, 2, &vector).unwrap();
        insert_node(&db, "test_table", "embedding", 2, 2, &vector).unwrap();
        insert_node(&db, "test_table", "embedding", 3, 2, &vector).unwrap();

        // Insert edges from node 1 to nodes 2 and 3 at level 1
        insert_edge(&db, "test_table", "embedding", 1, 2, 1, 0.5).unwrap();
        insert_edge(&db, "test_table", "embedding", 1, 3, 1, 0.7).unwrap();

        // Fetch neighbors
        let neighbors = fetch_neighbors(&db, "test_table", "embedding", 1, 1).unwrap();

        assert_eq!(neighbors.len(), 2);
        assert_eq!(neighbors[0], 2); // Should be sorted by distance
        assert_eq!(neighbors[1], 3);
    }

    #[test]
    fn test_delete_edges() {
        let db = Connection::open_in_memory().unwrap();
        shadow::create_hnsw_shadow_tables(&db, "test_table", "embedding").unwrap();

        let vector = vec![1u8; 12];
        insert_node(&db, "test_table", "embedding", 1, 2, &vector).unwrap();
        insert_node(&db, "test_table", "embedding", 2, 2, &vector).unwrap();

        // Insert edges
        insert_edge(&db, "test_table", "embedding", 1, 2, 0, 0.5).unwrap();
        insert_edge(&db, "test_table", "embedding", 1, 2, 1, 0.5).unwrap();

        // Delete edges at level 0
        delete_edges_from_level(&db, "test_table", "embedding", 1, 0).unwrap();

        // Level 0 should be empty
        let neighbors0 = fetch_neighbors(&db, "test_table", "embedding", 1, 0).unwrap();
        assert_eq!(neighbors0.len(), 0);

        // Level 1 should still have the edge
        let neighbors1 = fetch_neighbors(&db, "test_table", "embedding", 1, 1).unwrap();
        assert_eq!(neighbors1.len(), 1);
    }

    #[test]
    fn test_get_nodes_at_level() {
        let db = Connection::open_in_memory().unwrap();
        shadow::create_hnsw_shadow_tables(&db, "test_table", "embedding").unwrap();

        let vector = vec![1u8; 12];
        // Insert nodes with different levels
        insert_node(&db, "test_table", "embedding", 1, 0, &vector).unwrap(); // Level 0 only
        insert_node(&db, "test_table", "embedding", 2, 1, &vector).unwrap(); // Levels 0-1
        insert_node(&db, "test_table", "embedding", 3, 2, &vector).unwrap(); // Levels 0-2

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
            insert_node(&db, "test_table", "embedding", i, 1, &vector).unwrap();
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
        insert_node(&db, "test_table", "embedding", 1, 1, &vector).unwrap();
        insert_node(&db, "test_table", "embedding", 2, 1, &vector).unwrap();

        // Insert bidirectional edges
        insert_edge(&db, "test_table", "embedding", 1, 2, 0, 0.5).unwrap();
        insert_edge(&db, "test_table", "embedding", 2, 1, 0, 0.5).unwrap();

        // Both directions should work
        let neighbors1 = fetch_neighbors(&db, "test_table", "embedding", 1, 0).unwrap();
        assert_eq!(neighbors1, vec![2]);

        let neighbors2 = fetch_neighbors(&db, "test_table", "embedding", 2, 0).unwrap();
        assert_eq!(neighbors2, vec![1]);
    }
}
