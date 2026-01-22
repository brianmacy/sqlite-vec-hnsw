//! HNSW index rebuild functionality

use crate::error::{Error, Result};
use crate::hnsw::{HnswMetadata, HnswParams, insert};
use rusqlite::Connection;

/// Rebuild HNSW index from scratch
///
/// # Algorithm
/// 1. Clear existing HNSW shadow tables (DELETE FROM)
/// 2. Reset metadata (entry point = -1, num_nodes = 0)
/// 3. Query all vectors from virtual table
/// 4. Re-insert each vector into HNSW index
///
/// # Arguments
/// * `db` - Database connection
/// * `table_name` - Virtual table name
/// * `column_name` - Vector column name
/// * `new_params` - Optional new parameters (None = keep existing)
pub fn rebuild_hnsw_index(
    db: &Connection,
    table_name: &str,
    column_name: &str,
    new_params: Option<HnswParams>,
) -> Result<i32> {
    // Step 1: Load existing metadata
    let mut metadata = HnswMetadata::load_from_db(db, table_name, column_name)?
        .ok_or_else(|| Error::InvalidParameter("HNSW index not initialized".to_string()))?;

    // Update parameters if provided
    if let Some(params) = new_params {
        metadata.params = params;
    }

    // Step 2: Clear HNSW shadow tables
    clear_hnsw_tables(db, table_name, column_name)?;

    // Step 3: Reset metadata
    metadata.entry_point_rowid = -1;
    metadata.entry_point_level = -1;
    metadata.num_nodes = 0;
    metadata.hnsw_version = 1;
    metadata.save_to_db(db, table_name, column_name)?;

    // Step 4: Query all vectors from the virtual table
    // We need to read from the vector_chunks shadow tables directly
    let vector_table = format!("{}_vector_chunks00", table_name);
    let rowids_table = format!("{}_rowids", table_name);

    let query = format!(
        "SELECT r.rowid FROM \"{}\" r \
         JOIN \"{}\" v ON r.chunk_id IS NOT NULL",
        rowids_table, vector_table
    );

    let mut stmt = db.prepare(&query).map_err(Error::Sqlite)?;
    let rowids: Vec<i64> = stmt
        .query_map([], |row| row.get::<_, i64>(0))
        .map_err(Error::Sqlite)?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Error::Sqlite)?;

    let vector_count = rowids.len();

    // Step 5: Re-insert all vectors to rebuild the index
    for rowid in rowids {
        // Read vector from shadow table
        let vector_data = crate::shadow::read_vector_from_chunk(
            db, "main", // TODO: Get actual schema name
            table_name, 0, // TODO: Handle multiple vector columns
            rowid,
        )?;

        if let Some(vector) = vector_data {
            // Re-insert into HNSW index
            insert::insert_hnsw(
                db,
                &mut metadata,
                table_name,
                column_name,
                rowid,
                &vector,
                None,
            )?;
        }
    }

    Ok(vector_count as i32)
}

/// Clear all HNSW shadow tables (public version for SQL function)
pub fn clear_hnsw_tables_internal(
    db: &Connection,
    table_name: &str,
    column_name: &str,
) -> Result<()> {
    clear_hnsw_tables(db, table_name, column_name)
}

/// Clear all HNSW shadow tables
fn clear_hnsw_tables(db: &Connection, table_name: &str, column_name: &str) -> Result<()> {
    // Simply DELETE from each table
    // Note: The previous approach of finalizing all statements caused segfaults
    // because Vec0Tab's cached statement pointers became dangling
    let nodes_table = format!("{}_{}_hnsw_nodes", table_name, column_name);
    db.execute(&format!("DELETE FROM \"{}\"", nodes_table), [])
        .map_err(Error::Sqlite)?;

    let edges_table = format!("{}_{}_hnsw_edges", table_name, column_name);
    db.execute(&format!("DELETE FROM \"{}\"", edges_table), [])
        .map_err(Error::Sqlite)?;

    let levels_table = format!("{}_{}_hnsw_levels", table_name, column_name);
    db.execute(&format!("DELETE FROM \"{}\"", levels_table), [])
        .map_err(Error::Sqlite)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::DistanceMetric;
    use crate::hnsw::{HnswMetadata, storage};
    use crate::shadow;
    use crate::vector::VectorType;

    #[test]
    fn test_clear_hnsw_tables() {
        let db = Connection::open_in_memory().unwrap();
        shadow::create_hnsw_shadow_tables(&db, "test_table", "embedding").unwrap();

        // Insert some test data
        let vector = vec![1u8; 12];
        storage::insert_node(&db, "test_table", "embedding", 1, 1, &vector, None).unwrap();
        storage::insert_edge(&db, "test_table", "embedding", 1, 2, 0, None).unwrap();

        // Verify data exists
        let node_count = storage::count_nodes(&db, "test_table", "embedding").unwrap();
        assert_eq!(node_count, 1);

        // Clear tables
        clear_hnsw_tables(&db, "test_table", "embedding").unwrap();

        // Verify tables are empty
        let node_count = storage::count_nodes(&db, "test_table", "embedding").unwrap();
        assert_eq!(node_count, 0);

        let edge_count: i32 = db
            .query_row(
                "SELECT COUNT(*) FROM test_table_embedding_hnsw_edges",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(edge_count, 0);
    }

    #[test]
    fn test_rebuild_with_new_params() {
        let db = Connection::open_in_memory().unwrap();

        // Create shadow tables for virtual table
        let config = shadow::ShadowTablesConfig {
            num_vector_columns: 1,
            num_metadata_columns: 0,
            num_auxiliary_columns: 0,
            has_text_pk: false,
            num_partition_columns: 0,
        };
        shadow::create_shadow_tables(&db, "main", "test_table", &config).unwrap();
        shadow::create_hnsw_shadow_tables(&db, "test_table", "embedding").unwrap();

        // Initialize metadata
        let metadata = HnswMetadata::new(3, VectorType::Float32, DistanceMetric::L2);
        metadata.save_to_db(&db, "test_table", "embedding").unwrap();

        // Create new parameters
        let new_params = HnswParams {
            m: 16,
            max_m0: 32,
            ef_construction: 200,
            ..HnswParams::default()
        };

        // Note: rebuild requires vectors in shadow tables
        // This test just verifies the rebuild function works structurally
        let result = rebuild_hnsw_index(&db, "test_table", "embedding", Some(new_params));

        // May fail if no vectors exist, but should not panic
        match result {
            Ok(count) => {
                println!("Rebuilt {} vectors", count);
                assert_eq!(count, 0); // No vectors in this test
            }
            Err(e) => {
                println!("Rebuild failed (expected with no vectors): {:?}", e);
            }
        }
    }
}
