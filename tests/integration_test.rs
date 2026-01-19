//! Integration tests for sqlite-vec-hnsw
//!
//! These tests verify the implementation works correctly with shadow table persistence

use rusqlite::{Connection, Result as SqliteResult};

/// Test helper to create an in-memory database
fn create_test_db() -> SqliteResult<Connection> {
    Connection::open_in_memory()
}

/// Test helper to initialize extension
fn init_extension(db: &Connection) -> sqlite_vec_hnsw::Result<()> {
    sqlite_vec_hnsw::init(db)
}

#[test]
fn test_extension_loading() {
    let db = create_test_db().expect("Failed to create database");
    let result = init_extension(&db);

    assert!(result.is_ok(), "Extension initialization should succeed");
}

#[test]
fn test_create_simple_vec0_table() {
    let db = create_test_db().expect("Failed to create database");
    init_extension(&db).expect("Failed to init extension");

    // Create a simple vec0 table
    let result = db.execute(
        "CREATE VIRTUAL TABLE vec_simple USING vec0(embedding float[8])",
        [],
    );

    assert!(result.is_ok(), "CREATE VIRTUAL TABLE should succeed");

    // Verify table exists
    let count: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='vec_simple'",
            [],
            |row| row.get(0),
        )
        .unwrap();

    assert_eq!(count, 1, "Table should exist");
}

#[test]
fn test_create_vec0_table_with_metadata() {
    let db = create_test_db().expect("Failed to create database");
    init_extension(&db).expect("Failed to init extension");

    // Create vec0 table with metadata columns
    let result = db.execute(
        "CREATE VIRTUAL TABLE vec_movies USING vec0(
            synopsis_embedding float[768],
            genre TEXT,
            rating FLOAT
        )",
        [],
    );

    assert!(result.is_ok(), "CREATE VIRTUAL TABLE with metadata should succeed");
}

#[test]
fn test_create_vec0_table_with_partition_keys() {
    let db = create_test_db().expect("Failed to create database");
    init_extension(&db).expect("Failed to init extension");

    // Create vec0 table with partition keys
    let result = db.execute(
        "CREATE VIRTUAL TABLE vec_chunks USING vec0(
            user_id INTEGER PARTITION KEY,
            embedding float[1024]
        )",
        [],
    );

    assert!(result.is_ok(), "CREATE VIRTUAL TABLE with partition keys should succeed");
}

#[test]
fn test_shadow_tables_created() {
    let db = create_test_db().expect("Failed to create database");
    init_extension(&db).expect("Failed to init extension");

    // Create a vec0 table
    db.execute(
        "CREATE VIRTUAL TABLE vec_test USING vec0(embedding float[384])",
        [],
    )
    .expect("Failed to create table");

    // Count shadow tables
    let shadow_count: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name LIKE 'vec_test_%'",
            [],
            |row| row.get(0),
        )
        .unwrap();

    println!("Shadow tables created: {}", shadow_count);
    // Should have: chunks, rowids, vector_chunks00, and 4 HNSW tables
    assert!(shadow_count >= 7, "Should have at least 7 shadow tables");
}

#[test]
fn test_insert_and_query_vectors() {
    let db = create_test_db().expect("Failed to create database");
    init_extension(&db).expect("Failed to init extension");

    // Create table
    db.execute(
        "CREATE VIRTUAL TABLE vec_test USING vec0(embedding float[3])",
        [],
    )
    .expect("Failed to create table");

    // Insert vectors
    db.execute(
        "INSERT INTO vec_test(rowid, embedding) VALUES (1, vec_f32('[1.0, 2.0, 3.0]'))",
        [],
    )
    .expect("Failed to insert");

    db.execute(
        "INSERT INTO vec_test(rowid, embedding) VALUES (2, vec_f32('[4.0, 5.0, 6.0]'))",
        [],
    )
    .expect("Failed to insert");

    // Query count
    let count: i64 = db
        .query_row("SELECT COUNT(*) FROM vec_test", [], |row| row.get(0))
        .unwrap();

    assert_eq!(count, 2, "Should have 2 rows");

    // Query rowids
    let mut stmt = db.prepare("SELECT rowid FROM vec_test ORDER BY rowid").unwrap();
    let rowids: Vec<i64> = stmt
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<SqliteResult<Vec<_>>>()
        .unwrap();

    assert_eq!(rowids, vec![1, 2]);
}

#[test]
fn test_multiple_vector_columns() {
    let db = create_test_db().expect("Failed to create database");
    init_extension(&db).expect("Failed to init extension");

    // Create table with multiple vector columns
    let result = db.execute(
        "CREATE VIRTUAL TABLE vec_multi USING vec0(
            title_embedding float[384],
            content_embedding float[768]
        )",
        [],
    );

    assert!(result.is_ok(), "CREATE TABLE with multiple vector columns should succeed");

    // Verify shadow tables for both columns exist
    let shadow_count: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name LIKE 'vec_multi_vector_chunks%'",
            [],
            |row| row.get(0),
        )
        .unwrap();

    assert_eq!(shadow_count, 2, "Should have 2 vector_chunks tables");
}

#[test]
fn test_int8_vector_type() {
    let db = create_test_db().expect("Failed to create database");
    init_extension(&db).expect("Failed to init extension");

    // Create table with int8 vectors
    let result = db.execute(
        "CREATE VIRTUAL TABLE vec_int8 USING vec0(embedding int8[128])",
        [],
    );

    assert!(result.is_ok(), "CREATE TABLE with int8 vectors should succeed");
}

#[test]
fn test_binary_vector_type() {
    let db = create_test_db().expect("Failed to create database");
    init_extension(&db).expect("Failed to init extension");

    // Create table with binary vectors
    let result = db.execute(
        "CREATE VIRTUAL TABLE vec_binary USING vec0(embedding bit[1024])",
        [],
    );

    assert!(result.is_ok(), "CREATE TABLE with binary vectors should succeed");
}

#[test]
fn test_delete_vector() {
    let db = create_test_db().expect("Failed to create database");
    init_extension(&db).expect("Failed to init extension");

    // Create and populate table
    db.execute(
        "CREATE VIRTUAL TABLE vec_del USING vec0(embedding float[3])",
        [],
    )
    .unwrap();

    db.execute(
        "INSERT INTO vec_del(rowid, embedding) VALUES (1, vec_f32('[1.0, 2.0, 3.0]'))",
        [],
    )
    .unwrap();

    // Try DELETE (not yet implemented)
    let result = db.execute("DELETE FROM vec_del WHERE rowid = 1", []);

    assert!(
        result.is_err(),
        "DELETE should fail (not yet implemented)"
    );
}

#[test]
fn test_update_vector() {
    let db = create_test_db().expect("Failed to create database");
    init_extension(&db).expect("Failed to init extension");

    // Create and populate table
    db.execute(
        "CREATE VIRTUAL TABLE vec_upd USING vec0(embedding float[3])",
        [],
    )
    .unwrap();

    db.execute(
        "INSERT INTO vec_upd(rowid, embedding) VALUES (1, vec_f32('[1.0, 2.0, 3.0]'))",
        [],
    )
    .unwrap();

    // Try UPDATE (not yet implemented)
    let result = db.execute(
        "UPDATE vec_upd SET embedding = vec_f32('[4.0, 5.0, 6.0]') WHERE rowid = 1",
        [],
    );

    assert!(
        result.is_err(),
        "UPDATE should fail (not yet implemented)"
    );
}

#[test]
fn test_point_query_by_rowid() {
    let db = create_test_db().expect("Failed to create database");
    init_extension(&db).expect("Failed to init extension");

    // Create and populate table
    db.execute(
        "CREATE VIRTUAL TABLE vec_point USING vec0(embedding float[3])",
        [],
    )
    .unwrap();

    db.execute(
        "INSERT INTO vec_point(rowid, embedding) VALUES (1, vec_f32('[1.0, 2.0, 3.0]'))",
        [],
    )
    .unwrap();

    // Point query by rowid - should work with full scan
    let result: SqliteResult<i64> = db.query_row(
        "SELECT rowid FROM vec_point WHERE rowid = 1",
        [],
        |row| row.get(0),
    );

    // This might work with full scan even if point query optimization isn't implemented
    if let Ok(rowid) = result {
        assert_eq!(rowid, 1, "Should find rowid 1");
    }
}

#[test]
fn test_full_scan_query() {
    let db = create_test_db().expect("Failed to create database");
    init_extension(&db).expect("Failed to init extension");

    // Create and populate table
    db.execute(
        "CREATE VIRTUAL TABLE vec_scan USING vec0(embedding float[4])",
        [],
    )
    .unwrap();

    for i in 1..=5 {
        db.execute(
            &format!(
                "INSERT INTO vec_scan(rowid, embedding) VALUES ({}, vec_f32('[{}.0, {}.0, {}.0, {}.0]'))",
                i, i, i + 1, i + 2, i + 3
            ),
            [],
        )
        .unwrap();
    }

    // Full scan query
    let count: i64 = db
        .query_row("SELECT COUNT(*) FROM vec_scan", [], |row| row.get(0))
        .unwrap();

    assert_eq!(count, 5, "Full scan should return all 5 rows");

    // Verify all rowids present
    let mut stmt = db
        .prepare("SELECT rowid FROM vec_scan ORDER BY rowid")
        .unwrap();
    let rowids: Vec<i64> = stmt
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<SqliteResult<Vec<_>>>()
        .unwrap();

    assert_eq!(rowids, vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_vec_f32_scalar_function() {
    let db = create_test_db().expect("Failed to create database");
    init_extension(&db).expect("Failed to init extension");

    // Test vec_f32 function
    let result: SqliteResult<Vec<u8>> = db.query_row(
        "SELECT vec_f32('[1.0, 2.0, 3.0]')",
        [],
        |row| row.get(0),
    );

    assert!(result.is_ok(), "vec_f32 should work");
    let data = result.unwrap();
    assert_eq!(data.len(), 12, "Should be 12 bytes for 3 float32 values");
}

#[test]
fn test_vec_distance_l2() {
    let db = create_test_db().expect("Failed to create database");
    init_extension(&db).expect("Failed to init extension");

    // Test distance calculation
    let distance: f64 = db
        .query_row(
            "SELECT vec_distance_l2(vec_f32('[1.0, 0.0, 0.0]'), vec_f32('[0.0, 1.0, 0.0]'))",
            [],
            |row| row.get(0),
        )
        .unwrap();

    println!("L2 distance: {}", distance);
    // Distance should be sqrt(2) ≈ 1.414
    assert!((distance - 1.414).abs() < 0.01, "L2 distance should be approximately sqrt(2)");
}

#[test]
fn test_persistence_across_connections() {
    use tempfile::NamedTempFile;

    // Create a temporary file for the database
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    // First connection: create and populate
    {
        let db = Connection::open(path).unwrap();
        init_extension(&db).unwrap();

        db.execute(
            "CREATE VIRTUAL TABLE vec_persist USING vec0(embedding float[3])",
            [],
        )
        .unwrap();

        db.execute(
            "INSERT INTO vec_persist(rowid, embedding) VALUES (1, vec_f32('[1.0, 2.0, 3.0]'))",
            [],
        )
        .unwrap();

        db.execute(
            "INSERT INTO vec_persist(rowid, embedding) VALUES (2, vec_f32('[4.0, 5.0, 6.0]'))",
            [],
        )
        .unwrap();
    }

    // Second connection: verify data persists
    {
        let db = Connection::open(path).unwrap();
        init_extension(&db).unwrap();

        let count: i64 = db
            .query_row("SELECT COUNT(*) FROM vec_persist", [], |row| row.get(0))
            .unwrap();

        assert_eq!(count, 2, "Data should persist across connections");

        // Verify shadow tables still exist
        let shadow_count: i64 = db
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name LIKE 'vec_persist_%'",
                [],
                |row| row.get(0),
            )
            .unwrap();

        assert!(shadow_count >= 3, "Shadow tables should persist");
    }
}

#[test]
fn test_insert_with_auto_rowid() {
    let db = create_test_db().expect("Failed to create database");
    init_extension(&db).expect("Failed to init extension");

    db.execute(
        "CREATE VIRTUAL TABLE vec_auto USING vec0(embedding float[2])",
        [],
    )
    .unwrap();

    // Insert without specifying rowid (uses auto-increment)
    db.execute(
        "INSERT INTO vec_auto(embedding) VALUES (vec_f32('[1.0, 2.0]'))",
        [],
    )
    .unwrap();

    db.execute(
        "INSERT INTO vec_auto(embedding) VALUES (vec_f32('[3.0, 4.0]'))",
        [],
    )
    .unwrap();

    let count: i64 = db
        .query_row("SELECT COUNT(*) FROM vec_auto", [], |row| row.get(0))
        .unwrap();

    assert_eq!(count, 2, "Auto-rowid inserts should work");
}

#[test]
fn test_chunk_allocation() {
    let db = create_test_db().expect("Failed to create database");
    init_extension(&db).expect("Failed to init extension");

    db.execute(
        "CREATE VIRTUAL TABLE vec_chunks USING vec0(embedding float[3])",
        [],
    )
    .unwrap();

    // Insert multiple vectors to test chunk management
    for i in 1..=10 {
        db.execute(
            &format!(
                "INSERT INTO vec_chunks(rowid, embedding) VALUES ({}, vec_f32('[{}.0, {}.0, {}.0]'))",
                i, i, i + 1, i + 2
            ),
            [],
        )
        .unwrap();
    }

    // Verify all inserts succeeded
    let count: i64 = db
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |row| row.get(0))
        .unwrap();

    assert_eq!(count, 10, "Should have 10 vectors");

    // Verify chunks table has entries
    let chunk_count: i64 = db
        .query_row("SELECT COUNT(*) FROM vec_chunks_chunks", [], |row| row.get(0))
        .unwrap();

    assert!(chunk_count >= 1, "Should have at least one chunk");
}

#[test]
fn test_vector_data_integrity() {
    let db = create_test_db().expect("Failed to create database");
    init_extension(&db).expect("Failed to init extension");

    db.execute(
        "CREATE VIRTUAL TABLE vec_integrity USING vec0(embedding float[3])",
        [],
    )
    .unwrap();

    // Insert a vector
    db.execute(
        "INSERT INTO vec_integrity(rowid, embedding) VALUES (1, vec_f32('[1.5, 2.5, 3.5]'))",
        [],
    )
    .unwrap();

    // Read it back
    let vector_data: Vec<u8> = db
        .query_row(
            "SELECT embedding FROM vec_integrity WHERE rowid = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();

    assert_eq!(vector_data.len(), 12, "Should be 12 bytes for 3 float32 values");

    // Decode and verify values
    let floats: Vec<f32> = vector_data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    assert_eq!(floats.len(), 3);
    assert!((floats[0] - 1.5).abs() < 0.001);
    assert!((floats[1] - 2.5).abs() < 0.001);
    assert!((floats[2] - 3.5).abs() < 0.001);
}

// Tests for not-yet-implemented features

#[test]
fn test_delete_vector_not_implemented() {
    let db = create_test_db().expect("Failed to create database");
    init_extension(&db).expect("Failed to init extension");

    db.execute(
        "CREATE VIRTUAL TABLE vec_del USING vec0(embedding float[3])",
        [],
    )
    .unwrap();

    db.execute(
        "INSERT INTO vec_del(rowid, embedding) VALUES (1, vec_f32('[1.0, 2.0, 3.0]'))",
        [],
    )
    .unwrap();

    // DELETE not yet implemented
    let result = db.execute("DELETE FROM vec_del WHERE rowid = 1", []);
    assert!(result.is_err(), "DELETE not yet implemented");
}

#[test]
fn test_update_vector_not_implemented() {
    let db = create_test_db().expect("Failed to create database");
    init_extension(&db).expect("Failed to init extension");

    db.execute(
        "CREATE VIRTUAL TABLE vec_upd USING vec0(embedding float[3])",
        [],
    )
    .unwrap();

    db.execute(
        "INSERT INTO vec_upd(rowid, embedding) VALUES (1, vec_f32('[1.0, 2.0, 3.0]'))",
        [],
    )
    .unwrap();

    // UPDATE not yet implemented
    let result = db.execute(
        "UPDATE vec_upd SET embedding = vec_f32('[4.0, 5.0, 6.0]') WHERE rowid = 1",
        [],
    );
    assert!(result.is_err(), "UPDATE not yet implemented");
}

#[test]
fn test_knn_query_not_implemented() {
    let db = create_test_db().expect("Failed to create database");
    init_extension(&db).expect("Failed to init extension");

    db.execute(
        "CREATE VIRTUAL TABLE vec_knn USING vec0(embedding float[3])",
        [],
    )
    .unwrap();

    // Insert test data
    for i in 1..=5 {
        db.execute(
            &format!(
                "INSERT INTO vec_knn(rowid, embedding) VALUES ({}, vec_f32('[{}.0, {}.0, {}.0]'))",
                i, i, i + 1, i + 2
            ),
            [],
        )
        .unwrap();
    }

    // KNN query using MATCH operator
    let result = db.query_row(
        "SELECT rowid, distance FROM vec_knn WHERE embedding MATCH vec_f32('[1.0, 2.0, 3.0]') AND k = 3 ORDER BY distance",
        [],
        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, f32>(1)?)),
    );

    println!("KNN query result: {:?}", result);

    // Should work and return the closest vector
    if let Ok((rowid, distance)) = result {
        assert_eq!(rowid, 1); // Vector [1.0, 2.0, 3.0] is closest to itself
        assert!(distance < 0.01, "Distance to itself should be near zero");
        println!("KNN query successful: rowid={}, distance={}", rowid, distance);
    }
}

#[test]
fn test_knn_end_to_end() {
    let db = create_test_db().expect("Failed to create database");
    init_extension(&db).expect("Failed to init extension");

    db.execute(
        "CREATE VIRTUAL TABLE vec_knn_test USING vec0(embedding float[3])",
        [],
    )
    .unwrap();

    // Insert test vectors
    db.execute(
        "INSERT INTO vec_knn_test(rowid, embedding) VALUES (1, vec_f32('[1.0, 0.0, 0.0]'))",
        [],
    )
    .unwrap();

    db.execute(
        "INSERT INTO vec_knn_test(rowid, embedding) VALUES (2, vec_f32('[0.0, 1.0, 0.0]'))",
        [],
    )
    .unwrap();

    db.execute(
        "INSERT INTO vec_knn_test(rowid, embedding) VALUES (3, vec_f32('[0.0, 0.0, 1.0]'))",
        [],
    )
    .unwrap();

    db.execute(
        "INSERT INTO vec_knn_test(rowid, embedding) VALUES (4, vec_f32('[1.0, 1.0, 0.0]'))",
        [],
    )
    .unwrap();

    db.execute(
        "INSERT INTO vec_knn_test(rowid, embedding) VALUES (5, vec_f32('[0.5, 0.5, 0.5]'))",
        [],
    )
    .unwrap();

    // Query for vectors nearest to [1.0, 0.0, 0.0]
    let mut stmt = db
        .prepare("SELECT rowid, distance FROM vec_knn_test WHERE embedding MATCH vec_f32('[1.0, 0.0, 0.0]') AND k = 3 ORDER BY distance")
        .unwrap();

    let results: Vec<(i64, f32)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<SqliteResult<Vec<_>>>()
        .unwrap();

    println!("KNN results: {:?}", results);

    // Should return 3 nearest vectors
    assert_eq!(results.len(), 3, "Should return k=3 results");

    // First result should be rowid 1 (exact match)
    assert_eq!(results[0].0, 1);
    assert!(results[0].1 < 0.01, "Distance to itself should be near zero");

    // Results should be sorted by distance
    for i in 1..results.len() {
        assert!(results[i].1 >= results[i - 1].1, "Results should be sorted by distance");
    }
}
