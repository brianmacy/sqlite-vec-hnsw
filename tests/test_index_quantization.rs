//! Tests for HNSW index quantization feature
//!
//! Tests the `index_quantization=int8` option that quantizes float32 vectors
//! to int8 for HNSW index storage, reducing memory usage by ~75%.

use rusqlite::Connection;

/// Test that index_quantization=int8 can be parsed and tables created
#[test]
fn test_index_quantization_table_creation() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    // Create table with index_quantization option
    db.execute(
        "CREATE VIRTUAL TABLE test USING vec0(embedding float[8] index_quantization=int8)",
        [],
    )
    .unwrap();

    // Verify table was created successfully
    let table_exists: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='test'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(table_exists, 1, "Virtual table should exist");

    // Verify HNSW meta table has index_quantization column
    let index_quant: String = db
        .query_row(
            "SELECT index_quantization FROM test_embedding_hnsw_meta WHERE id = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(index_quant, "int8", "index_quantization should be 'int8'");
}

/// Test that default index_quantization is 'none'
#[test]
fn test_index_quantization_default() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    // Create table without index_quantization option
    db.execute(
        "CREATE VIRTUAL TABLE test USING vec0(embedding float[8])",
        [],
    )
    .unwrap();

    // Insert a vector to initialize HNSW metadata
    db.execute(
        "INSERT INTO test(rowid, embedding) VALUES (1, vec_f32('[1,2,3,4,5,6,7,8]'))",
        [],
    )
    .unwrap();

    // Verify default is 'none'
    let index_quant: String = db
        .query_row(
            "SELECT index_quantization FROM test_embedding_hnsw_meta WHERE id = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        index_quant, "none",
        "default index_quantization should be 'none'"
    );
}

/// Test insert and search with index_quantization=int8
#[test]
fn test_index_quantization_insert_and_search() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    // Create table with index_quantization=int8
    db.execute(
        "CREATE VIRTUAL TABLE test USING vec0(embedding float[4] index_quantization=int8)",
        [],
    )
    .unwrap();

    // Insert test vectors
    db.execute(
        "INSERT INTO test(rowid, embedding) VALUES (1, vec_f32('[1.0, 0.0, 0.0, 0.0]'))",
        [],
    )
    .unwrap();
    db.execute(
        "INSERT INTO test(rowid, embedding) VALUES (2, vec_f32('[0.0, 1.0, 0.0, 0.0]'))",
        [],
    )
    .unwrap();
    db.execute(
        "INSERT INTO test(rowid, embedding) VALUES (3, vec_f32('[0.5, 0.5, 0.0, 0.0]'))",
        [],
    )
    .unwrap();

    // Verify HNSW nodes table contains int8 vectors (1 byte per dimension instead of 4)
    // With 4 dimensions: float32 = 16 bytes, int8 = 4 bytes
    let vector_size: i32 = db
        .query_row(
            "SELECT LENGTH(vector) FROM test_embedding_hnsw_nodes WHERE rowid = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        vector_size, 4,
        "HNSW node should store int8 vectors (4 bytes for 4 dimensions)"
    );

    // KNN query should still work
    let mut stmt = db
        .prepare("SELECT rowid, distance FROM test WHERE embedding MATCH vec_f32('[1.0, 0.0, 0.0, 0.0]') AND k = 3 ORDER BY distance")
        .unwrap();

    let results: Vec<(i64, f64)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(results.len(), 3, "Should return k=3 results");
    // Closest should be rowid 1 (exact match after quantization)
    assert_eq!(results[0].0, 1, "Closest should be rowid 1");
}

/// Test that main storage still contains float32 vectors even with index_quantization=int8
#[test]
fn test_index_quantization_main_storage_unchanged() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    // Create table with index_quantization=int8
    db.execute(
        "CREATE VIRTUAL TABLE test USING vec0(embedding float[4] index_quantization=int8)",
        [],
    )
    .unwrap();

    // Insert a vector
    db.execute(
        "INSERT INTO test(rowid, embedding) VALUES (1, vec_f32('[1.0, 2.0, 3.0, 4.0]'))",
        [],
    )
    .unwrap();

    // Main storage (chunk tables) should still have float32 vectors (4 bytes per dimension)
    // Note: vector_chunks tables store vectors as BLOBs for entire chunks
    let chunk_vector_size: i32 = db
        .query_row(
            "SELECT LENGTH(vectors) FROM test_vector_chunks00 LIMIT 1",
            [],
            |row| row.get(0),
        )
        .unwrap();

    // The chunk stores float32 vectors (4 bytes per dimension)
    // One vector of 4 dimensions = 16 bytes
    assert!(
        chunk_vector_size >= 16,
        "Main storage should contain float32 vectors (got {} bytes)",
        chunk_vector_size
    );
}

/// Test that tables with non-vector columns first work correctly
/// This tests the col_idx vs vec_col_idx bug fix
#[test]
fn test_non_vector_columns_before_vector() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    // Create table with non-vector columns before the vector column
    // This should use vec_col_idx (0) not col_idx (2) for shadow table naming
    db.execute(
        "CREATE VIRTUAL TABLE test USING vec0(
            id INTEGER,
            label TEXT,
            embedding float[4]
        )",
        [],
    )
    .unwrap();

    // Insert should work - if using col_idx (2) it would fail with
    // "no such table: test_vector_chunks02"
    db.execute(
        "INSERT INTO test(rowid, id, label, embedding) VALUES (1, 100, 'first', vec_f32('[1.0, 0.0, 0.0, 0.0]'))",
        [],
    )
    .unwrap();

    // Verify the vector was stored in the correct chunk table (chunks00, not chunks02)
    let chunk_exists: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='test_vector_chunks00'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(chunk_exists, 1, "Should use test_vector_chunks00");

    // Insert more data
    db.execute(
        "INSERT INTO test(rowid, id, label, embedding) VALUES (2, 200, 'second', vec_f32('[0.0, 1.0, 0.0, 0.0]'))",
        [],
    )
    .unwrap();

    // KNN query should work
    let mut stmt = db
        .prepare("SELECT rowid, distance FROM test WHERE embedding MATCH vec_f32('[1.0, 0.0, 0.0, 0.0]') AND k = 2 ORDER BY distance")
        .unwrap();

    let results: Vec<(i64, f64)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(results.len(), 2, "Should return k=2 results");
    assert_eq!(results[0].0, 1, "Closest should be rowid 1");
}

/// Test UPDATE with non-vector columns before vector column
#[test]
fn test_update_non_vector_columns_before_vector() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    // Create table with non-vector columns first
    db.execute(
        "CREATE VIRTUAL TABLE test USING vec0(
            id INTEGER,
            embedding float[4]
        )",
        [],
    )
    .unwrap();

    // Insert initial data
    db.execute(
        "INSERT INTO test(rowid, id, embedding) VALUES (1, 100, vec_f32('[1.0, 0.0, 0.0, 0.0]'))",
        [],
    )
    .unwrap();

    // Update the vector
    db.execute(
        "UPDATE test SET embedding = vec_f32('[0.0, 1.0, 0.0, 0.0]') WHERE rowid = 1",
        [],
    )
    .unwrap();

    // Verify the update worked via KNN query
    let mut stmt = db
        .prepare("SELECT rowid, distance FROM test WHERE embedding MATCH vec_f32('[0.0, 1.0, 0.0, 0.0]') AND k = 1 ORDER BY distance")
        .unwrap();

    let results: Vec<(i64, f64)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 1);
    // Distance should be near zero for exact match
    assert!(results[0].1 < 0.001, "Distance should be near zero");
}
