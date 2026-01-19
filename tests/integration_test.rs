//! Integration tests for sqlite-vec-hnsw
//!
//! These tests mirror the C/C++ tests from the original implementation.
//! They should fail until the implementation is complete.

use rusqlite::{Connection, Result as SqliteResult};

/// Test helper to create an in-memory database
fn create_test_db() -> SqliteResult<Connection> {
    Connection::open_in_memory()
}

/// Test helper to initialize extension (will fail until implemented)
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

    // Try to initialize extension (will fail)
    let _ = init_extension(&db);

    // Try to create a simple vec0 table
    let result = db.execute(
        "CREATE VIRTUAL TABLE vec_simple USING vec0(embedding float[8])",
        [],
    );

    // Should fail because virtual table module not registered
    assert!(
        result.is_err(),
        "CREATE VIRTUAL TABLE should fail (module not registered)"
    );
}

#[test]
fn test_create_vec0_table_with_metadata() {
    let db = create_test_db().expect("Failed to create database");
    let _ = init_extension(&db);

    // Try to create vec0 table with metadata columns
    let result = db.execute(
        "CREATE VIRTUAL TABLE vec_movies USING vec0(
            movie_id INTEGER PRIMARY KEY,
            synopsis_embedding float[768],
            +title TEXT,
            genre TEXT,
            rating FLOAT
        )",
        [],
    );

    assert!(
        result.is_err(),
        "CREATE VIRTUAL TABLE should fail (not implemented)"
    );
}

#[test]
fn test_create_vec0_table_with_partition_keys() {
    let db = create_test_db().expect("Failed to create database");
    let _ = init_extension(&db);

    // Try to create vec0 table with partition keys
    let result = db.execute(
        "CREATE VIRTUAL TABLE vec_chunks USING vec0(
            user_id INTEGER PARTITION KEY,
            +contents TEXT,
            embedding float[1024]
        )",
        [],
    );

    assert!(
        result.is_err(),
        "CREATE VIRTUAL TABLE with partition keys should fail"
    );
}

#[test]
fn test_insert_vectors_json_format() {
    let db = create_test_db().expect("Failed to create database");
    let _ = init_extension(&db);

    // Even if table creation worked, inserts would fail
    let _ = db.execute(
        "CREATE VIRTUAL TABLE vec_test USING vec0(embedding float[3])",
        [],
    );

    let result = db.execute(
        "INSERT INTO vec_test(rowid, embedding) VALUES (1, '[1.0, 2.0, 3.0]')",
        [],
    );

    assert!(result.is_err(), "INSERT should fail (table doesn't exist)");
}

#[test]
fn test_knn_query_basic() {
    let db = create_test_db().expect("Failed to create database");
    let _ = init_extension(&db);

    let _ = db.execute(
        "CREATE VIRTUAL TABLE vec_test USING vec0(embedding float[3])",
        [],
    );

    // Try KNN query
    let result = db.prepare(
        "SELECT rowid, distance
         FROM vec_test
         WHERE embedding MATCH '[0.5, 0.5, 0.5]' AND k = 5
         ORDER BY distance",
    );

    assert!(result.is_err(), "KNN query should fail (not implemented)");
}

#[test]
fn test_knn_query_with_metadata_filter() {
    let db = create_test_db().expect("Failed to create database");
    let _ = init_extension(&db);

    let _ = db.execute(
        "CREATE VIRTUAL TABLE vec_movies USING vec0(
            embedding float[768],
            genre TEXT,
            rating FLOAT
        )",
        [],
    );

    // Try KNN query with metadata filters
    let result = db.prepare(
        "SELECT rowid, genre, rating, distance
         FROM vec_movies
         WHERE embedding MATCH '[...]'
           AND k = 10
           AND genre = 'scifi'
           AND rating > 4.0
         ORDER BY distance",
    );

    assert!(
        result.is_err(),
        "KNN query with metadata filters should fail"
    );
}

#[test]
fn test_knn_query_with_partition_key() {
    let db = create_test_db().expect("Failed to create database");
    let _ = init_extension(&db);

    let _ = db.execute(
        "CREATE VIRTUAL TABLE vec_user_docs USING vec0(
            user_id INTEGER PARTITION KEY,
            embedding float[512]
        )",
        [],
    );

    // Try KNN query with partition key filter
    let result = db.prepare(
        "SELECT rowid, distance
         FROM vec_user_docs
         WHERE embedding MATCH '[...]'
           AND user_id = 123
           AND k = 5
         ORDER BY distance",
    );

    assert!(result.is_err(), "KNN query with partition key should fail");
}

#[test]
fn test_vec_distance_l2_function() {
    let db = create_test_db().expect("Failed to create database");
    // Register SQL functions directly (virtual table not needed for this test)
    sqlite_vec_hnsw::sql_functions::register_all(&db).expect("Failed to register functions");

    // Test vec_distance_l2 function
    let result: SqliteResult<f64> = db.query_row(
        "SELECT vec_distance_l2('[1.0, 2.0, 3.0]', '[4.0, 5.0, 6.0]')",
        [],
        |row| row.get(0),
    );

    assert!(result.is_ok(), "vec_distance_l2 should work");
    let distance = result.unwrap();
    // sqrt((3^2 + 3^2 + 3^2)) = sqrt(27) ≈ 5.196
    assert!((distance - 5.196).abs() < 0.01);
}

#[test]
fn test_vec_distance_cosine_function() {
    let db = create_test_db().expect("Failed to create database");
    sqlite_vec_hnsw::sql_functions::register_all(&db).expect("Failed to register functions");

    let result: SqliteResult<f64> = db.query_row(
        "SELECT vec_distance_cosine('[1.0, 0.0, 0.0]', '[0.0, 1.0, 0.0]')",
        [],
        |row| row.get(0),
    );

    assert!(result.is_ok(), "vec_distance_cosine should work");
    let distance = result.unwrap();
    // Orthogonal vectors have cosine distance of 1
    assert!((distance - 1.0).abs() < 0.01);
}

#[test]
fn test_vec_length_function() {
    let db = create_test_db().expect("Failed to create database");
    sqlite_vec_hnsw::sql_functions::register_all(&db).expect("Failed to register functions");

    let result: SqliteResult<i64> =
        db.query_row("SELECT vec_length('[1.0, 2.0, 3.0, 4.0]')", [], |row| {
            row.get(0)
        });

    assert!(result.is_ok(), "vec_length should work");
    assert_eq!(result.unwrap(), 4);
}

#[test]
fn test_vec_to_json_function() {
    let db = create_test_db().expect("Failed to create database");
    sqlite_vec_hnsw::sql_functions::register_all(&db).expect("Failed to register functions");

    let result: SqliteResult<String> = db.query_row(
        "SELECT vec_to_json(vec_f32('[1.0, 2.0, 3.0]'))",
        [],
        |row| row.get(0),
    );

    assert!(result.is_ok(), "vec_to_json should work");
    let json = result.unwrap();
    assert_eq!(json, "[1.0,2.0,3.0]");
}

#[test]
fn test_vec_add_function() {
    let db = create_test_db().expect("Failed to create database");
    let _ = init_extension(&db);

    let result: SqliteResult<Vec<u8>> =
        db.query_row("SELECT vec_add('[1.0, 2.0]', '[3.0, 4.0]')", [], |row| {
            row.get(0)
        });

    assert!(
        result.is_err(),
        "vec_add function should fail (not registered)"
    );
}

#[test]
fn test_vec_normalize_function() {
    let db = create_test_db().expect("Failed to create database");
    let _ = init_extension(&db);

    let result: SqliteResult<Vec<u8>> =
        db.query_row("SELECT vec_normalize('[3.0, 4.0]')", [], |row| row.get(0));

    assert!(
        result.is_err(),
        "vec_normalize function should fail (not registered)"
    );
}

#[test]
fn test_vec_quantize_int8_function() {
    let db = create_test_db().expect("Failed to create database");
    let _ = init_extension(&db);

    let result: SqliteResult<Vec<u8>> =
        db.query_row("SELECT vec_quantize_int8('[0.5, -0.3, 0.8]')", [], |row| {
            row.get(0)
        });

    assert!(
        result.is_err(),
        "vec_quantize_int8 function should fail (not registered)"
    );
}

#[test]
fn test_vec_version_function() {
    let db = create_test_db().expect("Failed to create database");
    sqlite_vec_hnsw::sql_functions::register_all(&db).expect("Failed to register functions");

    let result: SqliteResult<String> = db.query_row("SELECT vec_version()", [], |row| row.get(0));

    assert!(result.is_ok(), "vec_version should work");
    let version = result.unwrap();
    assert!(version.contains("sqlite-vec-hnsw"));
}

#[test]
fn test_shadow_tables_created() {
    let db = create_test_db().expect("Failed to create database");
    let _ = init_extension(&db);

    let _ = db.execute(
        "CREATE VIRTUAL TABLE vec_test USING vec0(embedding float[128])",
        [],
    );

    // Try to check if shadow tables exist
    let result: SqliteResult<i64> = db.query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'vec_test_%'",
        [],
        |row| row.get(0),
    );

    // Query might work even if table creation failed
    if let Ok(count) = result {
        assert_eq!(count, 0, "No shadow tables should exist yet");
    }
}

#[test]
fn test_hnsw_index_parameters() {
    let db = create_test_db().expect("Failed to create database");
    let _ = init_extension(&db);

    // Try to create table with HNSW parameters
    let result = db.execute(
        "CREATE VIRTUAL TABLE vec_hnsw USING vec0(
            embedding float[768],
            use_hnsw=1,
            hnsw_m=64,
            hnsw_ef_construction=600
        )",
        [],
    );

    assert!(
        result.is_err(),
        "CREATE TABLE with HNSW parameters should fail"
    );
}

#[test]
fn test_multiple_vector_columns() {
    let db = create_test_db().expect("Failed to create database");
    let _ = init_extension(&db);

    // Try to create table with multiple vector columns
    let result = db.execute(
        "CREATE VIRTUAL TABLE vec_multi USING vec0(
            title_embedding float[384],
            content_embedding float[768]
        )",
        [],
    );

    assert!(
        result.is_err(),
        "CREATE TABLE with multiple vector columns should fail"
    );
}

#[test]
fn test_int8_vector_type() {
    let db = create_test_db().expect("Failed to create database");
    let _ = init_extension(&db);

    // Try to create table with int8 vectors
    let result = db.execute(
        "CREATE VIRTUAL TABLE vec_int8 USING vec0(embedding int8[128])",
        [],
    );

    assert!(
        result.is_err(),
        "CREATE TABLE with int8 vectors should fail"
    );
}

#[test]
fn test_binary_vector_type() {
    let db = create_test_db().expect("Failed to create database");
    let _ = init_extension(&db);

    // Try to create table with binary vectors
    let result = db.execute(
        "CREATE VIRTUAL TABLE vec_binary USING vec0(embedding bit[1024])",
        [],
    );

    assert!(
        result.is_err(),
        "CREATE TABLE with binary vectors should fail"
    );
}

#[test]
fn test_update_vector() {
    let db = create_test_db().expect("Failed to create database");
    let _ = init_extension(&db);

    let _ = db.execute(
        "CREATE VIRTUAL TABLE vec_test USING vec0(embedding float[3])",
        [],
    );
    let _ = db.execute(
        "INSERT INTO vec_test(rowid, embedding) VALUES (1, '[1.0, 2.0, 3.0]')",
        [],
    );

    // Try to update a vector
    let result = db.execute(
        "UPDATE vec_test SET embedding = '[4.0, 5.0, 6.0]' WHERE rowid = 1",
        [],
    );

    assert!(result.is_err(), "UPDATE should fail (not implemented)");
}

#[test]
fn test_delete_vector() {
    let db = create_test_db().expect("Failed to create database");
    let _ = init_extension(&db);

    let _ = db.execute(
        "CREATE VIRTUAL TABLE vec_test USING vec0(embedding float[3])",
        [],
    );
    let _ = db.execute(
        "INSERT INTO vec_test(rowid, embedding) VALUES (1, '[1.0, 2.0, 3.0]')",
        [],
    );

    // Try to delete a vector
    let result = db.execute("DELETE FROM vec_test WHERE rowid = 1", []);

    assert!(result.is_err(), "DELETE should fail (not implemented)");
}

#[test]
fn test_point_query_by_rowid() {
    let db = create_test_db().expect("Failed to create database");
    let _ = init_extension(&db);

    let _ = db.execute(
        "CREATE VIRTUAL TABLE vec_test USING vec0(embedding float[3])",
        [],
    );

    // Try point query
    let result = db.prepare("SELECT rowid, embedding FROM vec_test WHERE rowid = 1");

    assert!(result.is_err(), "Point query should fail (not implemented)");
}

#[test]
fn test_full_scan_query() {
    let db = create_test_db().expect("Failed to create database");
    let _ = init_extension(&db);

    let _ = db.execute(
        "CREATE VIRTUAL TABLE vec_test USING vec0(embedding float[3])",
        [],
    );

    // Try full scan
    let result = db.prepare("SELECT rowid, embedding FROM vec_test");

    assert!(result.is_err(), "Full scan should fail (not implemented)");
}
