// Test C compatibility fixes
use rusqlite::Connection;
use tempfile::TempDir;

#[test]
fn test_info_table_exists_and_populated() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute("CREATE VIRTUAL TABLE test USING vec0(v float[3])", [])
        .unwrap();

    println!("\n=== Testing _info Table ===");

    // Verify table exists
    let table_exists: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='test_info'",
            [],
            |row| row.get(0),
        )
        .unwrap();

    assert_eq!(table_exists, 1, "_info table should exist");
    println!("✓ _info table exists");

    // Verify version info is populated
    let version_keys: Vec<(String, rusqlite::types::Value)> = db
        .prepare("SELECT key, value FROM test_info ORDER BY key")
        .unwrap()
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();

    println!("\n_info table contents:");
    for (key, value) in &version_keys {
        println!("  {}: {:?}", key, value);
    }

    assert_eq!(
        version_keys.len(),
        5,
        "Should have 5 info keys (including STORAGE_SCHEMA)"
    );

    // Check each key exists (values are stored as INTEGER or TEXT in SQLite)
    let keys: Vec<&str> = version_keys.iter().map(|(k, _)| k.as_str()).collect();
    assert!(keys.contains(&"CREATE_VERSION"));
    assert!(keys.contains(&"CREATE_VERSION_MAJOR"));
    assert!(keys.contains(&"CREATE_VERSION_MINOR"));
    assert!(keys.contains(&"CREATE_VERSION_PATCH"));
    assert!(keys.contains(&"STORAGE_SCHEMA"));

    println!("✓ All info keys populated correctly");
    println!("\n✅ _info table compatibility verified");
}

#[test]
fn test_unified_storage_efficiency() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("unified_storage_test.db");

    let db = Connection::open(&db_path).unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute(
        "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[768])",
        [],
    )
    .unwrap();

    println!("\n=== Testing Unified Storage Architecture ===");

    // Insert 100 vectors
    db.execute("BEGIN TRANSACTION", []).unwrap();
    for i in 1..=100 {
        let vector: Vec<f32> = (0..768).map(|j| ((i + j) % 100) as f32 / 100.0).collect();
        let bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();
        db.execute(
            "INSERT INTO vectors(rowid, embedding) VALUES (?, ?)",
            rusqlite::params![i, bytes],
        )
        .unwrap();
    }
    db.execute("COMMIT", []).unwrap();

    let db_size = std::fs::metadata(&db_path).unwrap().len();
    let bytes_per_vector = db_size / 100;

    println!("Database size: {} bytes", db_size);
    println!("Per vector: {} bytes", bytes_per_vector);

    let raw_vector_size = 768 * 4; // 3,072 bytes
    let overhead = bytes_per_vector as f64 / raw_vector_size as f64;
    println!("Overhead: {:.2}x", overhead);

    // With unified storage (no chunking), we expect simpler overhead
    if bytes_per_vector > 10000 {
        println!("⚠️ Higher than expected (> 10KB/vector)");
    } else if bytes_per_vector < 5000 {
        println!("✅ Storage looks good (< 5KB/vector)");
    } else {
        println!("✓ Acceptable overhead");
    }

    println!("\n=== Checking unified _data table ===");
    let data_row_count: i64 = db
        .query_row("SELECT COUNT(*) FROM vectors_data", [], |row| row.get(0))
        .unwrap();

    println!("Rows in _data table: {}", data_row_count);
    assert_eq!(
        data_row_count, 100,
        "Should have 100 rows in unified _data table"
    );
    println!("✓ Unified _data table has correct row count");

    println!("\n✅ Unified storage architecture verified");
}

#[test]
fn test_shadow_table_names_unified_schema() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute(
        "CREATE VIRTUAL TABLE vecs USING vec0(emb float[3] hnsw())",
        [],
    )
    .unwrap();

    println!("\n=== Verifying Unified Storage Shadow Tables ===");

    let rust_tables: Vec<String> = db
        .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'vecs_%' ORDER BY name")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();

    println!("\nShadow tables created:");
    for table in &rust_tables {
        println!("  {}", table);
    }

    // Unified storage architecture creates:
    // - {table}_data (unified data table for all columns)
    // - {table}_info (version metadata key-value store)
    // - {table}_{column}_hnsw_meta (if HNSW enabled)
    // - {table}_{column}_hnsw_nodes
    // - {table}_{column}_hnsw_edges
    // Note: No more _chunks, _rowids, _vector_chunksNN tables

    let expected = vec![
        "vecs_data",
        "vecs_emb_hnsw_edges",
        "vecs_emb_hnsw_meta",
        "vecs_emb_hnsw_nodes",
        "vecs_info",
    ];

    println!("\nExpected tables (unified storage):");
    for table in &expected {
        println!("  {}", table);
    }

    // Verify all expected tables exist
    for expected_table in &expected {
        assert!(
            rust_tables.contains(&expected_table.to_string()),
            "Missing table: {}",
            expected_table
        );
    }

    println!("\n✅ All expected shadow tables present");
}
