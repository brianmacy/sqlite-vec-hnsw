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

    assert_eq!(version_keys.len(), 4, "Should have 4 version keys");

    // Check each key exists (values are stored as INTEGER or TEXT in SQLite)
    let keys: Vec<&str> = version_keys.iter().map(|(k, _)| k.as_str()).collect();
    assert!(keys.contains(&"CREATE_VERSION"));
    assert!(keys.contains(&"CREATE_VERSION_MAJOR"));
    assert!(keys.contains(&"CREATE_VERSION_MINOR"));
    assert!(keys.contains(&"CREATE_VERSION_PATCH"));

    println!("✓ All version keys populated correctly");
    println!("\n✅ _info table compatibility verified");
}

#[test]
fn test_storage_with_1024_chunk_size() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("chunk_1024_test.db");

    let db = Connection::open(&db_path).unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute(
        "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[768])",
        [],
    )
    .unwrap();

    println!("\n=== Testing Storage with chunk_size=1024 (C default) ===");

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
    println!("C expected: ~2,929 bytes/vector");

    let raw_vector_size = 768 * 4; // 3,072 bytes
    let overhead = bytes_per_vector as f64 / raw_vector_size as f64;
    println!("Overhead: {:.2}x", overhead);

    // With chunk_size=1024 and storing full vectors in HNSW nodes,
    // we expect ~2-3x overhead (not 5-6x like before)
    if bytes_per_vector > 10000 {
        println!("⚠️ Still has bloat (> 10KB/vector)");
        println!("Expected with HNSW: ~6-8KB/vector (vector + HNSW node + edges + metadata)");
    } else if bytes_per_vector < 5000 {
        println!("✅ Storage looks good (< 5KB/vector)");
    } else {
        println!("✓ Acceptable overhead");
    }

    println!("\n=== Checking chunk allocation ===");
    let chunk_info: (i64, i64, i64) = db
        .query_row(
            "SELECT chunk_id, size, length(validity) FROM vectors_chunks LIMIT 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();

    println!("Chunk ID: {}", chunk_info.0);
    println!("Chunk size (current vectors): {}", chunk_info.1);
    println!("Validity bitmap: {} bytes", chunk_info.2);

    // With chunk_size=1024, validity should be 128 bytes (1024 bits / 8)
    assert_eq!(
        chunk_info.2, 128,
        "Validity bitmap should be 128 bytes for chunk_size=1024"
    );
    println!("✓ Validity bitmap size matches C (128 bytes)");

    println!("\n✅ Chunk size compatibility verified");
}

#[test]
fn test_shadow_table_names_match_c() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute(
        "CREATE VIRTUAL TABLE vecs USING vec0(emb float[3] hnsw())",
        [],
    )
    .unwrap();

    println!("\n=== Comparing Shadow Table Names with C ===");

    let rust_tables: Vec<String> = db
        .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'vecs_%' ORDER BY name")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();

    println!("\nRust shadow tables:");
    for table in &rust_tables {
        println!("  {}", table);
    }

    // C creates these tables:
    // - {table}_chunks
    // - {table}_info
    // - {table}_rowids
    // - {table}_vector_chunks00 (renamed from _{column}_chunks00)
    // - {table}_{column}_hnsw_meta (if HNSW enabled)
    // - {table}_{column}_hnsw_nodes
    // - {table}_{column}_hnsw_edges
    // - {table}_{column}_hnsw_levels

    let expected = vec![
        "vecs_chunks",
        "vecs_emb_hnsw_edges",
        "vecs_emb_hnsw_levels",
        "vecs_emb_hnsw_meta",
        "vecs_emb_hnsw_nodes",
        "vecs_info",
        "vecs_rowids",
        "vecs_vector_chunks00",
    ];

    println!("\nExpected tables (from C):");
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
