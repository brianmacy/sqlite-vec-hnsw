use rusqlite::Connection;

#[test]
fn test_shadow_tables_created() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    // Create a virtual table with HNSW enabled
    db.execute(
        "CREATE VIRTUAL TABLE test_vec USING vec0(embedding float[3] hnsw())",
        [],
    )
    .unwrap();

    // Query for all tables
    let mut stmt = db
        .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        .unwrap();
    let tables: Vec<String> = stmt
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();

    println!("\nAll shadow tables:");
    for table in &tables {
        if table.starts_with("test_vec") {
            println!("  {}", table);
        }
    }

    // Check that expected shadow tables exist (unified storage architecture)
    assert!(
        tables.iter().any(|t| t == "test_vec_data"),
        "Missing test_vec_data (unified storage)"
    );
    assert!(
        tables.iter().any(|t| t == "test_vec_info"),
        "Missing test_vec_info"
    );

    // Check HNSW shadow tables
    assert!(
        tables.iter().any(|t| t == "test_vec_embedding_hnsw_nodes"),
        "Missing HNSW nodes table"
    );
    assert!(
        tables.iter().any(|t| t == "test_vec_embedding_hnsw_edges"),
        "Missing HNSW edges table"
    );
    assert!(
        tables.iter().any(|t| t == "test_vec_embedding_hnsw_meta"),
        "Missing HNSW meta table"
    );
}

#[test]
fn test_data_persisted_in_shadow_tables() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute(
        "CREATE VIRTUAL TABLE docs USING vec0(embedding float[3])",
        [],
    )
    .unwrap();

    // Insert a vector
    db.execute(
        "INSERT INTO docs(rowid, embedding) VALUES (1, vec_f32('[1.0, 0.0, 0.0]'))",
        [],
    )
    .unwrap();

    // Check unified _data table
    let data_count: i64 = db
        .query_row("SELECT COUNT(*) FROM docs_data", [], |row| row.get(0))
        .unwrap();
    assert_eq!(data_count, 1, "Should have 1 row in _data table");

    // Check that we can read the vector blob from _data
    let vec_blob: Vec<u8> = db
        .query_row("SELECT vec00 FROM docs_data WHERE rowid = 1", [], |row| {
            row.get(0)
        })
        .unwrap();

    println!(
        "\nRowid 1 stored in unified _data table: {} bytes",
        vec_blob.len()
    );
    assert_eq!(vec_blob.len(), 12, "3 floats = 12 bytes");

    // Verify we can read the vector back via virtual table (now returns JSON string)
    let embedding_json: String = db
        .query_row("SELECT embedding FROM docs WHERE rowid = 1", [], |row| {
            row.get(0)
        })
        .unwrap();

    // Parse JSON and verify we have 3 float values
    let trimmed = embedding_json.trim_start_matches('[').trim_end_matches(']');
    let floats: Vec<f32> = trimmed
        .split(',')
        .map(|s| s.trim().parse::<f32>().unwrap())
        .collect();
    println!("Vector has {} float values", floats.len());
    assert_eq!(floats.len(), 3, "3 float32 values");
}

#[test]
fn test_drop_table_cleans_up_shadow_tables() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    // Create a virtual table with HNSW enabled and non-vector columns
    db.execute(
        "CREATE VIRTUAL TABLE cleanup_test USING vec0(id INTEGER, name TEXT, embedding float[3] hnsw())",
        [],
    )
    .unwrap();

    // Insert some data
    db.execute(
        "INSERT INTO cleanup_test(rowid, id, name, embedding) VALUES (1, 100, 'test', vec_f32('[1.0, 0.0, 0.0]'))",
        [],
    )
    .unwrap();

    // Get list of shadow tables before drop
    let tables_before: Vec<String> = db
        .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'cleanup_test%' ORDER BY name")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();

    println!("\nShadow tables BEFORE DROP:");
    for table in &tables_before {
        println!("  {}", table);
    }
    assert!(
        !tables_before.is_empty(),
        "Should have shadow tables before drop"
    );

    // Drop the virtual table
    db.execute("DROP TABLE cleanup_test", []).unwrap();

    // Get list of shadow tables after drop
    let tables_after: Vec<String> = db
        .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'cleanup_test%' ORDER BY name")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();

    println!("\nShadow tables AFTER DROP:");
    for table in &tables_after {
        println!("  {}", table);
    }

    // All shadow tables should be cleaned up
    assert!(
        tables_after.is_empty(),
        "Shadow tables should be cleaned up after DROP TABLE, but found: {:?}",
        tables_after
    );

    // Verify we can recreate the table (would fail if shadow tables remain)
    db.execute(
        "CREATE VIRTUAL TABLE cleanup_test USING vec0(id INTEGER, name TEXT, embedding float[3] hnsw())",
        [],
    )
    .unwrap();

    println!("\n✅ DROP TABLE properly cleans up all shadow tables");
}
