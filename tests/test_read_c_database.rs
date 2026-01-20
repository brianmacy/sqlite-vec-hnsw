// Test reading a database created by C's szvec extension
use rusqlite::Connection;

#[test]
fn test_rust_reads_c_created_database() {
    println!("\n=== Testing Rust Reading C-Created Database ===");
    println!("Database: /tmp/c_created.db (created by C's szvec extension)");

    // Open the C-created database
    let db = Connection::open("/tmp/c_created.db").unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    println!("✓ Opened C-created database");

    // Check shadow tables exist
    let tables: Vec<String> = db
        .prepare(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'test_vectors%' ORDER BY name",
        )
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();

    println!("\nShadow tables found:");
    for table in &tables {
        println!("  {}", table);
    }

    // Verify expected tables exist
    assert!(tables.contains(&"test_vectors_chunks".to_string()));
    assert!(tables.contains(&"test_vectors_info".to_string()));
    assert!(tables.contains(&"test_vectors_rowids".to_string()));
    assert!(tables.contains(&"test_vectors_vector_chunks00".to_string()));

    println!("✓ All expected shadow tables present");

    // Read shadow tables directly (without virtual table access)
    let rowid_count: i64 = db
        .query_row("SELECT COUNT(*) FROM test_vectors_rowids", [], |row| {
            row.get(0)
        })
        .unwrap();

    println!("\n✓ Shadow table row count: {}", rowid_count);
    assert_eq!(rowid_count, 50, "Should have 50 rows in _rowids table");

    // Read chunk info
    let chunk_info: (i64, i64) = db
        .query_row(
            "SELECT chunk_id, size FROM test_vectors_chunks LIMIT 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();

    println!(
        "✓ Chunk info: chunk_id={}, size={}",
        chunk_info.0, chunk_info.1
    );

    // Read a specific rowid mapping
    let mapping: (i64, i64) = db
        .query_row(
            "SELECT chunk_id, chunk_offset FROM test_vectors_rowids WHERE rowid = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();

    println!(
        "✓ Rowid 1 mapping: chunk_id={}, chunk_offset={}",
        mapping.0, mapping.1
    );

    // Try to read vector data from vector_chunks table using BLOB API
    use std::io::Read;
    let mut blob = db
        .blob_open(
            "main",
            "test_vectors_vector_chunks00",
            "vectors",
            chunk_info.0,
            true,
        )
        .unwrap();

    let vector_size = 128 * 4; // 128 float32s
    let byte_offset = (mapping.1 as usize) * vector_size;

    let mut vector_data = vec![0u8; vector_size];
    use std::io::Seek;
    blob.seek(std::io::SeekFrom::Start(byte_offset as u64))
        .unwrap();
    blob.read_exact(&mut vector_data).unwrap();

    println!(
        "✓ Read vector from BLOB at offset {}: {} bytes",
        byte_offset,
        vector_data.len()
    );

    // Verify it's not all zeros
    let non_zero = vector_data.iter().any(|&b| b != 0);
    assert!(non_zero, "Vector should have non-zero data");

    println!("✓ Vector data is valid (not all zeros)");

    // Check _info table
    let info_count: i64 = db
        .query_row("SELECT COUNT(*) FROM test_vectors_info", [], |row| {
            row.get(0)
        })
        .unwrap();

    println!("✓ Info table has {} entries", info_count);

    println!("\n✅ Rust can successfully read C-created database!");
}
