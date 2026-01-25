// Test reading a database created by C's szvec extension
//
// NOTE: This test is ignored because the Rust implementation has migrated to
// a unified storage architecture (_data table) which is incompatible with
// the old C chunked storage schema (_chunks, _rowids, _vector_chunks).
// Existing databases created with the old C version would need migration.
use rusqlite::Connection;

#[test]
#[ignore = "Old C-created databases use incompatible chunked storage schema"]
fn test_rust_reads_c_created_database() {
    println!("\n=== Testing Rust Reading C-Created Database ===");
    println!("Database: /tmp/c_created.db (created by C's szvec extension)");
    println!("NOTE: This test is expected to fail - schema has changed to unified storage");

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

    // Old C schema expected these tables (no longer created with unified storage):
    // - test_vectors_chunks
    // - test_vectors_rowids
    // - test_vectors_vector_chunks00

    // New unified schema creates:
    // - test_vectors_data (unified storage)
    // - test_vectors_info (metadata)

    println!("NOTE: Old C databases need migration to new unified schema");
}
