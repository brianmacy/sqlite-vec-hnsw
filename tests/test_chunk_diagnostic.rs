// Diagnostic test to understand unified storage allocations
use rusqlite::Connection;
use tempfile::TempDir;

#[test]
fn test_1000_vector_unified_storage() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("diag.db");
    let db = Connection::open(&db_path).unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute("CREATE VIRTUAL TABLE v USING vec0(e float[768] hnsw())", [])
        .unwrap();
    db.execute("BEGIN", []).unwrap();

    for i in 1..=1000 {
        let vec: Vec<f32> = (0..768).map(|j| ((i + j) % 100) as f32 / 100.0).collect();
        let bytes: Vec<u8> = vec.iter().flat_map(|f| f.to_le_bytes()).collect();
        db.execute(
            "INSERT INTO v(rowid, e) VALUES (?, ?)",
            rusqlite::params![i, bytes],
        )
        .unwrap();
    }

    db.execute("COMMIT", []).unwrap();

    println!("\n=== Unified Storage Allocations (1000 vectors) ===");

    // Check unified _data table
    let data_row_count: i64 = db
        .query_row("SELECT COUNT(*) FROM v_data", [], |row| row.get(0))
        .unwrap();

    println!("v_data table: {} rows", data_row_count);
    assert_eq!(data_row_count, 1000, "Should have 1000 rows in _data table");
    println!("✓ All 1000 vectors stored in unified _data table");

    // Check HNSW node count
    let node_count: i64 = db
        .query_row("SELECT COUNT(*) FROM v_e_hnsw_nodes", [], |row| row.get(0))
        .unwrap();

    println!("\nHNSW nodes: {}", node_count);
    assert_eq!(node_count, 1000, "Should have 1000 HNSW nodes");
    println!("✓ HNSW index has 1000 nodes");

    // Summary
    let total_size = std::fs::metadata(&db_path).unwrap().len();
    println!(
        "\nTotal database: {} bytes ({} bytes/vector)",
        total_size,
        total_size / 1000
    );

    // With unified storage (no chunking overhead), storage should be more efficient
    let bytes_per_vector = total_size / 1000;
    let raw_vector_size = 768 * 4; // 3,072 bytes
    let overhead = bytes_per_vector as f64 / raw_vector_size as f64;
    println!("Raw vector size: {} bytes", raw_vector_size);
    println!("Overhead: {:.2}x", overhead);

    // With HNSW enabled, we expect ~2-3x overhead (vector in _data + vector in hnsw_nodes + edges)
    if bytes_per_vector < 10000 {
        println!("✅ Storage efficiency is acceptable (< 10KB/vector)");
    } else {
        println!("⚠️ Storage may have bloat (> 10KB/vector)");
    }
}
