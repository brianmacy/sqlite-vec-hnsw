// Diagnostic test to understand chunk allocations
use rusqlite::Connection;
use tempfile::TempDir;

#[test]
fn test_1000_vector_chunk_breakdown() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("diag.db");
    let db = Connection::open(&db_path).unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute("CREATE VIRTUAL TABLE v USING vec0(e float[768])", [])
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

    println!("\n=== Chunk Allocations (1000 vectors) ===");

    // Check _chunks table
    let chunks: Vec<(i64, i64)> = db
        .prepare("SELECT chunk_id, size FROM v_chunks")
        .unwrap()
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();

    println!("v_chunks table ({} rows):", chunks.len());
    for (id, size) in &chunks {
        println!("  Chunk {}: size={} (max 1024)", id, size);
    }

    if chunks.len() > 1 {
        println!("❌ FOUND BLOAT: Multiple chunks created for 1000 vectors!");
    } else {
        println!("✓ Only 1 chunk created");
    }

    // Check vector_chunks table
    let vec_chunks: Vec<(i64, i64)> = db
        .prepare("SELECT rowid, length(vectors) FROM v_vector_chunks00")
        .unwrap()
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();

    println!("\nv_vector_chunks00 table ({} rows):", vec_chunks.len());
    for (id, len) in &vec_chunks {
        println!("  Chunk rowid {}: {} bytes", id, len);
        let expected = 1024 * 768 * 4;
        println!(
            "    Expected: {} bytes (1024 vectors × 3,072 bytes)",
            expected
        );
        if *len != expected {
            println!("    ⚠️ Size mismatch!");
        }
    }

    if vec_chunks.len() > 1 {
        println!("❌ FOUND BLOAT: Multiple vector_chunks rows!");
    }

    // Check HNSW node count
    let node_count: i64 = db
        .query_row("SELECT COUNT(*) FROM v_e_hnsw_nodes", [], |row| row.get(0))
        .unwrap();

    println!("\nHNSW nodes: {}", node_count);
    if node_count != 1000 {
        println!("⚠️ Expected 1000 nodes, got {}", node_count);
    }

    // Summary
    let total_size = std::fs::metadata(&db_path).unwrap().len();
    println!(
        "\nTotal database: {} bytes ({} bytes/vector)",
        total_size,
        total_size / 1000
    );
    println!("C expected: ~2,929 bytes/vector");
    println!("Bloat: {:.1}x", (total_size / 1000) as f64 / 2929.0);
}
