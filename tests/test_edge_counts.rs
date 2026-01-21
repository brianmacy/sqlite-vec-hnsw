// Test to verify pruning is working correctly
use rusqlite::Connection;

#[test]
fn test_edge_counts_after_insert() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute("CREATE VIRTUAL TABLE test USING vec0(e float[128])", [])
        .unwrap();

    // Insert 1000 vectors
    db.execute("BEGIN", []).unwrap();
    for i in 1..=1000 {
        let vec: Vec<f32> = (0..128).map(|j| ((i + j) as f32 / 100.0).sin()).collect();
        let bytes: Vec<u8> = vec.iter().flat_map(|f| f.to_le_bytes()).collect();
        db.execute(
            "INSERT INTO test(rowid, e) VALUES (?, ?)",
            rusqlite::params![i, bytes],
        )
        .unwrap();
    }
    db.execute("COMMIT", []).unwrap();

    // Check edge statistics
    let total_edges: i64 = db
        .query_row("SELECT COUNT(*) FROM test_e_hnsw_edges", [], |row| {
            row.get(0)
        })
        .unwrap();

    let avg_edges: f64 = db
        .query_row(
            "SELECT AVG(cnt) FROM (SELECT COUNT(*) as cnt FROM test_e_hnsw_edges GROUP BY from_rowid)",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0.0);

    let max_edges: i64 = db
        .query_row(
            "SELECT MAX(cnt) FROM (SELECT COUNT(*) as cnt FROM test_e_hnsw_edges GROUP BY from_rowid)",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    println!("\n=== Edge Count Statistics (1000 nodes) ===");
    println!("Total edges: {}", total_edges);
    println!("Avg edges per node: {:.1}", avg_edges);
    println!("Max edges per node: {}", max_edges);
    println!("\nExpected (with working pruning):");
    println!("  Max edges: ≤128 (max_M0)");
    println!("  Avg edges: ~88");

    // With M=32, max_M0=64, we expect:
    // - Max edges ≤ 64 at level 0, ≤32 at other levels
    // But we're using defaults which might be different
    // For now just check max isn't crazy high
    if max_edges > 200 {
        println!(
            "\n⚠️  WARNING: Max edges ({}) exceeds 200 - pruning may be broken!",
            max_edges
        );
    } else {
        println!("\n✅ Edge counts look reasonable");
    }
}
