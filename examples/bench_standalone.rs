// Standalone benchmark for profiling with samply
use rusqlite::Connection;
use std::time::Instant;

fn main() {
    println!("=== Standalone HNSW Insert Benchmark ===");

    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute(
        "CREATE VIRTUAL TABLE test USING vec0(embedding float[128])",
        [],
    )
    .unwrap();

    println!("Inserting 1000 vectors...");
    println!("Note: Using autocommit mode (no explicit transaction) to match C test");
    let start = Instant::now();

    // NO BEGIN TRANSACTION - matches C test which uses autocommit
    // db.execute("BEGIN", []).unwrap();

    for i in 1..=1000 {
        let vec: Vec<f32> = (0..128).map(|j| ((i + j) as f32 / 100.0).sin()).collect();
        let bytes: Vec<u8> = vec.iter().flat_map(|f| f.to_le_bytes()).collect();

        db.execute(
            "INSERT INTO test(rowid, embedding) VALUES (?, ?)",
            rusqlite::params![i as i64, bytes],
        )
        .unwrap();
    }

    // db.execute("COMMIT", []).unwrap();

    let elapsed = start.elapsed();
    let ms = elapsed.as_millis();
    let avg_ms = ms as f64 / 1000.0;

    println!("Inserted 1000 vectors in {}ms", ms);
    println!("Average: {:.2}ms per vector", avg_ms);
    println!("C implementation: 2.79ms per vector");
    println!("Ratio: {:.2}x", avg_ms / 2.79);

    // Print detailed timing breakdown
    sqlite_vec_hnsw::hnsw::insert::print_timing_stats();

    // Count edges to compare with C
    let total_edges: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM test_embedding_hnsw_edges",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);
    let level_0_edges: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM test_embedding_hnsw_edges WHERE level = 0",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);
    let max_edges_per_node: i64 = db
        .query_row(
            "SELECT MAX(cnt) FROM (SELECT from_rowid, COUNT(*) as cnt FROM test_embedding_hnsw_edges WHERE level = 0 GROUP BY from_rowid)",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    println!("\n=== EDGE STATISTICS ===");
    println!("Total edges: {}", total_edges);
    println!("Level 0 edges: {}", level_0_edges);
    println!(
        "Avg edges per node at level 0: {:.1}",
        level_0_edges as f64 / 1000.0
    );
    println!("Max edges per node at level 0: {}", max_edges_per_node);
}
