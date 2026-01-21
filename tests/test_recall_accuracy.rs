// Test HNSW recall accuracy vs brute force ground truth
// Ported from test_SqliteVec_PluginCore_HNSWRecallQuality.cpp
use rusqlite::Connection;
use std::time::Instant;

#[test]
fn test_hnsw_recall_quality() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    const DIMENSIONS: usize = 128;
    const NUM_VECTORS: i64 = 1000;
    const K: i64 = 10;

    println!("\n=== Testing HNSW Recall Quality ===");
    println!(
        "Dimensions: {}, Vectors: {}, k: {}",
        DIMENSIONS, NUM_VECTORS, K
    );

    // Create table with HNSW index
    db.execute(
        "CREATE VIRTUAL TABLE test_hnsw USING vec0(embedding float[128])",
        [],
    )
    .unwrap();

    // Insert vectors using same formula as C test: (i * 100 + j) / 1000.0
    // where i = 0..999 (rowid - 1), j = 0..127 (dimension index)
    println!("Inserting {} vectors...", NUM_VECTORS);
    let insert_start = Instant::now();
    db.execute("BEGIN", []).unwrap();
    for i in 0..NUM_VECTORS {
        let vec: Vec<f32> = (0..DIMENSIONS)
            .map(|j| (i * 100 + j as i64) as f32 / 1000.0)
            .collect();
        let bytes: Vec<u8> = vec.iter().flat_map(|f| f.to_le_bytes()).collect();
        let rowid = i + 1; // rowids are 1-based
        db.execute(
            "INSERT INTO test_hnsw(rowid, embedding) VALUES (?, ?)",
            rusqlite::params![rowid, bytes],
        )
        .unwrap();
    }
    db.execute("COMMIT", []).unwrap();
    let insert_elapsed = insert_start.elapsed();
    println!(
        "✓ Inserted {} vectors in {:.2}s ({:.2}ms/vector)",
        NUM_VECTORS,
        insert_elapsed.as_secs_f64(),
        insert_elapsed.as_secs_f64() * 1000.0 / NUM_VECTORS as f64
    );

    // Debug: Check HNSW state
    let hnsw_node_count: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM test_hnsw_embedding_hnsw_nodes",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);
    let hnsw_edge_count: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM test_hnsw_embedding_hnsw_edges WHERE level = 0",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);
    println!(
        "HNSW state: {} nodes, {} edges (level 0)",
        hnsw_node_count, hnsw_edge_count
    );

    // Query vector: all 0.5 (same as C test)
    let query_vec: Vec<f32> = vec![0.5; DIMENSIONS];
    let query_bytes: Vec<u8> = query_vec.iter().flat_map(|f| f.to_le_bytes()).collect();

    // Get ground truth via brute force calculation
    let mut distances: Vec<(i64, f32)> = (0..NUM_VECTORS)
        .map(|i| {
            let rowid = i + 1;
            let vec: Vec<f32> = (0..DIMENSIONS)
                .map(|j| (i * 100 + j as i64) as f32 / 1000.0)
                .collect();
            let dist: f32 = query_vec
                .iter()
                .zip(vec.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            (rowid, dist)
        })
        .collect();
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let ground_truth: Vec<i64> = distances.iter().take(K as usize).map(|(r, _)| *r).collect();

    println!("\nGround truth (top {}): {:?}", K, ground_truth);

    // HNSW search
    let search_start = Instant::now();
    let mut stmt = db
        .prepare("SELECT rowid FROM test_hnsw WHERE embedding MATCH ? AND k = ? ORDER BY distance")
        .unwrap();
    let hnsw_results: Vec<i64> = stmt
        .query_map(rusqlite::params![query_bytes, K], |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    let search_elapsed = search_start.elapsed();

    println!("HNSW results (top {}): {:?}", K, hnsw_results);
    println!(
        "Search time: {:.2}ms",
        search_elapsed.as_secs_f64() * 1000.0
    );

    // Calculate recall@k
    let matches = hnsw_results
        .iter()
        .filter(|r| ground_truth.contains(r))
        .count();
    let recall = matches as f64 / K as f64;

    println!("\n=== Recall Summary ===");
    println!("Matches: {}/{}", matches, K);
    println!("HNSW Recall@{}: {:.1}%", K, recall * 100.0);

    // HNSW recall should be at least 95% - anything less indicates a critical bug
    assert!(
        recall >= 0.95,
        "HNSW recall should be at least 95%, got {:.1}%",
        recall * 100.0
    );

    println!("✅ Recall {:.1}% meets 95% threshold", recall * 100.0);
}
