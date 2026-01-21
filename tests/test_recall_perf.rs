// Test HNSW performance vs recall tradeoff
use rusqlite::Connection;
use std::time::Instant;

fn test_with_params(ef_construction: i32, ef_search: i32) -> (f64, f64, f64, f64) {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    const DIMENSIONS: usize = 128;
    const NUM_VECTORS: i64 = 1000;
    const K: i64 = 10;

    // Create table with L2 distance to match ground truth calculation
    db.execute(
        "CREATE VIRTUAL TABLE test_hnsw USING vec0(embedding float[128] hnsw(distance=l2))",
        [],
    )
    .unwrap();

    // Update HNSW parameters (single-row schema)
    db.execute(
        &format!(
            "UPDATE test_hnsw_embedding_hnsw_meta SET ef_construction = {}, ef_search = {} WHERE id = 1",
            ef_construction, ef_search
        ),
        [],
    )
    .unwrap();

    // Insert vectors
    let insert_start = Instant::now();
    db.execute("BEGIN", []).unwrap();
    for i in 0..NUM_VECTORS {
        let vec: Vec<f32> = (0..DIMENSIONS)
            .map(|j| (i * 100 + j as i64) as f32 / 1000.0)
            .collect();
        let bytes: Vec<u8> = vec.iter().flat_map(|f| f.to_le_bytes()).collect();
        let rowid = i + 1;
        db.execute(
            "INSERT INTO test_hnsw(rowid, embedding) VALUES (?, ?)",
            rusqlite::params![rowid, bytes],
        )
        .unwrap();
    }
    db.execute("COMMIT", []).unwrap();
    let insert_time = insert_start.elapsed().as_secs_f64();

    // Query vector
    let query_vec: Vec<f32> = vec![0.5; DIMENSIONS];
    let query_bytes: Vec<u8> = query_vec.iter().flat_map(|f| f.to_le_bytes()).collect();

    // Ground truth
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
    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;

    // Calculate recall
    let matches = hnsw_results
        .iter()
        .filter(|r| ground_truth.contains(r))
        .count();
    let recall = matches as f64 / K as f64;

    (
        insert_time,
        search_time,
        recall,
        insert_time * 1000.0 / NUM_VECTORS as f64,
    )
}

#[test]
fn test_perf_vs_recall_tradeoff() {
    println!("\n=== HNSW Performance vs Recall Tradeoff ===");
    println!("1000 vectors, 128D, k=10\n");
    println!(
        "{:>15} {:>12} {:>12} {:>12} {:>10}",
        "ef_construct", "insert(s)", "ms/vector", "search(ms)", "recall"
    );
    println!("{}", "-".repeat(65));

    let configs = [
        (50, 50),
        (100, 100),
        (200, 200),
        (400, 200), // default
    ];

    for (ef_c, ef_s) in configs {
        let (insert, search, recall, ms_per_vec) = test_with_params(ef_c, ef_s);
        println!(
            "{:>15} {:>12.2} {:>12.2} {:>12.2} {:>9.0}%",
            ef_c,
            insert,
            ms_per_vec,
            search,
            recall * 100.0
        );
    }
}
