// Benchmark matching C's LargeScaleScalabilityTest
// C result: 1000 vectors in 2430ms = 2.43ms per vector, search 1.35ms
use rusqlite::Connection;
use std::time::Instant;

#[test]
fn bench_hnsw_insert_1000_vectors() {
    println!("\n=== HNSW Insert Benchmark: 1000 vectors ===");

    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    // Create table with HNSW index (matches C test configuration)
    db.execute(
        "CREATE VIRTUAL TABLE test_vectors USING vec0(embedding float[128])",
        [],
    )
    .unwrap();

    // Generate 1000 random vectors (128D like C test)
    let num_vectors = 1000;
    let dimensions = 128;

    let mut vectors = Vec::new();
    for i in 0..num_vectors {
        let vec: Vec<f32> = (0..dimensions)
            .map(|j| ((i + j) as f32 / 100.0).sin())
            .collect();
        let bytes: Vec<u8> = vec.iter().flat_map(|f| f.to_le_bytes()).collect();
        vectors.push((i + 1, bytes));
    }

    // Measure insertion time
    println!("Inserting {} vectors...", num_vectors);
    let start = Instant::now();

    db.execute("BEGIN TRANSACTION", []).unwrap();

    for (rowid, vec_bytes) in &vectors {
        db.execute(
            "INSERT INTO test_vectors(rowid, embedding) VALUES (?, ?)",
            rusqlite::params![*rowid as i64, vec_bytes],
        )
        .unwrap();
    }

    db.execute("COMMIT", []).unwrap();

    let insert_duration = start.elapsed();
    let insert_ms = insert_duration.as_millis();
    let avg_insert_ms = insert_ms as f64 / num_vectors as f64;

    println!("Inserted {} vectors in {}ms", num_vectors, insert_ms);
    println!("Average insert time: {:.2}ms per vector", avg_insert_ms);

    // Measure search time (like C test - searches for same vectors)
    let num_queries = 20;
    println!("\nPerforming {} searches...", num_queries);
    let search_start = Instant::now();

    for i in 0..num_queries {
        let query_idx = i * (num_vectors / num_queries);
        let query_bytes = &vectors[query_idx].1;

        let mut stmt = db
            .prepare("SELECT rowid, distance FROM test_vectors WHERE embedding MATCH ? AND k = 10 ORDER BY distance")
            .unwrap();

        let results: Vec<(i64, f64)> = stmt
            .query_map([query_bytes], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        // Verify we got results
        assert!(!results.is_empty(), "Query {} returned no results", i);

        // Check if exact match found at position 0
        if results[0].1 < 0.001 {
            println!(
                "Query {}: Exact match at position 0 with distance {}",
                i, results[0].1
            );
        }
    }

    let search_duration = search_start.elapsed();
    let search_ms = search_duration.as_millis();
    let avg_search_ms = search_ms as f64 / num_queries as f64;

    println!("Performed {} searches in {}ms", num_queries, search_ms);
    println!("Average search time: {:.2}ms per search", avg_search_ms);

    // Performance comparison with C
    println!("\n=== Performance Comparison ===");
    println!("C implementation:");
    println!("  Insert: 2.43 ms/vector");
    println!("  Search: 1.35 ms/search");
    println!("\nRust implementation (this run):");
    println!("  Insert: {:.2} ms/vector", avg_insert_ms);
    println!("  Search: {:.2} ms/search", avg_search_ms);
    println!("\nRatio: {:.2}x (Rust/C insert time)", avg_insert_ms / 2.43);

    // Assertions (relaxed for now - just measuring)
    // We're aiming to match C performance (~2.4ms insert, ~1.4ms search)
    println!("\n✅ Benchmark complete");
}
