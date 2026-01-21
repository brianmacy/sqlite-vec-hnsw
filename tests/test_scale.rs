use rusqlite::Connection;

#[test]
fn test_scale_10k_vectors() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    // Create table with higher dimensional vectors and HNSW enabled
    db.execute(
        "CREATE VIRTUAL TABLE embeddings USING vec0(vector float[128] hnsw())",
        [],
    )
    .unwrap();

    println!("\n🔄 Inserting 10,000 vectors (128D)...");

    // Insert 10K vectors
    let start = std::time::Instant::now();
    for i in 0..10_000 {
        // Generate a simple test vector
        let vector: Vec<f32> = (0..128).map(|j| (i * 128 + j) as f32 / 10000.0).collect();
        let vector_json = format!(
            "[{}]",
            vector
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );

        db.execute(
            &format!(
                "INSERT INTO embeddings(rowid, vector) VALUES ({}, vec_f32('{}'))",
                i + 1,
                vector_json
            ),
            [],
        )
        .unwrap();

        if (i + 1) % 1000 == 0 {
            println!("  Inserted {} vectors", i + 1);
        }
    }
    let insert_duration = start.elapsed();

    println!(
        "✓ Inserted 10,000 vectors in {:.2}s ({:.0} vec/sec)",
        insert_duration.as_secs_f64(),
        10_000.0 / insert_duration.as_secs_f64()
    );

    // Verify count
    let count: i64 = db
        .query_row("SELECT COUNT(*) FROM embeddings_rowids", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(count, 10_000, "Should have 10,000 vectors");

    // Check HNSW index was built
    let node_count: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM embeddings_vector_hnsw_nodes",
            [],
            |row| row.get(0),
        )
        .unwrap();
    println!("📊 HNSW nodes: {}", node_count);
    assert_eq!(node_count, 10_000, "HNSW should have 10,000 nodes");

    // Test a KNN query
    println!("\n🔎 Testing KNN query on 10K vectors...");
    let query_vector: Vec<f32> = (0..128).map(|j| j as f32 / 128.0).collect();
    let vector_json = format!(
        "[{}]",
        query_vector
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );

    let query_start = std::time::Instant::now();
    let mut stmt = db
        .prepare(&format!(
            "SELECT rowid, distance FROM embeddings WHERE vector MATCH vec_f32('{}') AND k = 10 ORDER BY distance",
            vector_json
        ))
        .unwrap();

    let results: Vec<(i64, f64)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let query_duration = query_start.elapsed();

    println!(
        "✓ Query completed in {:.2}ms",
        query_duration.as_secs_f64() * 1000.0
    );
    println!("  Top 10 results:");
    for (i, (rowid, distance)) in results.iter().enumerate() {
        println!("    {}. rowid={}, distance={:.3}", i + 1, rowid, distance);
    }

    assert_eq!(results.len(), 10, "Should return k=10 results");

    // Verify query latency is reasonable (< 100ms for 10K vectors)
    assert!(
        query_duration.as_millis() < 100,
        "Query should be fast (< 100ms), got {}ms",
        query_duration.as_millis()
    );
}

#[test]
#[ignore] // Run with: cargo test test_scale_100k -- --ignored --nocapture
fn test_scale_100k_vectors() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    // Create table with production-like dimensions and HNSW enabled
    db.execute(
        "CREATE VIRTUAL TABLE embeddings USING vec0(vector float[768] hnsw())",
        [],
    )
    .unwrap();

    println!("\n🔄 Inserting 100,000 vectors (768D)...");
    println!("⚠️  This will take several minutes...");

    // Insert 100K vectors
    let start = std::time::Instant::now();
    for i in 0..100_000 {
        // Generate a simple test vector
        let vector: Vec<f32> = (0..768).map(|j| (i * 768 + j) as f32 / 100000.0).collect();
        let vector_json = format!(
            "[{}]",
            vector
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );

        db.execute(
            &format!(
                "INSERT INTO embeddings(rowid, vector) VALUES ({}, vec_f32('{}'))",
                i + 1,
                vector_json
            ),
            [],
        )
        .unwrap();

        if (i + 1) % 10_000 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = (i + 1) as f64 / elapsed;
            println!(
                "  Inserted {} vectors ({:.2}s, {:.0} vec/sec)",
                i + 1,
                elapsed,
                rate
            );
        }
    }
    let insert_duration = start.elapsed();

    println!(
        "✓ Inserted 100,000 vectors in {:.2}s ({:.0} vec/sec)",
        insert_duration.as_secs_f64(),
        100_000.0 / insert_duration.as_secs_f64()
    );

    // Verify count
    let count: i64 = db
        .query_row("SELECT COUNT(*) FROM embeddings_rowids", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(count, 100_000, "Should have 100,000 vectors");

    // Check HNSW index
    let node_count: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM embeddings_vector_hnsw_nodes",
            [],
            |row| row.get(0),
        )
        .unwrap();
    println!("📊 HNSW nodes: {}", node_count);

    // Test KNN queries
    println!("\n🔎 Testing KNN queries on 100K vectors...");

    // Run 10 queries and measure average latency
    let mut total_duration = std::time::Duration::ZERO;
    for q in 0..10 {
        let query_vector: Vec<f32> = (0..768).map(|j| (q * 768 + j) as f32 / 1000.0).collect();
        let vector_json = format!(
            "[{}]",
            query_vector
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );

        let query_start = std::time::Instant::now();
        let mut stmt = db
            .prepare(&format!(
                "SELECT rowid, distance FROM embeddings WHERE vector MATCH vec_f32('{}') AND k = 10 ORDER BY distance",
                vector_json
            ))
            .unwrap();

        let results: Vec<(i64, f64)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let query_duration = query_start.elapsed();

        total_duration += query_duration;
        assert_eq!(results.len(), 10, "Should return k=10 results");
    }

    let avg_latency = total_duration.as_secs_f64() / 10.0;
    println!(
        "✓ Average query latency: {:.2}ms (10 queries)",
        avg_latency * 1000.0
    );

    // Verify query latency is reasonable (< 10ms for 100K vectors with HNSW)
    assert!(
        avg_latency < 0.010,
        "Average query should be fast (< 10ms), got {:.2}ms",
        avg_latency * 1000.0
    );

    println!("\n✅ Scale test passed: 100K vectors with efficient KNN queries!");
}
