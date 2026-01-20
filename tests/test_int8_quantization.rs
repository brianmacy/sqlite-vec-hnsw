// Tests for int8 quantization support with HNSW indexing
use rusqlite::Connection;
use tempfile::TempDir;

#[test]
fn test_int8_basic_insert_and_read() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("int8_test.db");

    let db = Connection::open(&db_path).unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    // Create table with int8 vector column
    db.execute(
        "CREATE VIRTUAL TABLE vectors_int8 USING vec0(embedding int8[128])",
        [],
    )
    .unwrap();

    // Create int8 vector: values in range [-128, 127]
    let int8_values: Vec<i8> = (0..128).map(|i| (i % 256) as i8).collect();
    let vector_json = format!(
        "[{}]",
        int8_values
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );

    // Insert using vec_int8
    db.execute(
        &format!(
            "INSERT INTO vectors_int8(rowid, embedding) VALUES (1, vec_int8('{}'))",
            vector_json
        ),
        [],
    )
    .unwrap();

    // Verify we can read it back
    let embedding: Vec<u8> = db
        .query_row(
            "SELECT embedding FROM vectors_int8 WHERE rowid = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();

    assert_eq!(embedding.len(), 128, "Int8 vector should be 128 bytes");
    println!("✓ Int8 vector insert and read works");
}

#[test]
fn test_int8_quantization_from_float32() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("quantize_test.db");

    let db = Connection::open(&db_path).unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute(
        "CREATE VIRTUAL TABLE vectors_int8 USING vec0(embedding int8[128])",
        [],
    )
    .unwrap();

    // Create float32 vector, then quantize to int8
    let float_values: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();
    let vector_json = format!(
        "[{}]",
        float_values
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );

    // Insert using vec_quantize_int8
    db.execute(
        &format!(
            "INSERT INTO vectors_int8(rowid, embedding) VALUES (1, vec_quantize_int8(vec_f32('{}')))",
            vector_json
        ),
        [],
    )
    .unwrap();

    // Verify quantization worked
    let embedding: Vec<u8> = db
        .query_row(
            "SELECT embedding FROM vectors_int8 WHERE rowid = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();

    assert_eq!(embedding.len(), 128);
    println!("✓ Float32 to int8 quantization works");
}

#[test]
fn test_int8_hnsw_indexing() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("int8_hnsw_test.db");

    let db = Connection::open(&db_path).unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    // Create table with int8 vectors (should create HNSW tables automatically)
    db.execute(
        "CREATE VIRTUAL TABLE vectors_int8 USING vec0(embedding int8[128])",
        [],
    )
    .unwrap();

    // Verify HNSW shadow tables were created
    let tables: Vec<String> = db
        .prepare(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'vectors_int8%' ORDER BY name",
        )
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();

    println!("\nInt8 shadow tables created:");
    for table in &tables {
        println!("  {}", table);
    }

    assert!(
        tables.contains(&"vectors_int8_embedding_hnsw_nodes".to_string()),
        "HNSW nodes table should exist for int8"
    );
    assert!(
        tables.contains(&"vectors_int8_embedding_hnsw_edges".to_string()),
        "HNSW edges table should exist for int8"
    );
    assert!(
        tables.contains(&"vectors_int8_embedding_hnsw_meta".to_string()),
        "HNSW meta table should exist for int8"
    );

    // Insert some int8 vectors
    for i in 1..=50 {
        let int8_values: Vec<i8> = (0..128).map(|j| ((i * j + i) % 256) as i8).collect();
        let vector_json = format!(
            "[{}]",
            int8_values
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );

        db.execute(
            &format!(
                "INSERT INTO vectors_int8(rowid, embedding) VALUES ({}, vec_int8('{}'))",
                i, vector_json
            ),
            [],
        )
        .unwrap();
    }

    // Verify HNSW index was built
    let node_count: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM vectors_int8_embedding_hnsw_nodes",
            [],
            |row| row.get(0),
        )
        .unwrap();

    println!("✓ HNSW index built with {} nodes", node_count);
    assert_eq!(node_count, 50, "Should have 50 nodes in HNSW index");

    // Verify we can do KNN search with int8
    let query_values: Vec<i8> = (0..128).map(|i| ((i * 2) % 256) as i8).collect();
    let query_json = format!(
        "[{}]",
        query_values
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );

    let mut stmt = db
        .prepare(&format!(
            "SELECT rowid, distance FROM vectors_int8 \
             WHERE embedding MATCH vec_int8('{}') AND k = 5 \
             ORDER BY distance",
            query_json
        ))
        .unwrap();

    let results: Vec<(i64, f64)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();

    println!("✓ KNN search returned {} results", results.len());
    assert_eq!(results.len(), 5, "Should return k=5 results");

    println!("Top 5 nearest neighbors:");
    for (rowid, distance) in &results {
        println!("  rowid={}, distance={:.2}", rowid, distance);
    }

    println!("\n✅ Int8 HNSW indexing fully functional!");
}

#[test]
fn test_int8_insert_performance() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("int8_perf.db");

    let db = Connection::open(&db_path).unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute(
        "CREATE VIRTUAL TABLE vectors_int8 USING vec0(embedding int8[768])",
        [],
    )
    .unwrap();

    println!("\n=== Int8 Insert Performance (768D, disk I/O) ===");

    // Insert 500 int8 vectors and measure time
    let start = std::time::Instant::now();
    for i in 1..=500 {
        let int8_values: Vec<i8> = (0..768).map(|j| (((i + j) * 17) % 256) as i8).collect();
        let vector_json = format!(
            "[{}]",
            int8_values
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );

        db.execute(
            &format!(
                "INSERT INTO vectors_int8(rowid, embedding) VALUES ({}, vec_int8('{}'))",
                i, vector_json
            ),
            [],
        )
        .unwrap();
    }

    let elapsed = start.elapsed();
    let rate = 500.0 / elapsed.as_secs_f64();

    println!("Int8 (768D): {:.1} vectors/sec", rate);
    println!("Total time: {:.2}s for 500 vectors", elapsed.as_secs_f64());

    // Compare to expected C performance
    let c_int8_rate = 184.0; // From C benchmark
    let ratio = rate / c_int8_rate;
    println!("\nComparison to C:");
    println!("  C int8: 184 vec/sec");
    println!("  Rust int8: {:.1} vec/sec", rate);
    println!("  Ratio: {:.2}x", ratio);

    if ratio < 0.5 {
        println!("  ⚠️ Rust is more than 2x slower than C");
    } else if ratio < 0.8 {
        println!("  ⚠️ Rust is slower than C but within 2x");
    } else if ratio < 1.2 {
        println!("  ✅ Rust performance within 20% of C");
    } else {
        println!("  ✅ Rust faster than C!");
    }

    // Verify database size
    let metadata = std::fs::metadata(&db_path).unwrap();
    println!(
        "\nDatabase size: {:.1} MB",
        metadata.len() as f64 / 1_000_000.0
    );

    println!("\n✅ Int8 performance measured with actual disk I/O");
}

#[test]
fn test_float32_vs_int8_comparison() {
    let temp_dir = TempDir::new().unwrap();

    println!("\n=== Float32 vs Int8 Performance Comparison ===");

    // Test Float32
    let db_path_f32 = temp_dir.path().join("float32_comp.db");
    let db_f32 = Connection::open(&db_path_f32).unwrap();
    sqlite_vec_hnsw::init(&db_f32).unwrap();

    db_f32
        .execute(
            "CREATE VIRTUAL TABLE vectors_f32 USING vec0(embedding float[768])",
            [],
        )
        .unwrap();

    let start_f32 = std::time::Instant::now();
    for i in 1..=200 {
        let float_values: Vec<f32> = (0..768).map(|j| ((i + j) % 100) as f32 / 100.0).collect();
        let vector_json = format!(
            "[{}]",
            float_values
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );

        db_f32
            .execute(
                &format!(
                    "INSERT INTO vectors_f32(rowid, embedding) VALUES ({}, vec_f32('{}'))",
                    i, vector_json
                ),
                [],
            )
            .unwrap();
    }
    let elapsed_f32 = start_f32.elapsed();
    let rate_f32 = 200.0 / elapsed_f32.as_secs_f64();
    let size_f32 = std::fs::metadata(&db_path_f32).unwrap().len();

    // Test Int8
    let db_path_i8 = temp_dir.path().join("int8_comp.db");
    let db_i8 = Connection::open(&db_path_i8).unwrap();
    sqlite_vec_hnsw::init(&db_i8).unwrap();

    db_i8
        .execute(
            "CREATE VIRTUAL TABLE vectors_i8 USING vec0(embedding int8[768])",
            [],
        )
        .unwrap();

    let start_i8 = std::time::Instant::now();
    for i in 1..=200 {
        let int8_values: Vec<i8> = (0..768).map(|j| (((i + j) * 17) % 256) as i8).collect();
        let vector_json = format!(
            "[{}]",
            int8_values
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );

        db_i8
            .execute(
                &format!(
                    "INSERT INTO vectors_i8(rowid, embedding) VALUES ({}, vec_int8('{}'))",
                    i, vector_json
                ),
                [],
            )
            .unwrap();
    }
    let elapsed_i8 = start_i8.elapsed();
    let rate_i8 = 200.0 / elapsed_i8.as_secs_f64();
    let size_i8 = std::fs::metadata(&db_path_i8).unwrap().len();

    // Print comparison
    println!("\n| Metric | Float32 | Int8 | Ratio |");
    println!("|--------|---------|------|-------|");
    println!(
        "| Storage (MB) | {:.1} | {:.1} | {:.2}x |",
        size_f32 as f64 / 1_000_000.0,
        size_i8 as f64 / 1_000_000.0,
        size_f32 as f64 / size_i8 as f64
    );
    println!(
        "| Insert rate (vec/s) | {:.1} | {:.1} | {:.2}x |",
        rate_f32,
        rate_i8,
        rate_i8 / rate_f32
    );
    println!(
        "| Insert time (s) | {:.2} | {:.2} | {:.2}x |",
        elapsed_f32.as_secs_f64(),
        elapsed_i8.as_secs_f64(),
        elapsed_f32.as_secs_f64() / elapsed_i8.as_secs_f64()
    );

    println!("\n✅ Float32 vs Int8 comparison complete");
}
