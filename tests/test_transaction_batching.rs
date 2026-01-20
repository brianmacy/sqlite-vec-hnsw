// Test insert performance with transaction batching (matching C benchmark)
use rusqlite::Connection;
use tempfile::TempDir;

#[test]
fn test_float32_with_transaction_batching() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("batch_test.db");

    let db = Connection::open(&db_path).unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    // Use default settings (matching C benchmark)
    // - synchronous=FULL (default)
    // - journal_mode=DELETE (default)

    db.execute(
        "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[768])",
        [],
    )
    .unwrap();

    println!("\n=== Float32 Insert with Transaction Batching (500 vectors) ===");
    println!("Settings: synchronous=FULL (default), journal_mode=DELETE (default)");

    let start = std::time::Instant::now();

    // BEGIN TRANSACTION (like C benchmark does)
    db.execute("BEGIN TRANSACTION", []).unwrap();

    for i in 1..=500 {
        let vector: Vec<f32> = (0..768).map(|j| ((i + j) % 100) as f32 / 100.0).collect();
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
                "INSERT INTO vectors(rowid, embedding) VALUES ({}, vec_f32('{}'))",
                i, vector_json
            ),
            [],
        )
        .unwrap();
    }

    // COMMIT (like C benchmark does)
    db.execute("COMMIT", []).unwrap();

    let elapsed = start.elapsed();
    let rate = 500.0 / elapsed.as_secs_f64();

    println!("Rust float32: {:.1} vec/sec", rate);
    println!("Total time: {:.2}s for 500 vectors", elapsed.as_secs_f64());

    let c_float32_rate = 162.0;
    let ratio = rate / c_float32_rate;
    println!("\nComparison to C:");
    println!("  C float32 (with transactions): 162 vec/sec");
    println!("  Rust float32 (with transactions): {:.1} vec/sec", rate);
    println!("  Ratio: {:.2}x", ratio);

    if (0.8..=1.2).contains(&ratio) {
        println!("  ✅ Within 20% of C performance!");
    } else if ratio > 1.2 {
        println!("  ✅ Faster than C!");
    } else {
        println!("  ⚠️ Slower than C");
    }

    let metadata = std::fs::metadata(&db_path).unwrap();
    println!(
        "\nDatabase size: {:.1} MB",
        metadata.len() as f64 / 1_000_000.0
    );

    println!("\n✅ Transaction batching test complete");
}

#[test]
fn test_int8_with_transaction_batching() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("batch_int8_test.db");

    let db = Connection::open(&db_path).unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute(
        "CREATE VIRTUAL TABLE vectors USING vec0(embedding int8[768])",
        [],
    )
    .unwrap();

    println!("\n=== Int8 Insert with Transaction Batching (500 vectors) ===");

    let start = std::time::Instant::now();

    db.execute("BEGIN TRANSACTION", []).unwrap();

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
                "INSERT INTO vectors(rowid, embedding) VALUES ({}, vec_int8('{}'))",
                i, vector_json
            ),
            [],
        )
        .unwrap();
    }

    db.execute("COMMIT", []).unwrap();

    let elapsed = start.elapsed();
    let rate = 500.0 / elapsed.as_secs_f64();

    println!("Rust int8: {:.1} vec/sec", rate);
    println!("Total time: {:.2}s for 500 vectors", elapsed.as_secs_f64());

    let c_int8_rate = 184.0;
    let ratio = rate / c_int8_rate;
    println!("\nComparison to C:");
    println!("  C int8 (with transactions): 184 vec/sec");
    println!("  Rust int8 (with transactions): {:.1} vec/sec", rate);
    println!("  Ratio: {:.2}x", ratio);

    if (0.8..=1.2).contains(&ratio) {
        println!("  ✅ Within 20% of C performance!");
    } else if ratio > 1.2 {
        println!("  ✅ Faster than C!");
    } else {
        println!("  ⚠️ Slower than C");
    }

    println!("\n✅ Int8 transaction batching test complete");
}

#[test]
fn test_wal_mode_with_transactions() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("wal_batch_test.db");

    let db = Connection::open(&db_path).unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    // Enable WAL mode (C benchmark uses journal_mode=DELETE by default)
    let journal_mode: String = db
        .query_row("PRAGMA journal_mode=WAL", [], |row| row.get(0))
        .unwrap();
    println!("\n=== WAL Mode with Transaction Batching ===");
    println!("Journal mode: {}", journal_mode);

    db.execute(
        "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[768])",
        [],
    )
    .unwrap();

    let start = std::time::Instant::now();

    db.execute("BEGIN TRANSACTION", []).unwrap();

    for i in 1..=500 {
        let vector: Vec<f32> = (0..768).map(|j| ((i + j) % 100) as f32 / 100.0).collect();
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
                "INSERT INTO vectors(rowid, embedding) VALUES ({}, vec_f32('{}'))",
                i, vector_json
            ),
            [],
        )
        .unwrap();
    }

    db.execute("COMMIT", []).unwrap();

    let elapsed = start.elapsed();
    let rate = 500.0 / elapsed.as_secs_f64();

    println!("WAL + transactions: {:.1} vec/sec", rate);
    println!("Total time: {:.2}s for 500 vectors", elapsed.as_secs_f64());

    println!("\n✅ WAL mode test complete");
}
