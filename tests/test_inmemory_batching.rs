// Test in-memory performance with transaction batching (matching C benchmark exactly)
use rusqlite::Connection;

#[test]
fn test_inmemory_float32_with_transactions() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute(
        "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[768])",
        [],
    )
    .unwrap();

    println!("\n=== IN-MEMORY + Transaction Batching (matching C benchmark) ===");
    println!("Database: :memory: (no disk I/O)");
    println!("Transactions: BEGIN ... COMMIT");

    let start = std::time::Instant::now();

    // BEGIN TRANSACTION (matching C)
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

    // COMMIT (matching C)
    db.execute("COMMIT", []).unwrap();

    let elapsed = start.elapsed();
    let rate = 500.0 / elapsed.as_secs_f64();

    println!("\nRust (in-memory + transactions): {:.1} vec/sec", rate);
    println!("C (in-memory + transactions): 162 vec/sec");

    let ratio = rate / 162.0;
    println!("Ratio: {:.2}x", ratio);

    if (0.8..=1.2).contains(&ratio) {
        println!("✅ Within 20% of C performance!");
    } else if ratio > 1.2 {
        println!("✅ FASTER than C!");
    } else {
        println!("⚠️ Slower than C ({:.0}% of C speed)", ratio * 100.0);
    }

    println!("\n=== CORRECTED COMPARISON ===");
    println!("Previous comparison was WRONG:");
    println!("  - C: 162 vec/sec (in-memory)");
    println!("  - Rust: 19.6 vec/sec (DISK-based)");
    println!("  - Wrong conclusion: 8.3x slower");
    println!("\nCorrect comparison (both in-memory):");
    println!("  - C: 162 vec/sec");
    println!("  - Rust: {:.1} vec/sec", rate);
    println!("  - Actual ratio: {:.2}x", ratio);
}

#[test]
fn test_inmemory_int8_with_transactions() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute(
        "CREATE VIRTUAL TABLE vectors USING vec0(embedding int8[768])",
        [],
    )
    .unwrap();

    println!("\n=== IN-MEMORY Int8 + Transaction Batching ===");

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

    println!("Rust int8 (in-memory + transactions): {:.1} vec/sec", rate);
    println!("C int8 (in-memory + transactions): 184 vec/sec");
    println!("Ratio: {:.2}x", rate / 184.0);
}
