// Test with prepared statements (matching C's approach exactly)
use rusqlite::Connection;

#[test]
fn test_prepared_statement_float32() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute(
        "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[768])",
        [],
    )
    .unwrap();

    println!("\n=== Prepared Statement Test (matching C exactly) ===");
    println!("Using: INSERT INTO vectors(rowid, embedding) VALUES (?, ?)");
    println!("With parameter binding (no SQL string rebuilding)");

    let start = std::time::Instant::now();

    db.execute("BEGIN TRANSACTION", []).unwrap();

    // Prepare statement ONCE (like C does)
    let mut stmt = db
        .prepare("INSERT INTO vectors(rowid, embedding) VALUES (?, ?)")
        .unwrap();

    for i in 1..=500 {
        // Generate vector
        let vector: Vec<f32> = (0..768).map(|j| ((i + j) % 100) as f32 / 100.0).collect();

        // Convert f32 vector to raw bytes (skipping JSON parsing overhead!)
        let bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Bind parameters and execute
        stmt.execute(rusqlite::params![i, bytes]).unwrap();
    }

    drop(stmt); // Finalize statement

    db.execute("COMMIT", []).unwrap();

    let elapsed = start.elapsed();
    let rate = 500.0 / elapsed.as_secs_f64();

    println!("\nRust (prepared statement): {:.1} vec/sec", rate);
    println!("C (prepared statement): 162 vec/sec");
    println!("Ratio: {:.2}x", rate / 162.0);

    if rate >= 130.0 {
        println!("✅ WITHIN 20% OF C!");
    } else {
        println!("⚠️ Still slower - but much better than before");
    }
}

#[test]
fn test_comparison_prepared_vs_formatted() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    println!("\n=== Direct Comparison: Prepared vs Formatted SQL ===");

    // Test 1: Formatted SQL (current approach)
    db.execute(
        "CREATE VIRTUAL TABLE vectors1 USING vec0(embedding float[768])",
        [],
    )
    .unwrap();

    let start1 = std::time::Instant::now();
    db.execute("BEGIN TRANSACTION", []).unwrap();

    for i in 1..=200 {
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
                "INSERT INTO vectors1(rowid, embedding) VALUES ({}, vec_f32('{}'))",
                i, vector_json
            ),
            [],
        )
        .unwrap();
    }

    db.execute("COMMIT", []).unwrap();
    let elapsed1 = start1.elapsed();
    let rate1 = 200.0 / elapsed1.as_secs_f64();

    // Test 2: Prepared statement (C approach)
    db.execute(
        "CREATE VIRTUAL TABLE vectors2 USING vec0(embedding float[768])",
        [],
    )
    .unwrap();

    let start2 = std::time::Instant::now();
    db.execute("BEGIN TRANSACTION", []).unwrap();

    let mut stmt = db
        .prepare("INSERT INTO vectors2(rowid, embedding) VALUES (?, ?)")
        .unwrap();

    for i in 1..=200 {
        let vector: Vec<f32> = (0..768).map(|j| ((i + j) % 100) as f32 / 100.0).collect();
        let bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();

        stmt.execute(rusqlite::params![i, bytes]).unwrap();
    }

    drop(stmt);
    db.execute("COMMIT", []).unwrap();
    let elapsed2 = start2.elapsed();
    let rate2 = 200.0 / elapsed2.as_secs_f64();

    println!("\nFormatted SQL:      {:.1} vec/sec", rate1);
    println!("Prepared statement: {:.1} vec/sec", rate2);
    println!("Speedup:            {:.2}x", rate2 / rate1);

    println!("\n=== Analysis ===");
    println!("Formatted SQL overhead:");
    println!("  - format!() string building every insert");
    println!("  - SQL parsing every insert");
    println!("  - Statement preparation every insert");
    println!("\nPrepared statement benefits:");
    println!("  - Parse SQL once, reuse statement");
    println!("  - Parameter binding (faster)");
    println!("  - Statement cache reuse");

    if rate2 / rate1 > 2.0 {
        println!(
            "\n🎯 FOUND THE BOTTLENECK! Prepared statements are {:.1}x faster",
            rate2 / rate1
        );
    }
}
