/// Profile insert performance to identify bottlenecks
///
/// Run with: cargo run --release --example profile_insert
use rusqlite::Connection;
use std::time::Instant;
use tempfile::TempDir;

fn main() {
    println!("\n=== Insert Path Profiling ===\n");

    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("profile.db");

    let db = Connection::open(&db_path).unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    // Enable WAL mode for better performance
    let _: String = db
        .query_row("PRAGMA journal_mode=WAL", [], |row| row.get(0))
        .unwrap();

    // Disable synchronous for profiling (unsafe but faster)
    db.execute("PRAGMA synchronous=OFF", []).unwrap();

    println!("Creating table...");
    let create_start = Instant::now();
    db.execute(
        "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[768])",
        [],
    )
    .unwrap();
    println!(
        "  Table created in {:.3}s\n",
        create_start.elapsed().as_secs_f64()
    );

    // Profile different stages of insert
    let num_vectors = 100;
    println!("Profiling {} inserts...\n", num_vectors);

    let mut json_gen_time = 0.0;
    let mut sql_exec_time = 0.0;

    let overall_start = Instant::now();

    for i in 1..=num_vectors {
        // Time: Generate vector data and JSON
        let json_start = Instant::now();
        let vector: Vec<f32> = (0..768).map(|j| ((i + j) % 100) as f32 / 100.0).collect();
        let vector_json = format!(
            "[{}]",
            vector
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        json_gen_time += json_start.elapsed().as_secs_f64();

        // Time: SQL execution (includes BLOB write, HNSW insert, etc.)
        let sql_start = Instant::now();
        db.execute(
            &format!(
                "INSERT INTO vectors(rowid, embedding) VALUES ({}, vec_f32('{}'))",
                i, vector_json
            ),
            [],
        )
        .unwrap();
        let sql_elapsed = sql_start.elapsed().as_secs_f64();
        sql_exec_time += sql_elapsed;

        // Print detailed timing for every 10th insert
        if i % 10 == 0 {
            println!("  Insert {}: {:.3}s", i, sql_elapsed);
        }
    }

    let total_time = overall_start.elapsed().as_secs_f64();

    println!("\n=== Timing Breakdown ===");
    println!("Total time:       {:.3}s", total_time);
    println!(
        "JSON generation:  {:.3}s ({:.1}%)",
        json_gen_time,
        json_gen_time / total_time * 100.0
    );
    println!(
        "SQL execution:    {:.3}s ({:.1}%)",
        sql_exec_time,
        sql_exec_time / total_time * 100.0
    );
    println!(
        "Other overhead:   {:.3}s ({:.1}%)",
        total_time - json_gen_time - sql_exec_time,
        (total_time - json_gen_time - sql_exec_time) / total_time * 100.0
    );
    println!(
        "\nInsert rate: {:.1} vec/sec",
        num_vectors as f64 / total_time
    );

    // Now profile with transaction batching
    println!("\n=== Testing Transaction Batching ===\n");

    let batch_db_path = temp_dir.path().join("batch.db");
    let batch_db = Connection::open(&batch_db_path).unwrap();
    sqlite_vec_hnsw::init(&batch_db).unwrap();

    let _: String = batch_db
        .query_row("PRAGMA journal_mode=WAL", [], |row| row.get(0))
        .unwrap();
    batch_db.execute("PRAGMA synchronous=OFF", []).unwrap();

    batch_db
        .execute(
            "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[768])",
            [],
        )
        .unwrap();

    let batch_start = Instant::now();

    // Begin transaction
    batch_db.execute("BEGIN TRANSACTION", []).unwrap();

    for i in 1..=num_vectors {
        let vector: Vec<f32> = (0..768).map(|j| ((i + j) % 100) as f32 / 100.0).collect();
        let vector_json = format!(
            "[{}]",
            vector
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );

        batch_db
            .execute(
                &format!(
                    "INSERT INTO vectors(rowid, embedding) VALUES ({}, vec_f32('{}'))",
                    i, vector_json
                ),
                [],
            )
            .unwrap();
    }

    // Commit transaction
    batch_db.execute("COMMIT", []).unwrap();

    let batch_elapsed = batch_start.elapsed().as_secs_f64();
    let batch_rate = num_vectors as f64 / batch_elapsed;

    println!("Batch insert rate: {:.1} vec/sec", batch_rate);
    println!(
        "Speedup: {:.1}x",
        batch_rate / (num_vectors as f64 / total_time)
    );

    // Compare different PRAGMA settings
    println!("\n=== Testing PRAGMA Settings ===\n");

    for (mode, desc) in &[
        ("OFF", "No sync (fastest, unsafe)"),
        ("NORMAL", "Sync on critical moments"),
        ("FULL", "Sync on every commit (default, safest)"),
    ] {
        let pragma_db_path = temp_dir.path().join(format!("pragma_{}.db", mode));
        let pragma_db = Connection::open(&pragma_db_path).unwrap();
        sqlite_vec_hnsw::init(&pragma_db).unwrap();

        let _: String = pragma_db
            .query_row("PRAGMA journal_mode=WAL", [], |row| row.get(0))
            .unwrap();
        pragma_db
            .execute(&format!("PRAGMA synchronous={}", mode), [])
            .unwrap();

        pragma_db
            .execute(
                "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[768])",
                [],
            )
            .unwrap();

        let pragma_start = Instant::now();
        pragma_db.execute("BEGIN TRANSACTION", []).unwrap();

        for i in 1..=50 {
            let vector: Vec<f32> = (0..768).map(|j| ((i + j) % 100) as f32 / 100.0).collect();
            let vector_json = format!(
                "[{}]",
                vector
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );

            pragma_db
                .execute(
                    &format!(
                        "INSERT INTO vectors(rowid, embedding) VALUES ({}, vec_f32('{}'))",
                        i, vector_json
                    ),
                    [],
                )
                .unwrap();
        }

        pragma_db.execute("COMMIT", []).unwrap();
        let pragma_elapsed = pragma_start.elapsed().as_secs_f64();
        let pragma_rate = 50.0 / pragma_elapsed;

        println!("{}: {:.1} vec/sec - {}", mode, pragma_rate, desc);
    }

    println!("\n=== Summary ===");
    println!("The bottleneck is likely in SQL execution, which includes:");
    println!("  1. vec_f32() JSON parsing");
    println!("  2. Shadow table BLOB writes");
    println!("  3. HNSW node/edge insertion");
    println!("  4. Transaction overhead (auto-commit per insert)");
    println!("\nTo reach C performance (~162 vec/sec), need:");
    println!("  - Transaction batching (5-10x improvement)");
    println!("  - Prepared statement caching (2-5x improvement)");
    println!("  - Binary vector format instead of JSON parsing");
}
