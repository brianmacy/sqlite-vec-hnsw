/// Profile insert performance for samply analysis
///
/// Run with: cargo run --release --example samply_profile
use rusqlite::Connection;
use std::time::Instant;
use tempfile::TempDir;

fn main() {
    println!("\n=== Insert Profiling (Zero-Copy Distance) ===\n");

    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("profile.db");

    let db = Connection::open(&db_path).unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    // Optimal SQLite settings
    db.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         PRAGMA cache_size=10000;
         PRAGMA temp_store=MEMORY;
         PRAGMA page_size=16384;",
    )
    .unwrap();

    println!("Creating table with 128D vectors...");
    db.execute(
        "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[128])",
        [],
    )
    .unwrap();

    // Profile 2000 inserts for meaningful profiling data
    let num_vectors = 2000;
    println!("Inserting {} vectors...\n", num_vectors);

    let overall_start = Instant::now();

    for i in 1..=num_vectors {
        let vector: Vec<f32> = (0..128).map(|j| ((i + j) % 100) as f32 / 100.0).collect();
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

        if i % 500 == 0 {
            let rate = i as f64 / overall_start.elapsed().as_secs_f64();
            println!("  {} inserts, {:.0} vec/sec", i, rate);
        }
    }

    let total_time = overall_start.elapsed().as_secs_f64();

    println!("\n=== Results ===");
    println!("Total time:   {:.3}s", total_time);
    println!(
        "Insert rate:  {:.1} vec/sec",
        num_vectors as f64 / total_time
    );

    // Print the timing stats from hnsw::insert
    sqlite_vec_hnsw::hnsw::insert::print_timing_stats();

    // Print detailed search_layer breakdown
    sqlite_vec_hnsw::hnsw::search::print_search_timing_stats();
}
