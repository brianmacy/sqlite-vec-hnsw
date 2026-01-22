// Deep performance analysis of target configuration:
// hnsw(index_quantization=int8, M=64, ef_construction=200)

use rusqlite::Connection;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;
use tempfile::TempDir;

const EMBEDDINGS_FILE: &str = "test_data/opensanctions_embeddings.jsonl";
const VECTOR_DIM: usize = 384;

fn load_embeddings(max_vectors: usize) -> Vec<Vec<f32>> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let path = format!("{}/{}", manifest_dir, EMBEDDINGS_FILE);
    let file = File::open(&path).unwrap();
    let reader = BufReader::new(file);
    reader
        .lines()
        .take(max_vectors)
        .map(|line| serde_json::from_str(&line.unwrap()).unwrap())
        .collect()
}

fn vector_to_json(vector: &[f32]) -> String {
    format!(
        "[{}]",
        vector
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

#[test]
#[ignore]
#[allow(clippy::needless_range_loop)]
fn profile_target_config_detailed() {
    println!("\n{}", "=".repeat(80));
    println!("DETAILED PERFORMANCE ANALYSIS");
    println!("Config: hnsw(index_quantization=int8, M=64, ef_construction=200)");
    println!("{}", "=".repeat(80));

    let vectors = load_embeddings(20000);
    println!("\nLoaded {} vectors of {}D\n", vectors.len(), VECTOR_DIM);

    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("profile.db");

    let db = Connection::open(&db_path).unwrap();
    db.execute_batch(
        "PRAGMA page_size=16384;
         PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         PRAGMA cache_size=10000;
         PRAGMA temp_store=MEMORY;",
    )
    .unwrap();

    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute(
        &format!(
            "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[{}] hnsw(index_quantization=int8, M=64, ef_construction=200))",
            VECTOR_DIM
        ),
        [],
    ).unwrap();

    // Profile insert performance at different index sizes
    println!(
        "{:>10} {:>10} {:>12} {:>10} {:>12} {:>10}",
        "Vectors", "Batch", "Time (s)", "Vec/sec", "Edges", "Edges/N"
    );
    println!("{}", "-".repeat(70));

    let checkpoints = [1000, 2000, 5000, 10000, 15000, 20000];
    let mut last_checkpoint = 0;
    let mut total_time = std::time::Duration::ZERO;

    for &checkpoint in &checkpoints {
        if checkpoint > vectors.len() {
            break;
        }

        let batch_size = checkpoint - last_checkpoint;
        let start = Instant::now();

        for i in last_checkpoint..checkpoint {
            let sql = format!(
                "INSERT INTO vectors(rowid, embedding) VALUES ({}, vec_f32('{}'))",
                i + 1,
                vector_to_json(&vectors[i])
            );
            db.execute(&sql, []).unwrap();
        }

        let batch_time = start.elapsed();
        total_time += batch_time;
        let rate = batch_size as f64 / batch_time.as_secs_f64();

        let edges: i64 = db
            .query_row(
                "SELECT COUNT(*) FROM vectors_embedding_hnsw_edges",
                [],
                |r| r.get(0),
            )
            .unwrap();

        println!(
            "{:>10} {:>10} {:>12.2} {:>10.1} {:>12} {:>10.1}",
            checkpoint,
            batch_size,
            batch_time.as_secs_f64(),
            rate,
            edges,
            edges as f64 / checkpoint as f64
        );

        last_checkpoint = checkpoint;
    }

    // Overall stats
    println!("\n{}", "-".repeat(70));
    println!(
        "Total: {} vectors in {:.2}s ({:.1} vec/sec overall)",
        last_checkpoint,
        total_time.as_secs_f64(),
        last_checkpoint as f64 / total_time.as_secs_f64()
    );

    // Check database size
    let db_size = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);
    println!("Database size: {:.1} MB", db_size as f64 / 1024.0 / 1024.0);

    // Profile search performance
    println!("\n{}", "=".repeat(80));
    println!("SEARCH PERFORMANCE");
    println!("{}", "=".repeat(80));

    let num_queries = 100;
    let mut search_times = Vec::new();

    for i in 0..num_queries {
        let query = &vectors[i * 37 % last_checkpoint];
        let sql = format!(
            "SELECT rowid, distance FROM vectors WHERE embedding MATCH vec_f32('{}') AND k = 10 ORDER BY distance",
            vector_to_json(query)
        );

        let start = Instant::now();
        let mut stmt = db.prepare(&sql).unwrap();
        let _results: Vec<(i64, f64)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        search_times.push(start.elapsed().as_micros());
    }

    search_times.sort();
    let avg_us = search_times.iter().sum::<u128>() / num_queries as u128;
    let p50 = search_times[num_queries / 2];
    let p95 = search_times[95];
    let p99 = search_times[99];

    println!("\n{} queries on {} vectors:", num_queries, last_checkpoint);
    println!(
        "  Avg:  {} µs ({:.1} qps)",
        avg_us,
        1_000_000.0 / avg_us as f64
    );
    println!("  p50:  {} µs", p50);
    println!("  p95:  {} µs", p95);
    println!("  p99:  {} µs", p99);

    // Profile what's in the shadow tables
    println!("\n{}", "=".repeat(80));
    println!("SHADOW TABLE ANALYSIS");
    println!("{}", "=".repeat(80));

    let node_count: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM vectors_embedding_hnsw_nodes",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let edge_count: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM vectors_embedding_hnsw_edges",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let level_dist: Vec<(i32, i64)> = {
        let mut stmt = db.prepare(
            "SELECT level, COUNT(*) FROM vectors_embedding_hnsw_edges GROUP BY level ORDER BY level"
        ).unwrap();
        stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect()
    };

    println!("\nNodes: {}", node_count);
    println!(
        "Edges: {} ({:.1} per node)",
        edge_count,
        edge_count as f64 / node_count as f64
    );
    println!("\nEdge distribution by level:");
    for (level, count) in &level_dist {
        println!(
            "  Level {}: {} edges ({:.1}%)",
            level,
            count,
            *count as f64 / edge_count as f64 * 100.0
        );
    }

    // Print internal timing breakdown
    println!("\n{}", "=".repeat(80));
    println!("INTERNAL TIMING BREAKDOWN");
    println!("{}", "=".repeat(80));
    sqlite_vec_hnsw::hnsw::insert::print_timing_stats();

    // Sample edge query to understand I/O pattern
    println!("\n{}", "=".repeat(80));
    println!("I/O PATTERN ANALYSIS");
    println!("{}", "=".repeat(80));

    // Time 1000 edge fetches
    let start = Instant::now();
    for i in 1..=1000 {
        let _: Vec<i64> = {
            let mut stmt = db.prepare(
                "SELECT to_rowid FROM vectors_embedding_hnsw_edges WHERE from_rowid = ? AND level = 0"
            ).unwrap();
            stmt.query_map([i], |row| row.get(0))
                .unwrap()
                .filter_map(|r| r.ok())
                .collect()
        };
    }
    let edge_fetch_time = start.elapsed();
    println!(
        "\n1000 edge fetches: {:.1}ms ({:.1} µs/fetch)",
        edge_fetch_time.as_millis(),
        edge_fetch_time.as_micros() as f64 / 1000.0
    );

    // Time 1000 vector fetches
    let start = Instant::now();
    for i in 1..=1000 {
        let _: Vec<u8> = db
            .query_row(
                "SELECT vector FROM vectors_embedding_hnsw_nodes WHERE rowid = ?",
                [i],
                |r| r.get(0),
            )
            .unwrap();
    }
    let vector_fetch_time = start.elapsed();
    println!(
        "1000 vector fetches: {:.1}ms ({:.1} µs/fetch)",
        vector_fetch_time.as_millis(),
        vector_fetch_time.as_micros() as f64 / 1000.0
    );

    println!("\n{}", "=".repeat(80));
}

#[test]
#[ignore]
fn profile_insert_breakdown() {
    println!("\n{}", "=".repeat(80));
    println!("INSERT OPERATION BREAKDOWN");
    println!("Config: hnsw(index_quantization=int8, M=64, ef_construction=200)");
    println!("{}", "=".repeat(80));

    let vectors = load_embeddings(5000);
    println!("\nLoaded {} vectors\n", vectors.len());

    // Pre-convert to JSON strings to isolate SQL overhead
    let json_vectors: Vec<String> = vectors.iter().map(|v| vector_to_json(v)).collect();

    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("profile.db");

    let db = Connection::open(&db_path).unwrap();
    db.execute_batch(
        "PRAGMA page_size=16384;
         PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         PRAGMA cache_size=10000;
         PRAGMA temp_store=MEMORY;",
    )
    .unwrap();

    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute(
        &format!(
            "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[{}] hnsw(index_quantization=int8, M=64, ef_construction=200))",
            VECTOR_DIM
        ),
        [],
    ).unwrap();

    // Insert in batches and measure
    let batch_size = 100;
    let num_batches = 50;

    println!(
        "Inserting {} batches of {} vectors...\n",
        num_batches, batch_size
    );

    let mut batch_times = Vec::new();
    for batch in 0..num_batches {
        let start_idx = batch * batch_size;
        let start = Instant::now();

        for i in 0..batch_size {
            let rowid = start_idx + i + 1;
            let sql = format!(
                "INSERT INTO vectors(rowid, embedding) VALUES ({}, vec_f32('{}'))",
                rowid,
                &json_vectors[rowid - 1]
            );
            db.execute(&sql, []).unwrap();
        }

        batch_times.push(start.elapsed().as_millis());
    }

    // Analyze batch times
    let total: u128 = batch_times.iter().sum();
    let avg = total / num_batches as u128;

    println!("Batch timing (ms) - {} vectors per batch:", batch_size);
    println!("  First 5:  {:?}", &batch_times[..5]);
    println!("  Last 5:   {:?}", &batch_times[num_batches - 5..]);
    println!("  Min:      {} ms", batch_times.iter().min().unwrap());
    println!("  Max:      {} ms", batch_times.iter().max().unwrap());
    println!(
        "  Avg:      {} ms ({:.1} vec/sec)",
        avg,
        batch_size as f64 * 1000.0 / avg as f64
    );

    // Show degradation curve
    println!("\nPerformance degradation as index grows:");
    for (i, &time) in batch_times.iter().step_by(10).enumerate() {
        let vectors_in_index = (i * 10 + 1) * batch_size;
        let rate = batch_size as f64 * 1000.0 / time as f64;
        println!(
            "  {:>5} vectors: {} ms/batch ({:.1} vec/sec)",
            vectors_in_index, time, rate
        );
    }

    println!("\n{}", "=".repeat(80));
}

#[test]
#[ignore]
fn compare_sync_modes() {
    println!("\n{}", "=".repeat(80));
    println!("SQLite SYNCHRONOUS Mode Comparison");
    println!("{}", "=".repeat(80));

    let vectors = load_embeddings(5000);

    for sync_mode in ["OFF", "NORMAL", "FULL"] {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("profile.db");

        let db = Connection::open(&db_path).unwrap();
        db.execute_batch(&format!(
            "PRAGMA page_size=16384;
             PRAGMA journal_mode=WAL;
             PRAGMA synchronous={};
             PRAGMA cache_size=10000;
             PRAGMA temp_store=MEMORY;",
            sync_mode
        ))
        .unwrap();

        sqlite_vec_hnsw::init(&db).unwrap();

        db.execute(
            &format!(
                "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[{}] hnsw(index_quantization=int8, M=64, ef_construction=200))",
                VECTOR_DIM
            ),
            [],
        ).unwrap();

        let start = Instant::now();
        for (i, v) in vectors.iter().enumerate() {
            let sql = format!(
                "INSERT INTO vectors(rowid, embedding) VALUES ({}, vec_f32('{}'))",
                i + 1,
                vector_to_json(v)
            );
            db.execute(&sql, []).unwrap();
        }
        let elapsed = start.elapsed();
        let rate = vectors.len() as f64 / elapsed.as_secs_f64();

        println!(
            "SYNCHRONOUS={}: {:.1}s ({:.1} vec/sec)",
            sync_mode,
            elapsed.as_secs_f64(),
            rate
        );
    }

    println!("\n{}", "=".repeat(80));
}
