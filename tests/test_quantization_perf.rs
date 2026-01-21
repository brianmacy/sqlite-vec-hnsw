//! Performance and recall comparison tests: float32 vs int8 index quantization
//!
//! Compares:
//! - Insert throughput
//! - Search throughput
//! - Recall accuracy
//! - Storage size
//!
//! Run with: cargo test --test test_quantization_perf -- --nocapture

use rusqlite::Connection;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;

const DIMENSIONS: usize = 128;
const NUM_VECTORS: i64 = 5000;
const K: i64 = 10;
const NUM_QUERIES: usize = 100;

/// Generate deterministic test vector with values spread across [-0.9, 0.9]
/// Values are spread to make good use of int8 quantization range [-127, 127]
fn generate_vector(index: usize, dimensions: usize) -> Vec<f32> {
    (0..dimensions)
        .map(|j| {
            // Use formula that creates unique patterns for each index
            // Map to range [-0.9, 0.9] to stay within quantization bounds
            let raw = ((index * 137 + j * 79 + 11) % 999983) as f64 / 999983.0;
            (raw * 1.8 - 0.9) as f32 // Map [0,1] to [-0.9, 0.9]
        })
        .collect()
}

/// Convert vector to bytes for SQLite blob
fn vector_to_bytes(vec: &[f32]) -> Vec<u8> {
    vec.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Format vector as JSON for SQL
fn vector_to_json(vector: &[f32]) -> String {
    format!(
        "[{}]",
        vector
            .iter()
            .map(|v| format!("{:.6}", v))
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// Compute L2 distance between two vectors
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Results from a single benchmark run
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct BenchmarkResult {
    name: String,
    insert_time_secs: f64,
    insert_throughput: f64,
    search_time_ms: f64,
    search_throughput: f64,
    recall: f64,
    hnsw_storage_bytes: i64,
    main_storage_bytes: i64,
}

/// Run benchmark with specified quantization mode
fn run_benchmark(use_int8_quantization: bool) -> BenchmarkResult {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    let name = if use_int8_quantization {
        "int8 quantization"
    } else {
        "float32 (no quantization)"
    };

    // Create table with or without index_quantization
    let create_sql = if use_int8_quantization {
        format!(
            "CREATE VIRTUAL TABLE test USING vec0(embedding float[{}] index_quantization=int8)",
            DIMENSIONS
        )
    } else {
        format!(
            "CREATE VIRTUAL TABLE test USING vec0(embedding float[{}])",
            DIMENSIONS
        )
    };
    db.execute(&create_sql, []).unwrap();

    // Insert vectors
    let insert_start = Instant::now();
    db.execute("BEGIN", []).unwrap();
    for i in 0..NUM_VECTORS {
        let vec = generate_vector(i as usize, DIMENSIONS);
        let bytes = vector_to_bytes(&vec);
        let rowid = i + 1;
        db.execute(
            "INSERT INTO test(rowid, embedding) VALUES (?, ?)",
            rusqlite::params![rowid, bytes],
        )
        .unwrap();
    }
    db.execute("COMMIT", []).unwrap();
    let insert_time = insert_start.elapsed().as_secs_f64();
    let insert_throughput = NUM_VECTORS as f64 / insert_time;

    // Generate query vectors and compute ground truth
    let query_vectors: Vec<Vec<f32>> = (0..NUM_QUERIES)
        .map(|i| generate_vector(i * 17 + 500, DIMENSIONS))
        .collect();

    // Compute brute-force ground truth for each query
    let ground_truths: Vec<Vec<i64>> = query_vectors
        .iter()
        .map(|query| {
            let mut distances: Vec<(i64, f32)> = (0..NUM_VECTORS)
                .map(|i| {
                    let rowid = i + 1;
                    let vec = generate_vector(i as usize, DIMENSIONS);
                    let dist = l2_distance(query, &vec);
                    (rowid, dist)
                })
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            distances.iter().take(K as usize).map(|(r, _)| *r).collect()
        })
        .collect();

    // HNSW search benchmark
    let search_start = Instant::now();
    let mut total_recall = 0.0;

    for (query_vec, ground_truth) in query_vectors.iter().zip(ground_truths.iter()) {
        let query_bytes = vector_to_bytes(query_vec);

        let mut stmt = db
            .prepare("SELECT rowid FROM test WHERE embedding MATCH ? AND k = ? ORDER BY distance")
            .unwrap();
        let hnsw_results: Vec<i64> = stmt
            .query_map(rusqlite::params![query_bytes, K], |row| row.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();

        let matches = hnsw_results
            .iter()
            .filter(|r| ground_truth.contains(r))
            .count();
        total_recall += matches as f64 / K as f64;
    }

    let search_time_ms = search_start.elapsed().as_secs_f64() * 1000.0;
    let search_throughput = NUM_QUERIES as f64 / (search_time_ms / 1000.0);
    let avg_recall = total_recall / NUM_QUERIES as f64;

    // Measure storage sizes
    let hnsw_storage: i64 = db
        .query_row(
            "SELECT SUM(LENGTH(vector)) FROM test_embedding_hnsw_nodes",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    let main_storage: i64 = db
        .query_row(
            "SELECT SUM(LENGTH(vectors)) FROM test_vector_chunks00",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    BenchmarkResult {
        name: name.to_string(),
        insert_time_secs: insert_time,
        insert_throughput,
        search_time_ms,
        search_throughput,
        recall: avg_recall,
        hnsw_storage_bytes: hnsw_storage,
        main_storage_bytes: main_storage,
    }
}

#[test]
fn test_quantization_recall_comparison() {
    println!("\n=== Index Quantization Recall Comparison ===");
    println!(
        "Config: {} vectors, {}D, k={}, {} queries\n",
        NUM_VECTORS, DIMENSIONS, K, NUM_QUERIES
    );

    let float32_result = run_benchmark(false);
    let int8_result = run_benchmark(true);

    println!("{:<25} {:>12} {:>12}", "", "float32", "int8");
    println!("{}", "-".repeat(52));
    println!(
        "{:<25} {:>11.1}% {:>11.1}%",
        "Recall@10:",
        float32_result.recall * 100.0,
        int8_result.recall * 100.0
    );
    println!(
        "{:<25} {:>12.2} {:>12.2}",
        "Insert time (s):", float32_result.insert_time_secs, int8_result.insert_time_secs
    );
    println!(
        "{:<25} {:>12.0} {:>12.0}",
        "Insert throughput (vec/s):",
        float32_result.insert_throughput,
        int8_result.insert_throughput
    );
    println!(
        "{:<25} {:>12.2} {:>12.2}",
        "Search time (ms):", float32_result.search_time_ms, int8_result.search_time_ms
    );
    println!(
        "{:<25} {:>12.0} {:>12.0}",
        "Search throughput (q/s):", float32_result.search_throughput, int8_result.search_throughput
    );
    println!(
        "{:<25} {:>12} {:>12}",
        "HNSW storage (bytes):", float32_result.hnsw_storage_bytes, int8_result.hnsw_storage_bytes
    );

    // Calculate storage savings
    if float32_result.hnsw_storage_bytes > 0 {
        let savings = (1.0
            - int8_result.hnsw_storage_bytes as f64 / float32_result.hnsw_storage_bytes as f64)
            * 100.0;
        println!("{:<25} {:>12} {:>11.0}%", "Storage savings:", "-", savings);
    }

    println!("\n=== Summary ===");

    // int8 should have comparable recall (>90%)
    assert!(
        int8_result.recall >= 0.90,
        "int8 recall should be >= 90%, got {:.1}%",
        int8_result.recall * 100.0
    );

    // int8 should use less HNSW storage (approximately 75% savings for 128D)
    if float32_result.hnsw_storage_bytes > 0 {
        let savings_pct = (1.0
            - int8_result.hnsw_storage_bytes as f64 / float32_result.hnsw_storage_bytes as f64)
            * 100.0;
        assert!(
            savings_pct >= 70.0,
            "int8 should save >= 70% storage, got {:.1}%",
            savings_pct
        );
        println!("Storage: int8 saves {:.0}% HNSW index storage", savings_pct);
    }

    println!(
        "Recall: float32={:.1}%, int8={:.1}%",
        float32_result.recall * 100.0,
        int8_result.recall * 100.0
    );

    println!("\n[OK] Quantization comparison test passed!");
}

/// Multi-threaded stress test configuration
struct ThreadedStressConfig {
    num_insert_threads: usize,
    num_search_threads: usize,
    vectors_per_thread: usize,
    searches_per_thread: usize,
    dimensions: usize,
    k: usize,
}

/// Statistics for threaded test
#[derive(Default)]
struct ThreadedStats {
    inserts_completed: AtomicU64,
    searches_completed: AtomicU64,
    insert_errors: AtomicU64,
    search_errors: AtomicU64,
}

/// Opens a connection with proper settings
fn open_threaded_connection(db_path: &PathBuf) -> Result<Connection, String> {
    let db = Connection::open(db_path).map_err(|e| e.to_string())?;
    db.busy_timeout(Duration::from_secs(60))
        .map_err(|e| e.to_string())?;
    db.execute_batch("PRAGMA cache_size=5000; PRAGMA temp_store=MEMORY;")
        .map_err(|e| e.to_string())?;
    sqlite_vec_hnsw::init(&db).map_err(|e| e.to_string())?;
    Ok(db)
}

/// Insert worker for threaded test
fn insert_worker_quantized(
    db_path: PathBuf,
    thread_id: usize,
    config: Arc<ThreadedStressConfig>,
    stats: Arc<ThreadedStats>,
    stop_flag: Arc<AtomicBool>,
) {
    let db = match open_threaded_connection(&db_path) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("Insert thread {} failed to open: {}", thread_id, e);
            return;
        }
    };

    let base_rowid = thread_id * config.vectors_per_thread + 1;

    for i in 0..config.vectors_per_thread {
        if stop_flag.load(Ordering::Relaxed) {
            break;
        }

        let rowid = base_rowid + i;
        let vector = generate_vector(rowid, config.dimensions);
        let vector_json = vector_to_json(&vector);

        let sql = format!(
            "INSERT INTO vectors(rowid, embedding) VALUES ({}, vec_f32('{}'))",
            rowid, vector_json
        );

        match db.execute(&sql, []) {
            Ok(_) => {
                stats.inserts_completed.fetch_add(1, Ordering::Relaxed);
            }
            Err(e) => {
                if !e.to_string().contains("BUSY") {
                    eprintln!("Insert error: {}", e);
                }
                stats.insert_errors.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Search worker for threaded test
fn search_worker_quantized(
    db_path: PathBuf,
    thread_id: usize,
    config: Arc<ThreadedStressConfig>,
    stats: Arc<ThreadedStats>,
    stop_flag: Arc<AtomicBool>,
) {
    let db = match open_threaded_connection(&db_path) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("Search thread {} failed to open: {}", thread_id, e);
            return;
        }
    };

    for i in 0..config.searches_per_thread {
        if stop_flag.load(Ordering::Relaxed) {
            break;
        }

        let query_vector = generate_vector(i * 7 + thread_id * 1000, config.dimensions);
        let vector_json = vector_to_json(&query_vector);

        let sql = format!(
            "SELECT rowid, distance FROM vectors WHERE embedding MATCH vec_f32('{}') AND k = {} ORDER BY distance",
            vector_json, config.k
        );

        match db.prepare(&sql) {
            Ok(mut stmt) => {
                match stmt.query_map([], |row| {
                    let rowid: i64 = row.get(0)?;
                    let distance: f64 = row.get(1)?;
                    Ok((rowid, distance))
                }) {
                    Ok(rows) => {
                        let results: Vec<_> = rows.filter_map(|r| r.ok()).collect();
                        if !results.is_empty() {
                            stats.searches_completed.fetch_add(1, Ordering::Relaxed);
                        }
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => {
                        stats.search_errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            Err(_) => {
                stats.search_errors.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Run multi-threaded stress test with specified quantization
fn run_threaded_benchmark(use_int8_quantization: bool) -> (f64, u64, u64, u64, u64) {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("stress.db");

    // Setup database
    {
        let db = Connection::open(&db_path).unwrap();
        db.execute_batch(
            "PRAGMA page_size=16384;
             PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA cache_size=5000;
             PRAGMA temp_store=MEMORY;",
        )
        .unwrap();
        sqlite_vec_hnsw::init(&db).unwrap();

        let create_sql = if use_int8_quantization {
            "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[128] index_quantization=int8)"
        } else {
            "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[128])"
        };
        db.execute(create_sql, []).unwrap();
    }

    let config = Arc::new(ThreadedStressConfig {
        num_insert_threads: 4,
        num_search_threads: 2,
        vectors_per_thread: 1000,
        searches_per_thread: 500,
        dimensions: 128,
        k: 10,
    });

    let stats = Arc::new(ThreadedStats::default());
    let stop_flag = Arc::new(AtomicBool::new(false));

    let start = Instant::now();
    let mut handles = Vec::new();

    // Spawn insert threads
    for thread_id in 0..config.num_insert_threads {
        let db_path = db_path.clone();
        let config = Arc::clone(&config);
        let stats = Arc::clone(&stats);
        let stop_flag = Arc::clone(&stop_flag);
        handles.push(thread::spawn(move || {
            insert_worker_quantized(db_path, thread_id, config, stats, stop_flag);
        }));
    }

    // Spawn search threads
    for thread_id in 0..config.num_search_threads {
        let db_path = db_path.clone();
        let config = Arc::clone(&config);
        let stats = Arc::clone(&stats);
        let stop_flag = Arc::clone(&stop_flag);
        handles.push(thread::spawn(move || {
            search_worker_quantized(db_path, thread_id, config, stats, stop_flag);
        }));
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    let elapsed = start.elapsed().as_secs_f64();
    let inserts = stats.inserts_completed.load(Ordering::Relaxed);
    let searches = stats.searches_completed.load(Ordering::Relaxed);
    let insert_errors = stats.insert_errors.load(Ordering::Relaxed);
    let search_errors = stats.search_errors.load(Ordering::Relaxed);

    (elapsed, inserts, searches, insert_errors, search_errors)
}

#[test]
fn test_quantization_threaded_stress() {
    println!("\n=== Multi-threaded Stress Test: float32 vs int8 ===");
    println!("Config: 4 insert threads, 2 search threads, 1000 vec/thread, 500 queries/thread\n");

    println!("Running float32 benchmark...");
    let (f32_time, f32_inserts, f32_searches, f32_ierr, f32_serr) = run_threaded_benchmark(false);

    println!("Running int8 benchmark...");
    let (i8_time, i8_inserts, i8_searches, i8_ierr, i8_serr) = run_threaded_benchmark(true);

    println!("\n{:<25} {:>12} {:>12}", "", "float32", "int8");
    println!("{}", "-".repeat(52));
    println!("{:<25} {:>12.2} {:>12.2}", "Time (s):", f32_time, i8_time);
    println!(
        "{:<25} {:>12} {:>12}",
        "Inserts completed:", f32_inserts, i8_inserts
    );
    println!(
        "{:<25} {:>12} {:>12}",
        "Searches completed:", f32_searches, i8_searches
    );
    println!("{:<25} {:>12} {:>12}", "Insert errors:", f32_ierr, i8_ierr);
    println!("{:<25} {:>12} {:>12}", "Search errors:", f32_serr, i8_serr);
    println!(
        "{:<25} {:>12.0} {:>12.0}",
        "Insert throughput (vec/s):",
        f32_inserts as f64 / f32_time,
        i8_inserts as f64 / i8_time
    );
    println!(
        "{:<25} {:>12.0} {:>12.0}",
        "Search throughput (q/s):",
        f32_searches as f64 / f32_time,
        i8_searches as f64 / i8_time
    );

    // Verify no errors
    assert_eq!(f32_ierr, 0, "float32 should have no insert errors");
    assert_eq!(i8_ierr, 0, "int8 should have no insert errors");

    // Verify inserts completed
    assert_eq!(f32_inserts, 4000, "float32 should complete all inserts");
    assert_eq!(i8_inserts, 4000, "int8 should complete all inserts");

    println!("\n[OK] Threaded stress test passed for both modes!");
}

#[test]
fn test_quantization_large_scale_recall() {
    println!("\n=== Large Scale Recall Test: 10K vectors ===\n");

    const LARGE_VECTORS: i64 = 10000;
    const LARGE_K: i64 = 20;

    for use_int8 in [false, true] {
        let mode = if use_int8 { "int8" } else { "float32" };

        let db = Connection::open_in_memory().unwrap();
        sqlite_vec_hnsw::init(&db).unwrap();

        let create_sql = if use_int8 {
            format!(
                "CREATE VIRTUAL TABLE test USING vec0(embedding float[{}] index_quantization=int8)",
                DIMENSIONS
            )
        } else {
            format!(
                "CREATE VIRTUAL TABLE test USING vec0(embedding float[{}])",
                DIMENSIONS
            )
        };
        db.execute(&create_sql, []).unwrap();

        // Insert vectors
        let insert_start = Instant::now();
        db.execute("BEGIN", []).unwrap();
        for i in 0..LARGE_VECTORS {
            let vec = generate_vector(i as usize, DIMENSIONS);
            let bytes = vector_to_bytes(&vec);
            db.execute(
                "INSERT INTO test(rowid, embedding) VALUES (?, ?)",
                rusqlite::params![i + 1, bytes],
            )
            .unwrap();
        }
        db.execute("COMMIT", []).unwrap();
        let insert_time = insert_start.elapsed();

        // Query with random vector
        let query_vec = generate_vector(5555, DIMENSIONS);
        let query_bytes = vector_to_bytes(&query_vec);

        // Ground truth
        let mut distances: Vec<(i64, f32)> = (0..LARGE_VECTORS)
            .map(|i| {
                let vec = generate_vector(i as usize, DIMENSIONS);
                let dist = l2_distance(&query_vec, &vec);
                (i + 1, dist)
            })
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let ground_truth: Vec<i64> = distances
            .iter()
            .take(LARGE_K as usize)
            .map(|(r, _)| *r)
            .collect();

        // HNSW search
        let search_start = Instant::now();
        let mut stmt = db
            .prepare("SELECT rowid FROM test WHERE embedding MATCH ? AND k = ? ORDER BY distance")
            .unwrap();
        let results: Vec<i64> = stmt
            .query_map(rusqlite::params![query_bytes, LARGE_K], |row| row.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        let search_time = search_start.elapsed();

        let matches = results.iter().filter(|r| ground_truth.contains(r)).count();
        let recall = matches as f64 / LARGE_K as f64;

        // Get storage size
        let hnsw_storage: i64 = db
            .query_row(
                "SELECT SUM(LENGTH(vector)) FROM test_embedding_hnsw_nodes",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        println!("{}:", mode);
        println!(
            "  Insert: {:.2}s ({:.0} vec/s)",
            insert_time.as_secs_f64(),
            LARGE_VECTORS as f64 / insert_time.as_secs_f64()
        );
        println!("  Search: {:.2}ms", search_time.as_secs_f64() * 1000.0);
        println!(
            "  Recall@{}: {:.1}% ({}/{})",
            LARGE_K,
            recall * 100.0,
            matches,
            LARGE_K
        );
        println!(
            "  HNSW storage: {} bytes ({:.2} MB)",
            hnsw_storage,
            hnsw_storage as f64 / 1024.0 / 1024.0
        );
        println!();

        // Assert reasonable recall
        assert!(
            recall >= 0.85,
            "{} recall should be >= 85%, got {:.1}%",
            mode,
            recall * 100.0
        );
    }

    println!("[OK] Large scale recall test passed!");
}
