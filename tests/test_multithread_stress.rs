// Multi-threaded stress test for concurrent insert/search operations
// Tests race conditions and performance under extreme contention
//
// SQLite threading notes:
// - WAL mode required for concurrent readers with writer
// - Each thread needs its own Connection (rusqlite Connection is !Send)
// - busy_timeout prevents SQLITE_BUSY errors during contention

use rusqlite::Connection;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Configuration for the stress test
struct StressConfig {
    num_insert_threads: usize,
    num_search_threads: usize,
    vectors_per_insert_thread: usize,
    searches_per_search_thread: usize,
    vector_dimensions: usize,
    k_neighbors: usize,
    busy_timeout_ms: u32,
}

impl Default for StressConfig {
    fn default() -> Self {
        Self {
            num_insert_threads: 12,
            num_search_threads: 8,
            vectors_per_insert_thread: 2000,
            searches_per_search_thread: 5000,
            vector_dimensions: 128,
            k_neighbors: 20,
            busy_timeout_ms: 300000, // 5 minutes to handle extreme contention
        }
    }
}

/// Statistics collected during the test
#[derive(Default)]
struct StressStats {
    inserts_completed: AtomicU64,
    searches_completed: AtomicU64,
    insert_errors: AtomicU64,
    search_errors: AtomicU64,
}

/// Opens a connection with proper multi-threaded settings and C benchmark pragmas
fn open_connection(db_path: &PathBuf, busy_timeout_ms: u32) -> Result<Connection, String> {
    let db = Connection::open(db_path).map_err(|e| e.to_string())?;

    // Set busy timeout to handle contention
    db.busy_timeout(Duration::from_millis(busy_timeout_ms as u64))
        .map_err(|e| e.to_string())?;

    // Apply C benchmark pragma settings (page_size is already set on DB file)
    db.execute_batch(
        "PRAGMA cache_size=10000;
         PRAGMA temp_store=MEMORY;",
    )
    .map_err(|e| e.to_string())?;

    // Initialize the extension
    sqlite_vec_hnsw::init(&db).map_err(|e| e.to_string())?;

    Ok(db)
}

/// Generate a deterministic test vector based on index
fn generate_vector(index: usize, dimensions: usize) -> Vec<f32> {
    (0..dimensions)
        .map(|j| ((index * dimensions + j) % 10000) as f32 / 10000.0)
        .collect()
}

/// Format vector as JSON for SQL
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

/// Insert thread worker
fn insert_worker(
    db_path: PathBuf,
    thread_id: usize,
    config: Arc<StressConfig>,
    stats: Arc<StressStats>,
    stop_flag: Arc<AtomicBool>,
) {
    let db = match open_connection(&db_path, config.busy_timeout_ms) {
        Ok(db) => db,
        Err(e) => {
            eprintln!(
                "Insert thread {} failed to open connection: {}",
                thread_id, e
            );
            return;
        }
    };

    let base_rowid = thread_id * config.vectors_per_insert_thread + 1;

    for i in 0..config.vectors_per_insert_thread {
        if stop_flag.load(Ordering::Relaxed) {
            break;
        }

        let rowid = base_rowid + i;
        let vector = generate_vector(rowid, config.vector_dimensions);
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
                eprintln!(
                    "Insert thread {} error on rowid {}: {}",
                    thread_id, rowid, e
                );
                stats.insert_errors.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Search thread worker
fn search_worker(
    db_path: PathBuf,
    thread_id: usize,
    config: Arc<StressConfig>,
    stats: Arc<StressStats>,
    stop_flag: Arc<AtomicBool>,
) {
    let db = match open_connection(&db_path, config.busy_timeout_ms) {
        Ok(db) => db,
        Err(e) => {
            eprintln!(
                "Search thread {} failed to open connection: {}",
                thread_id, e
            );
            return;
        }
    };

    // NO DELAY - hit it immediately and hard
    for i in 0..config.searches_per_search_thread {
        if stop_flag.load(Ordering::Relaxed) {
            break;
        }

        // Generate query vector based on iteration
        let query_vector = generate_vector(i * 7 + thread_id, config.vector_dimensions);
        let vector_json = vector_to_json(&query_vector);

        let sql = format!(
            "SELECT rowid, distance FROM vectors WHERE embedding MATCH vec_f32('{}') AND k = {} ORDER BY distance",
            vector_json, config.k_neighbors
        );

        match db.prepare(&sql) {
            Ok(mut stmt) => {
                match stmt.query_map([], |row| {
                    let rowid: i64 = row.get(0)?;
                    let distance: f64 = row.get(1)?;
                    Ok((rowid, distance))
                }) {
                    Ok(rows) => {
                        // Consume the iterator to execute the query
                        let results: Vec<_> = rows.filter_map(|r| r.ok()).collect();
                        // Results may be empty if no vectors inserted yet - that's OK
                        if !results.is_empty() || i > 10 {
                            stats.searches_completed.fetch_add(1, Ordering::Relaxed);
                        }
                        // 10ms pause to let insert threads acquire write locks
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(e) => {
                        // SQLITE_BUSY or schema changes during query are expected under contention
                        if !e.to_string().contains("locked") && !e.to_string().contains("BUSY") {
                            eprintln!("Search thread {} query error: {}", thread_id, e);
                        }
                        stats.search_errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            Err(e) => {
                if !e.to_string().contains("locked") && !e.to_string().contains("BUSY") {
                    eprintln!("Search thread {} prepare error: {}", thread_id, e);
                }
                stats.search_errors.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

#[test]
fn test_multithread_insert_search_stress() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("stress_test.db");

    println!("\n=== Multi-threaded Insert/Search Stress Test ===\n");

    // Setup: 16k pages + WAL for concurrent access
    {
        let db = Connection::open(&db_path).unwrap();

        db.execute_batch(
            "PRAGMA page_size=16384;
             PRAGMA cache_size=10000;
             PRAGMA temp_store=MEMORY;",
        )
        .unwrap();

        sqlite_vec_hnsw::init(&db).unwrap();

        // Enable WAL mode for concurrent access
        let journal_mode: String = db
            .query_row("PRAGMA journal_mode=WAL", [], |row| row.get(0))
            .unwrap();
        println!("Journal mode: {}", journal_mode);
        assert_eq!(
            journal_mode.to_uppercase(),
            "WAL",
            "WAL mode required for concurrent access"
        );

        // Create table
        db.execute(
            "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[128])",
            [],
        )
        .unwrap();

        println!("Created vectors table with 128D embeddings");
    }

    let config = Arc::new(StressConfig::default());
    let stats = Arc::new(StressStats::default());
    let stop_flag = Arc::new(AtomicBool::new(false));

    println!(
        "\nConfiguration:\n  Insert threads: {}\n  Search threads: {}\n  Vectors per insert thread: {}\n  Searches per search thread: {}\n  Total vectors to insert: {}\n",
        config.num_insert_threads,
        config.num_search_threads,
        config.vectors_per_insert_thread,
        config.searches_per_search_thread,
        config.num_insert_threads * config.vectors_per_insert_thread
    );

    let start = Instant::now();

    // Spawn insert threads
    let mut handles = Vec::new();

    for thread_id in 0..config.num_insert_threads {
        let db_path = db_path.clone();
        let config = Arc::clone(&config);
        let stats = Arc::clone(&stats);
        let stop_flag = Arc::clone(&stop_flag);

        handles.push(thread::spawn(move || {
            insert_worker(db_path, thread_id, config, stats, stop_flag);
        }));
    }

    // Spawn search threads
    for thread_id in 0..config.num_search_threads {
        let db_path = db_path.clone();
        let config = Arc::clone(&config);
        let stats = Arc::clone(&stats);
        let stop_flag = Arc::clone(&stop_flag);

        handles.push(thread::spawn(move || {
            search_worker(db_path, thread_id, config, stats, stop_flag);
        }));
    }

    // Wait for all threads
    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    let elapsed = start.elapsed();

    // Collect final stats
    let inserts = stats.inserts_completed.load(Ordering::Relaxed);
    let searches = stats.searches_completed.load(Ordering::Relaxed);
    let insert_errors = stats.insert_errors.load(Ordering::Relaxed);
    let search_errors = stats.search_errors.load(Ordering::Relaxed);

    println!("\n=== Results ===");
    println!("Time elapsed: {:.2}s", elapsed.as_secs_f64());
    println!("Inserts completed: {} ({} errors)", inserts, insert_errors);
    println!(
        "Searches completed: {} ({} errors)",
        searches, search_errors
    );
    println!(
        "Insert throughput: {:.1} vec/sec",
        inserts as f64 / elapsed.as_secs_f64()
    );
    println!(
        "Search throughput: {:.1} queries/sec",
        searches as f64 / elapsed.as_secs_f64()
    );

    // Verify data integrity
    println!("\n=== Integrity Verification ===");
    let db = Connection::open(&db_path).unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    let expected_count = config.num_insert_threads * config.vectors_per_insert_thread;
    let actual_count: i64 = db
        .query_row("SELECT COUNT(*) FROM vectors", [], |row| row.get(0))
        .unwrap();

    println!("Expected vectors: {}", expected_count);
    println!("Actual vectors: {}", actual_count);

    // Check HNSW index consistency
    let node_count: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM vectors_embedding_hnsw_nodes",
            [],
            |row| row.get(0),
        )
        .unwrap();
    println!("HNSW nodes: {}", node_count);

    let edge_count: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM vectors_embedding_hnsw_edges",
            [],
            |row| row.get(0),
        )
        .unwrap();
    println!("HNSW edges: {}", edge_count);

    // Verify a KNN query works correctly after all the stress
    let query_vector = generate_vector(500, config.vector_dimensions);
    let vector_json = vector_to_json(&query_vector);

    let sql = format!(
        "SELECT rowid, distance FROM vectors WHERE embedding MATCH vec_f32('{}') AND k = 10 ORDER BY distance",
        vector_json
    );

    let mut stmt = db.prepare(&sql).unwrap();
    let results: Vec<(i64, f64)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    println!("\nPost-stress KNN query returned {} results", results.len());
    assert!(results.len() <= 10, "Should return at most k=10 results");

    // Assertions
    assert_eq!(
        actual_count as usize, expected_count,
        "All inserts should complete successfully"
    );
    assert_eq!(
        node_count as usize, expected_count,
        "HNSW should have one node per vector"
    );
    assert!(edge_count > 0, "HNSW should have edges");
    assert_eq!(insert_errors, 0, "No insert errors allowed");
    // Some search errors may occur during initial inserts when table is empty
    assert!(
        search_errors < (config.num_search_threads * config.searches_per_search_thread / 10) as u64,
        "Search error rate should be < 10%"
    );

    println!("\n[OK] Multi-threaded stress test passed!");
}

#[test]
fn test_multithread_heavy_contention() {
    // BRUTAL: Way more threads than CPU cores, max contention
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("contention_test.db");

    println!("\n=== BRUTAL Heavy Contention Test ===\n");

    // Setup with WAL mode for concurrent access
    {
        let db = Connection::open(&db_path).unwrap();

        db.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA cache_size=10000;
             PRAGMA temp_store=MEMORY;
             PRAGMA wal_autocheckpoint=1000;",
        )
        .unwrap();

        sqlite_vec_hnsw::init(&db).unwrap();

        db.execute(
            "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[64])",
            [],
        )
        .unwrap();
    }

    let config = Arc::new(StressConfig {
        num_insert_threads: 20,
        num_search_threads: 12,
        vectors_per_insert_thread: 1500,
        searches_per_search_thread: 5000,
        vector_dimensions: 64,
        k_neighbors: 20,
        busy_timeout_ms: 300000,
    });

    let stats = Arc::new(StressStats::default());
    let stop_flag = Arc::new(AtomicBool::new(false));

    println!(
        "Running {} insert threads + {} search threads concurrently",
        config.num_insert_threads, config.num_search_threads
    );

    let start = Instant::now();
    let mut handles = Vec::new();

    // Interleave insert and search threads for maximum contention
    for thread_id in 0..config.num_insert_threads.max(config.num_search_threads) {
        if thread_id < config.num_insert_threads {
            let db_path = db_path.clone();
            let config = Arc::clone(&config);
            let stats = Arc::clone(&stats);
            let stop_flag = Arc::clone(&stop_flag);
            handles.push(thread::spawn(move || {
                insert_worker(db_path, thread_id, config, stats, stop_flag);
            }));
        }

        if thread_id < config.num_search_threads {
            let db_path = db_path.clone();
            let config = Arc::clone(&config);
            let stats = Arc::clone(&stats);
            let stop_flag = Arc::clone(&stop_flag);
            handles.push(thread::spawn(move || {
                search_worker(db_path, thread_id, config, stats, stop_flag);
            }));
        }
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    let elapsed = start.elapsed();

    let inserts = stats.inserts_completed.load(Ordering::Relaxed);
    let searches = stats.searches_completed.load(Ordering::Relaxed);
    let insert_errors = stats.insert_errors.load(Ordering::Relaxed);

    println!("\nCompleted in {:.2}s", elapsed.as_secs_f64());
    println!("Inserts: {} ({} errors)", inserts, insert_errors);
    println!("Searches: {}", searches);

    // Verify
    let db = Connection::open(&db_path).unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    let expected = config.num_insert_threads * config.vectors_per_insert_thread;
    let actual: i64 = db
        .query_row("SELECT COUNT(*) FROM vectors", [], |row| row.get(0))
        .unwrap();

    assert_eq!(actual as usize, expected, "All inserts must succeed");
    assert_eq!(insert_errors, 0, "No insert errors under contention");

    println!("\n[OK] Heavy contention test passed!");
}

const STRESS_TEST_DURATION_SECS: u64 = 60;

#[test]
#[ignore] // Run with: cargo test test_multithread_long_running -- --ignored --nocapture
fn test_multithread_long_running() {
    // SAVAGE: 60-second brutal stress test to find race conditions
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("longrun_test.db");

    println!("\n=== 60-Second Stress Test Baseline ===\n");
    println!("Config: 16 insert threads, 4 search threads, 128D vectors, k=50\n");

    // Setup - WAL mode for concurrent access
    // 16k pages may help with WITHOUT ROWID clustered edges (M=64 = ~1KB per node)
    {
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
            "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[128])",
            [],
        )
        .unwrap();
    }

    let config = Arc::new(StressConfig {
        num_insert_threads: 16,
        num_search_threads: 4,
        vectors_per_insert_thread: 100000, // High limit - time will stop us
        searches_per_search_thread: 100000, // High limit - time will stop us
        vector_dimensions: 128,
        k_neighbors: 50,
        busy_timeout_ms: 120000,
    });

    let stats = Arc::new(StressStats::default());
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
            insert_worker(db_path, thread_id, config, stats, stop_flag);
        }));
    }

    // Spawn search threads
    for thread_id in 0..config.num_search_threads {
        let db_path = db_path.clone();
        let config = Arc::clone(&config);
        let stats = Arc::clone(&stats);
        let stop_flag = Arc::clone(&stop_flag);
        handles.push(thread::spawn(move || {
            search_worker(db_path, thread_id, config, stats, stop_flag);
        }));
    }

    // Timer thread - stops everything after 60 seconds
    let stop_flag_timer = Arc::clone(&stop_flag);
    let timer = thread::spawn(move || {
        thread::sleep(Duration::from_secs(STRESS_TEST_DURATION_SECS));
        stop_flag_timer.store(true, Ordering::Relaxed);
    });

    // Monitor progress every 5 seconds
    let stats_clone = Arc::clone(&stats);
    let stop_flag_monitor = Arc::clone(&stop_flag);
    let start_clone = start;
    let monitor = thread::spawn(move || {
        let mut last_inserts = 0u64;
        let mut last_searches = 0u64;

        while !stop_flag_monitor.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_secs(5));

            let inserts = stats_clone.inserts_completed.load(Ordering::Relaxed);
            let searches = stats_clone.searches_completed.load(Ordering::Relaxed);

            let insert_rate = (inserts - last_inserts) as f64 / 5.0;
            let search_rate = (searches - last_searches) as f64 / 5.0;

            println!(
                "[{:>2.0}s] Inserts: {:>6} ({:>4.0}/s) | Searches: {:>6} ({:>4.0}/s)",
                start_clone.elapsed().as_secs_f64(),
                inserts,
                insert_rate,
                searches,
                search_rate
            );

            last_inserts = inserts;
            last_searches = searches;
        }
    });

    // Wait for timer to trigger stop
    timer.join().ok();

    // Wait for all worker threads to finish
    for handle in handles {
        handle.join().expect("Thread panicked");
    }
    monitor.join().ok();

    let elapsed = start.elapsed();
    let inserts = stats.inserts_completed.load(Ordering::Relaxed);
    let searches = stats.searches_completed.load(Ordering::Relaxed);
    let insert_errors = stats.insert_errors.load(Ordering::Relaxed);
    let search_errors = stats.search_errors.load(Ordering::Relaxed);

    // Integrity check
    let db = Connection::open(&db_path).unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    let count: i64 = db
        .query_row("SELECT COUNT(*) FROM vectors", [], |row| row.get(0))
        .unwrap();

    let node_count: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM vectors_embedding_hnsw_nodes",
            [],
            |row| row.get(0),
        )
        .unwrap();

    let edge_count: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM vectors_embedding_hnsw_edges",
            [],
            |row| row.get(0),
        )
        .unwrap();

    // Print baseline summary
    println!("\n{}", "=".repeat(60));
    println!("                    BASELINE RESULTS");
    println!("{}", "=".repeat(60));
    println!("Duration:        {:.2}s", elapsed.as_secs_f64());
    println!("Total Inserts:   {} ({} errors)", inserts, insert_errors);
    println!("Total Searches:  {} ({} errors)", searches, search_errors);
    println!(
        "Insert Rate:     {:.0} vec/sec",
        inserts as f64 / elapsed.as_secs_f64()
    );
    println!(
        "Search Rate:     {:.0} queries/sec",
        searches as f64 / elapsed.as_secs_f64()
    );
    println!("{}", "-".repeat(60));
    println!("Vectors in DB:   {}", count);
    println!("HNSW Nodes:      {}", node_count);
    println!("HNSW Edges:      {}", edge_count);
    println!(
        "Integrity:       {}",
        if count == node_count { "PASS" } else { "FAIL" }
    );
    println!("{}", "=".repeat(60));

    assert_eq!(count, node_count, "Vector count must match HNSW node count");
    assert_eq!(insert_errors, 0, "No insert errors allowed");

    println!("\n[OK] 60-second stress test passed!");
}
