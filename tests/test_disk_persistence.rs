// Tests that verify actual disk persistence and realistic performance
use rusqlite::Connection;
use tempfile::TempDir;

#[test]
fn test_disk_persistence_across_connections() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.db");

    // Phase 1: Create database and insert vectors
    {
        let db = Connection::open(&db_path).unwrap();
        sqlite_vec_hnsw::init(&db).unwrap();

        db.execute(
            "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[128])",
            [],
        )
        .unwrap();

        // Insert 100 vectors
        for i in 1..=100 {
            let vector: Vec<f32> = (0..128).map(|j| (i * j) as f32 / 128.0).collect();
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

        println!("✓ Inserted 100 vectors to disk database");
    } // Database closed here

    // Verify database file exists and has size > 0
    let metadata = std::fs::metadata(&db_path).unwrap();
    println!("✓ Database file size: {} bytes", metadata.len());
    assert!(
        metadata.len() > 10_000,
        "Database file too small ({}), shadow tables may not be persisting",
        metadata.len()
    );

    // Phase 2: Reopen database and verify data persists
    {
        let db = Connection::open(&db_path).unwrap();
        sqlite_vec_hnsw::init(&db).unwrap();

        // Count rows
        let count: i64 = db
            .query_row("SELECT COUNT(*) FROM vectors", [], |row| row.get(0))
            .unwrap();

        println!("✓ Reopened database has {} rows", count);
        assert_eq!(count, 100, "Should have 100 rows after reopening");

        // Verify we can read vectors
        let embedding: Vec<u8> = db
            .query_row(
                "SELECT embedding FROM vectors WHERE rowid = 50",
                [],
                |row| row.get(0),
            )
            .unwrap();

        assert_eq!(
            embedding.len(),
            128 * 4,
            "Vector should be 128 float32s = 512 bytes"
        );
        println!("✓ Can read vectors from disk");

        // Verify HNSW index persists
        let node_count: i64 = db
            .query_row(
                "SELECT COUNT(*) FROM vectors_embedding_hnsw_nodes",
                [],
                |row| row.get(0),
            )
            .unwrap();

        println!("✓ HNSW index has {} nodes", node_count);
        assert_eq!(node_count, 100, "HNSW index should have 100 nodes");

        // Verify edges persist
        let edge_count: i64 = db
            .query_row(
                "SELECT COUNT(*) FROM vectors_embedding_hnsw_edges",
                [],
                |row| row.get(0),
            )
            .unwrap();

        println!("✓ HNSW index has {} edges", edge_count);
        assert!(edge_count > 0, "HNSW index should have edges");
    }

    println!("\n✅ Disk persistence verified!");
}

#[test]
fn test_realistic_insert_performance() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("perf_test.db");

    let db = Connection::open(&db_path).unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute(
        "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[768])",
        [],
    )
    .unwrap();

    // Insert 500 vectors and measure time (with disk I/O)
    let start = std::time::Instant::now();
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

    let elapsed = start.elapsed();
    let rate = 500.0 / elapsed.as_secs_f64();

    println!("\n=== Realistic Performance (with disk I/O) ===");
    println!("768D float32: {:.1} vectors/sec", rate);
    println!("Total time: {:.2}s for 500 vectors", elapsed.as_secs_f64());

    // Verify database size
    let metadata = std::fs::metadata(&db_path).unwrap();
    println!("Database size: {} MB", metadata.len() / 1_000_000);

    // Verify data persists
    let count: i64 = db
        .query_row("SELECT COUNT(*) FROM vectors", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 500);

    println!("\n✅ Realistic performance measured with actual disk I/O");
}

#[test]
fn test_wal_mode_performance() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("wal_test.db");

    let db = Connection::open(&db_path).unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    // Enable WAL mode
    let journal_mode: String = db
        .query_row("PRAGMA journal_mode=WAL", [], |row| row.get(0))
        .unwrap();
    println!("Journal mode: {}", journal_mode);

    db.execute(
        "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[768])",
        [],
    )
    .unwrap();

    // Insert with WAL mode
    let start = std::time::Instant::now();
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

    let elapsed = start.elapsed();
    let rate = 500.0 / elapsed.as_secs_f64();

    println!("\n=== WAL Mode Performance ===");
    println!("768D float32: {:.1} vectors/sec", rate);
    println!("Total time: {:.2}s for 500 vectors", elapsed.as_secs_f64());

    // Verify WAL file exists
    let wal_path = temp_dir.path().join("wal_test.db-wal");
    if wal_path.exists() {
        let wal_size = std::fs::metadata(&wal_path).unwrap().len();
        println!("WAL file size: {} KB", wal_size / 1024);
    }

    println!("\n✅ WAL mode performance measured");
}
