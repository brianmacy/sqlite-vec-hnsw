/// Create a test database with Rust extension for C compatibility testing
///
/// Run with: cargo run --release --example create_test_db
use rusqlite::Connection;

fn main() {
    let db_path = "/tmp/rust_created.db";

    // Delete old database if it exists
    let _ = std::fs::remove_file(db_path);

    let db = Connection::open(db_path).unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    println!("Creating test database for C compatibility...");

    // Create virtual table
    db.execute(
        "CREATE VIRTUAL TABLE test_vectors USING vec0(embedding float[128])",
        [],
    )
    .unwrap();

    println!("✓ Created virtual table");

    // Insert 50 vectors using prepared statement + transaction
    db.execute("BEGIN TRANSACTION", []).unwrap();

    let mut stmt = db
        .prepare("INSERT INTO test_vectors(rowid, embedding) VALUES (?, ?)")
        .unwrap();

    for i in 1..=50 {
        // Generate vector: [i*1/128, i*2/128, i*3/128, ..., i*128/128]
        let vector: Vec<f32> = (0..128).map(|j| (i * (j + 1)) as f32 / 128.0).collect();
        let bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();

        stmt.execute(rusqlite::params![i, bytes]).unwrap();
    }

    drop(stmt);
    db.execute("COMMIT", []).unwrap();

    println!("✓ Inserted 50 vectors");

    // Verify
    let count: i64 = db
        .query_row("SELECT COUNT(*) FROM test_vectors", [], |row| row.get(0))
        .unwrap();

    println!("✓ Verified: {} rows", count);

    // List shadow tables
    let tables: Vec<String> = db
        .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'test_vectors%' ORDER BY name")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();

    println!("\nShadow tables created by Rust:");
    for table in &tables {
        println!("  {}", table);
    }

    drop(db);

    let size = std::fs::metadata(db_path).unwrap().len();
    println!("\nDatabase size: {} bytes", size);

    println!("\n✅ Rust database created at {}", db_path);
    println!("Ready for C compatibility testing");
}
