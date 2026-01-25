// Verify storage format matches C implementation
use rusqlite::Connection;
use tempfile::TempDir;

#[test]
fn test_storage_breakdown_float32() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("storage_test.db");

    let db = Connection::open(&db_path).unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute(
        "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[768])",
        [],
    )
    .unwrap();

    println!("\n=== Storage Format Analysis (100 vectors, 768D float32) ===");

    // Insert 100 vectors
    db.execute("BEGIN TRANSACTION", []).unwrap();
    for i in 1..=100 {
        let vector: Vec<f32> = (0..768).map(|j| ((i + j) % 100) as f32 / 100.0).collect();
        let bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();
        db.execute(
            "INSERT INTO vectors(rowid, embedding) VALUES (?, ?)",
            rusqlite::params![i, bytes],
        )
        .unwrap();
    }
    db.execute("COMMIT", []).unwrap();

    // Analyze each shadow table
    let tables: Vec<(String, i64)> = db
        .prepare(
            "SELECT name,
             (SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size() LIMIT 1) as size
             FROM sqlite_master
             WHERE type='table' AND name LIKE 'vectors%'
             ORDER BY name",
        )
        .unwrap()
        .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();

    let mut total_size = 0;
    println!("\nShadow Table Sizes:");
    for (table_name, _size) in &tables {
        let row_count: i64 = db
            .query_row(
                &format!("SELECT COUNT(*) FROM \"{}\"", table_name),
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        let table_size: i64 = db
            .query_row(
                &format!(
                    "SELECT SUM(pgsize) FROM dbstat WHERE name = '{}'",
                    table_name
                ),
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        if table_size > 0 {
            total_size += table_size;
            println!(
                "  {}: {} bytes ({} rows, {} bytes/row)",
                table_name,
                table_size,
                row_count,
                if row_count > 0 {
                    table_size / row_count
                } else {
                    0
                }
            );
        }
    }

    let db_file_size = std::fs::metadata(&db_path).unwrap().len();

    println!("\nTotal Analysis:");
    println!(
        "  Raw vector data: {} bytes (768 floats × 4 bytes × 100)",
        768 * 4 * 100
    );
    println!("  Shadow tables: {} bytes", total_size);
    println!("  Database file: {} bytes", db_file_size);
    println!(
        "  Overhead: {:.1}x",
        db_file_size as f64 / (768.0 * 4.0 * 100.0)
    );
    println!("\nC Implementation (24K vectors):");
    println!("  Float32: 70.3 MB = 2,929 bytes/vector");
    println!("  Expected for 100 vectors: {} bytes", 2929 * 100);
    println!("\nRust Implementation (100 vectors):");
    println!(
        "  Actual: {} bytes = {} bytes/vector",
        db_file_size,
        db_file_size / 100
    );
    println!(
        "  Bloat factor: {:.1}x",
        (db_file_size / 100) as f64 / 2929.0
    );

    if db_file_size / 100 > 4000 {
        println!("\n❌ STORAGE FORMAT INCOMPATIBLE WITH C");
        println!("   Rust is storing significantly more data than C");
    }
}

#[test]
fn test_unified_storage_inspection() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("unified_test.db");

    let db = Connection::open(&db_path).unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute(
        "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[768])",
        [],
    )
    .unwrap();

    println!("\n=== Unified Storage Inspection ===");

    // Insert 300 vectors (all in unified _data table)
    db.execute("BEGIN TRANSACTION", []).unwrap();
    for i in 1..=300 {
        let vector: Vec<f32> = (0..768).map(|j| ((i + j) % 100) as f32 / 100.0).collect();
        let bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();
        db.execute(
            "INSERT INTO vectors(rowid, embedding) VALUES (?, ?)",
            rusqlite::params![i, bytes],
        )
        .unwrap();
    }
    db.execute("COMMIT", []).unwrap();

    // Inspect unified _data table
    let data_row_count: i64 = db
        .query_row("SELECT COUNT(*) FROM vectors_data", [], |row| row.get(0))
        .unwrap();

    println!("Unified _data table:");
    println!("  Total rows: {}", data_row_count);
    assert_eq!(data_row_count, 300, "Should have 300 rows in _data table");

    // Check vector sizes
    let avg_vec_size: f64 = db
        .query_row("SELECT AVG(LENGTH(vec00)) FROM vectors_data", [], |row| {
            row.get(0)
        })
        .unwrap();

    println!("  Average vector size: {:.0} bytes", avg_vec_size);
    println!("  Expected: {} bytes (768 floats × 4 bytes)", 768 * 4);

    println!("\nExpected for 300 vectors:");
    println!(
        "  Raw data: {} bytes (768 floats × 4 bytes × 300)",
        768 * 4 * 300
    );
    println!("  With unified storage: minimal overhead");

    let actual_size = std::fs::metadata(&db_path).unwrap().len();
    println!("\nActual database size: {} bytes", actual_size);
    println!("Bloat: {:.1}x", actual_size as f64 / (768.0 * 4.0 * 300.0));
}
