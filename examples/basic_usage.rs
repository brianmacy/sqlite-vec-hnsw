//! Basic usage example for sqlite-vec-hnsw
//!
//! Demonstrates:
//! - Creating a virtual table with vector columns
//! - Inserting vectors
//! - Querying vectors
//! - DELETE and UPDATE operations
//! - HNSW index automatic building

use rusqlite::Connection;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Open an in-memory database
    let db = Connection::open_in_memory()?;

    // Initialize the extension
    sqlite_vec_hnsw::init(&db)?;

    println!("✓ Initialized sqlite-vec-hnsw extension");

    // Create a virtual table with a 3-dimensional float32 vector column
    db.execute(
        "CREATE VIRTUAL TABLE documents USING vec0(embedding float[3])",
        [],
    )?;

    println!("✓ Created virtual table 'documents' with 3D vector column");

    // Insert some vectors
    let vectors = vec![
        (1, "[1.0, 2.0, 3.0]"),
        (2, "[4.0, 5.0, 6.0]"),
        (3, "[7.0, 8.0, 9.0]"),
        (4, "[10.0, 11.0, 12.0]"),
        (5, "[13.0, 14.0, 15.0]"),
    ];

    for (rowid, vector) in &vectors {
        db.execute(
            &format!(
                "INSERT INTO documents(rowid, embedding) VALUES ({}, vec_f32('{}'))",
                rowid, vector
            ),
            [],
        )?;
    }

    println!(
        "✓ Inserted {} vectors with automatic HNSW indexing",
        vectors.len()
    );

    // Query the count
    let count: i64 = db.query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))?;
    println!("✓ Table has {} vectors", count);

    // Read vectors back and verify
    let mut stmt = db.prepare("SELECT rowid, embedding FROM documents ORDER BY rowid")?;
    let rows = stmt.query_map([], |row| {
        let rowid: i64 = row.get(0)?;
        let vector: Vec<u8> = row.get(1)?;
        Ok((rowid, vector))
    })?;

    println!("\n📊 Stored vectors:");
    for row in rows {
        let (rowid, vector) = row?;
        // Decode float32 bytes
        let floats: Vec<f32> = vector
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        println!("  Row {}: {:?}", rowid, floats);
    }

    // Use SQL functions
    let distance: f64 = db.query_row(
        "SELECT vec_distance_l2('[1.0, 2.0, 3.0]', '[4.0, 5.0, 6.0]')",
        [],
        |row| row.get(0),
    )?;
    println!("\n📏 Distance between vectors: {:.3}", distance);

    // Get vector length
    let length: i64 = db.query_row("SELECT vec_length(vec_f32('[1.0, 2.0, 3.0]'))", [], |row| {
        row.get(0)
    })?;
    println!("📐 Vector dimensions: {}", length);

    // Get version
    let version: String = db.query_row("SELECT vec_version()", [], |row| row.get(0))?;
    println!("ℹ️  Library version: {}", version);

    // Update a vector
    db.execute(
        "UPDATE documents SET embedding = vec_f32('[99.0, 98.0, 97.0]') WHERE rowid = 3",
        [],
    )?;
    println!("\n✏️  Updated vector at rowid 3");

    // Verify the update
    let updated: Vec<u8> = db.query_row(
        "SELECT embedding FROM documents WHERE rowid = 3",
        [],
        |row| row.get(0),
    )?;
    let updated_floats: Vec<f32> = updated
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    println!("  New value: {:?}", updated_floats);

    // Delete a vector
    db.execute("DELETE FROM documents WHERE rowid = 5", [])?;
    println!("\n🗑️  Deleted vector at rowid 5");

    // Verify deletion
    let final_count: i64 = db.query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))?;
    println!("  Remaining vectors: {}", final_count);

    // Query shadow tables to see persistence
    let shadow_tables: Vec<String> = db
        .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'documents_%'")?
        .query_map([], |row| row.get(0))?
        .collect::<Result<_, _>>()?;

    println!("\n📂 Shadow tables created:");
    for table in &shadow_tables {
        println!("  - {}", table);
    }

    // Check HNSW metadata
    let hnsw_nodes: i64 = db.query_row(
        "SELECT COUNT(*) FROM documents_embedding_hnsw_nodes",
        [],
        |row| row.get(0),
    )?;
    let hnsw_edges: i64 = db.query_row(
        "SELECT COUNT(*) FROM documents_embedding_hnsw_edges",
        [],
        |row| row.get(0),
    )?;

    println!("\n🔗 HNSW index statistics:");
    println!("  Nodes: {}", hnsw_nodes);
    println!("  Edges: {}", hnsw_edges);

    println!("\n✅ Example completed successfully!");

    Ok(())
}
