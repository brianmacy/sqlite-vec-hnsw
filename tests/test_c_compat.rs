use rusqlite::Connection;

#[test]
#[ignore] // Run with: cargo test test_c_compat -- --ignored --nocapture
fn test_read_c_created_database() {
    let db_path = "/Users/brianmacy/Downloads/opensanctions_export2_embedding.db";

    // Check if the C-created database exists
    if !std::path::Path::new(db_path).exists() {
        println!("⚠️  C-created database not found at: {}", db_path);
        println!("   Skipping C compatibility test");
        return;
    }

    println!("\n🔍 Testing C compatibility with database: {}", db_path);

    // Open the C-created database
    let db = Connection::open(db_path).unwrap();

    // Initialize our extension (registers vec0 module)
    sqlite_vec_hnsw::init(&db).unwrap();

    println!("✓ Database opened successfully");

    // Check the table exists (it's called SEMANTIC_VALUE in this database)
    let count: i64 = db
        .query_row("SELECT COUNT(*) FROM SEMANTIC_VALUE_rowids", [], |row| {
            row.get(0)
        })
        .unwrap();

    println!("📊 Total vectors in C database: {}", count);
    assert!(count > 0, "Should have vectors");

    // Check HNSW index was built by C version
    let node_count: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM SEMANTIC_VALUE_EMBEDDING_hnsw_nodes",
            [],
            |row| row.get(0),
        )
        .unwrap();

    println!("📊 HNSW nodes: {}", node_count);
    assert_eq!(node_count, count, "All vectors should be indexed");

    // Check HNSW metadata
    let m_value: String = db
        .query_row(
            "SELECT value FROM SEMANTIC_VALUE_EMBEDDING_hnsw_meta WHERE key='M'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    println!("   M = {}", m_value);

    let ef_value: String = db
        .query_row(
            "SELECT value FROM SEMANTIC_VALUE_EMBEDDING_hnsw_meta WHERE key='ef_construction'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    println!("   ef_construction = {}", ef_value);

    let entry_point: String = db
        .query_row(
            "SELECT value FROM SEMANTIC_VALUE_EMBEDDING_hnsw_meta WHERE key='entry_point_rowid'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    println!("   entry_point_rowid = {}", entry_point);

    // Try to read some vectors directly from shadow tables
    println!("\n🔍 Reading vectors from shadow tables...");

    let mut stmt = db
        .prepare("SELECT rowid, chunk_id, chunk_offset FROM SEMANTIC_VALUE_rowids LIMIT 5")
        .unwrap();

    let rowid_mappings: Vec<(i64, i64, i64)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    println!("   First 5 rowid mappings:");
    for (rowid, chunk_id, chunk_offset) in rowid_mappings {
        println!(
            "     rowid={}, chunk_id={}, chunk_offset={}",
            rowid, chunk_id, chunk_offset
        );
    }

    // Check HNSW edges
    let edge_count: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM SEMANTIC_VALUE_EMBEDDING_hnsw_edges",
            [],
            |row| row.get(0),
        )
        .unwrap();
    println!("\n📊 HNSW edges: {}", edge_count);
    assert!(edge_count > 0, "Should have HNSW edges");

    // Sample some edges
    let mut stmt = db
        .prepare("SELECT from_rowid, to_rowid, level, distance FROM SEMANTIC_VALUE_EMBEDDING_hnsw_edges LIMIT 5")
        .unwrap();

    let edges: Vec<(i64, i64, i64, Option<f64>)> = stmt
        .query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    println!("   First 5 HNSW edges:");
    for (from_rowid, to_rowid, level, distance) in edges {
        if let Some(dist) = distance {
            println!(
                "     {} -> {} (level={}, dist={:.3})",
                from_rowid, to_rowid, level, dist
            );
        } else {
            println!(
                "     {} -> {} (level={}, dist=NULL)",
                from_rowid, to_rowid, level
            );
        }
    }

    println!("\n✅ C compatibility test passed!");
    println!("   - Successfully read C-created database");
    println!("   - Shadow tables schema is compatible");
    println!("   - HNSW index metadata is readable");
    println!("   - Can query rowid mappings and HNSW graph");
}

#[test]
fn test_write_for_c_compatibility() {
    // Create a database that the C version should be able to read
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    println!("\n🔧 Creating database compatible with C version...");

    // Create a table with similar schema to C version
    db.execute(
        "CREATE VIRTUAL TABLE test_vectors USING vec0(embedding float[128])",
        [],
    )
    .unwrap();

    // Insert some test vectors
    for i in 1..=100 {
        let vector: Vec<f32> = (0..128).map(|j| (i * 128 + j) as f32 / 1000.0).collect();
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
                "INSERT INTO test_vectors(rowid, embedding) VALUES ({}, vec_f32('{}'))",
                i, vector_json
            ),
            [],
        )
        .unwrap();
    }

    println!("✓ Inserted 100 vectors");

    // Verify shadow tables match expected schema
    let tables: Vec<String> = db
        .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'test_vectors%' ORDER BY name")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    println!("\n📋 Shadow tables created:");
    for table in &tables {
        println!("   - {}", table);
    }

    // Verify expected shadow tables exist
    assert!(
        tables.contains(&"test_vectors_chunks".to_string()),
        "Missing chunks table"
    );
    assert!(
        tables.contains(&"test_vectors_rowids".to_string()),
        "Missing rowids table"
    );
    assert!(
        tables
            .iter()
            .any(|t| t.contains("_chunks") && t.contains("chunks0")),
        "Missing vector chunks table"
    );
    assert!(
        tables.iter().any(|t| t.contains("hnsw_meta")),
        "Missing HNSW meta table"
    );
    assert!(
        tables.iter().any(|t| t.contains("hnsw_nodes")),
        "Missing HNSW nodes table"
    );
    assert!(
        tables.iter().any(|t| t.contains("hnsw_edges")),
        "Missing HNSW edges table"
    );
    assert!(
        tables.iter().any(|t| t.contains("hnsw_levels")),
        "Missing HNSW levels table"
    );

    // Check shadow table schemas match C expectations
    println!("\n🔍 Verifying shadow table schemas...");

    // Check chunks table schema
    let chunks_schema: String = db
        .query_row(
            "SELECT sql FROM sqlite_master WHERE name='test_vectors_chunks'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    println!("   chunks: {}", chunks_schema);
    assert!(chunks_schema.contains("chunk_id"));
    assert!(chunks_schema.contains("size"));
    assert!(chunks_schema.contains("validity"));

    // Check rowids table schema
    let rowids_schema: String = db
        .query_row(
            "SELECT sql FROM sqlite_master WHERE name='test_vectors_rowids'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    println!("   rowids: {}", rowids_schema);
    assert!(rowids_schema.contains("rowid"));
    assert!(rowids_schema.contains("chunk_id"));
    assert!(rowids_schema.contains("chunk_offset"));

    // Check HNSW metadata (find the meta table dynamically)
    let meta_table = tables
        .iter()
        .find(|t| t.contains("hnsw_meta"))
        .expect("Should have HNSW meta table");

    let meta_count: i64 = db
        .query_row(
            &format!("SELECT COUNT(*) FROM \"{}\"", meta_table),
            [],
            |row| row.get(0),
        )
        .unwrap();
    println!("\n📊 HNSW metadata rows: {}", meta_count);
    assert_eq!(meta_count, 1, "Should have exactly one metadata row");

    // Verify metadata values (single-row schema)
    let (m, max_m0, ef_construction, ef_search, entry_point_rowid, num_nodes): (
        i32,
        i32,
        i32,
        i32,
        i64,
        i32,
    ) = db
        .query_row(
            &format!(
                "SELECT m, max_m0, ef_construction, ef_search, entry_point_rowid, num_nodes \
                 FROM \"{}\" WHERE id = 1",
                meta_table
            ),
            [],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                ))
            },
        )
        .unwrap();

    println!("   Metadata:");
    println!("     m = {}", m);
    println!("     max_m0 = {}", max_m0);
    println!("     ef_construction = {}", ef_construction);
    println!("     ef_search = {}", ef_search);
    println!("     entry_point_rowid = {}", entry_point_rowid);
    println!("     num_nodes = {}", num_nodes);

    assert_eq!(m, 32, "M should be 32");
    assert_eq!(max_m0, 64, "max_M0 should be 64");
    assert!(num_nodes > 0, "Should have nodes indexed");

    println!("\n✅ Schema compatibility test passed!");
    println!("   - All expected shadow tables created");
    println!("   - Shadow table schemas match expected");
    println!("   - HNSW metadata is populated");
}
