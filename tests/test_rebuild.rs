use rusqlite::Connection;

#[test]
fn test_vec_rebuild_hnsw_basic() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    // Create table with HNSW
    db.execute(
        "CREATE VIRTUAL TABLE docs USING vec0(embedding float[3])",
        [],
    )
    .unwrap();

    // Insert some vectors
    db.execute(
        "INSERT INTO docs(rowid, embedding) VALUES (1, vec_f32('[1.0, 0.0, 0.0]'))",
        [],
    )
    .unwrap();
    db.execute(
        "INSERT INTO docs(rowid, embedding) VALUES (2, vec_f32('[0.0, 1.0, 0.0]'))",
        [],
    )
    .unwrap();
    db.execute(
        "INSERT INTO docs(rowid, embedding) VALUES (3, vec_f32('[0.0, 0.0, 1.0]'))",
        [],
    )
    .unwrap();

    // Rebuild the HNSW index
    let result: String = db
        .query_row("SELECT vec_rebuild_hnsw('docs', 'embedding')", [], |row| {
            row.get(0)
        })
        .unwrap();

    assert!(result.contains("Rebuilt HNSW index"));
    assert!(result.contains("3 vectors"));
}

#[test]
fn test_vec_rebuild_hnsw_with_params() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    // Create table
    db.execute(
        "CREATE VIRTUAL TABLE docs USING vec0(embedding float[3])",
        [],
    )
    .unwrap();

    // Insert vectors
    for i in 1..=5 {
        db.execute(
            &format!(
                "INSERT INTO docs(rowid, embedding) VALUES ({}, vec_f32('[{}, {}, {}]'))",
                i,
                i as f32,
                (i * 2) as f32,
                (i * 3) as f32
            ),
            [],
        )
        .unwrap();
    }

    // Rebuild with custom parameters
    let result: String = db
        .query_row(
            "SELECT vec_rebuild_hnsw('docs', 'embedding', 16, 200)",
            [],
            |row| row.get(0),
        )
        .unwrap();

    assert!(result.contains("Rebuilt HNSW index"));
    assert!(result.contains("5 vectors"));

    // Verify the new parameters were set in metadata
    let m_value: String = db
        .query_row(
            "SELECT value FROM docs_embedding_hnsw_meta WHERE key='M'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(m_value, "16");

    let ef_value: String = db
        .query_row(
            "SELECT value FROM docs_embedding_hnsw_meta WHERE key='ef_construction'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(ef_value, "200");
}

#[test]
fn test_vec_rebuild_hnsw_invalid_params() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute(
        "CREATE VIRTUAL TABLE docs USING vec0(embedding float[3])",
        [],
    )
    .unwrap();

    // Test invalid M value (too low)
    let result = db.query_row(
        "SELECT vec_rebuild_hnsw('docs', 'embedding', 1, 100)",
        [],
        |row| row.get::<_, String>(0),
    );
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("M must be between 2 and 100")
    );

    // Test invalid ef_construction value (too high)
    let result = db.query_row(
        "SELECT vec_rebuild_hnsw('docs', 'embedding', 16, 3000)",
        [],
        |row| row.get::<_, String>(0),
    );
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("ef_construction must be between 10 and 2000")
    );
}

#[test]
fn test_vec_rebuild_hnsw_wrong_arg_count() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute(
        "CREATE VIRTUAL TABLE docs USING vec0(embedding float[3])",
        [],
    )
    .unwrap();

    // Test with wrong number of arguments (1 arg)
    let result = db.query_row("SELECT vec_rebuild_hnsw('docs')", [], |row| {
        row.get::<_, String>(0)
    });
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("requires 2 or 4 arguments")
    );

    // Test with wrong number of arguments (3 args)
    let result = db.query_row(
        "SELECT vec_rebuild_hnsw('docs', 'embedding', 16)",
        [],
        |row| row.get::<_, String>(0),
    );
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("requires 2 or 4 arguments")
    );
}
