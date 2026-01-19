use rusqlite::Connection;

#[test]
fn test_knn_query_simple() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    // Create table
    db.execute(
        "CREATE VIRTUAL TABLE test USING vec0(embedding float[3])",
        [],
    )
    .unwrap();

    // Insert test vectors
    db.execute(
        "INSERT INTO test(rowid, embedding) VALUES (1, vec_f32('[1.0, 0.0, 0.0]'))",
        [],
    )
    .unwrap();
    db.execute(
        "INSERT INTO test(rowid, embedding) VALUES (2, vec_f32('[0.0, 1.0, 0.0]'))",
        [],
    )
    .unwrap();
    db.execute(
        "INSERT INTO test(rowid, embedding) VALUES (3, vec_f32('[0.0, 0.0, 1.0]'))",
        [],
    )
    .unwrap();

    // Try to prepare KNN query
    println!("\nAttempting to prepare KNN query...");
    match db.prepare("SELECT rowid, distance FROM test WHERE embedding MATCH vec_f32('[1.0, 0.0, 0.0]') AND k = 2 ORDER BY distance") {
        Ok(mut stmt) => {
            println!("✓ Query prepared successfully!");

            // Try to execute
            println!("Attempting to execute query...");
            match stmt.query([]) {
                Ok(mut rows) => {
                    println!("✓ Query executed successfully!");

                    let mut results = Vec::new();
                    while let Some(row) = rows.next().unwrap() {
                        let rowid: i64 = row.get(0).unwrap();
                        let distance: f64 = row.get(1).unwrap();
                        results.push((rowid, distance));
                        println!("  Result: rowid={}, distance={}", rowid, distance);
                    }

                    assert_eq!(results.len(), 2, "Should return k=2 results");
                    assert_eq!(results[0].0, 1, "Closest should be rowid 1");
                },
                Err(e) => {
                    println!("✗ Query execution failed: {:?}", e);
                    panic!("Query execution should work");
                }
            }
        },
        Err(e) => {
            println!("✗ Query preparation failed: {:?}", e);
            panic!("Query preparation should work");
        }
    }
}
