// Debug collinear vector insertion
use rusqlite::Connection;

fn main() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute(
        "CREATE VIRTUAL TABLE test USING vec0(embedding float[4])",
        [],
    )
    .unwrap();

    // Insert collinear vectors: [1,0,0,0], [2,0,0,0], ..., [10,0,0,0]
    for i in 1..=10i64 {
        let vec: Vec<f32> = vec![i as f32, 0.0, 0.0, 0.0];
        let bytes: Vec<u8> = vec.iter().flat_map(|f| f.to_le_bytes()).collect();
        db.execute(
            "INSERT INTO test(rowid, embedding) VALUES (?, ?)",
            rusqlite::params![i, bytes],
        )
        .unwrap();

        // Check edge count after each insert
        let edge_count: i64 = db
            .query_row(
                "SELECT COUNT(*) FROM test_embedding_hnsw_edges",
                [],
                |row| row.get(0),
            )
            .unwrap();
        println!("After insert {}: {} edges", i, edge_count);
    }

    // Show all edges
    println!("\n=== All Edges ===");
    let mut stmt = db.prepare("SELECT from_rowid, to_rowid, level FROM test_embedding_hnsw_edges ORDER BY from_rowid, to_rowid").unwrap();
    let edges: Vec<(i64, i64, i64)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    for (from, to, level) in &edges {
        println!("  {} -> {} @ L{}", from, to, level);
    }

    println!("\nTotal edges: {}", edges.len());
}
