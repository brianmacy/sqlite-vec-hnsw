// Debug HNSW graph connectivity
use rusqlite::Connection;

fn main() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute(
        "CREATE VIRTUAL TABLE test USING vec0(embedding float[3])",
        [],
    )
    .unwrap();

    // Insert 5 vectors
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
    db.execute(
        "INSERT INTO test(rowid, embedding) VALUES (4, vec_f32('[1.0, 1.0, 0.0]'))",
        [],
    )
    .unwrap();
    db.execute(
        "INSERT INTO test(rowid, embedding) VALUES (5, vec_f32('[0.5, 0.5, 0.5]'))",
        [],
    )
    .unwrap();

    // Check metadata (single-row schema)
    let entry_point: i64 = db
        .query_row(
            "SELECT entry_point_rowid FROM test_embedding_hnsw_meta WHERE id = 1",
            [],
            |row| row.get(0),
        )
        .unwrap_or(-1);
    println!("Entry point rowid: {}", entry_point);

    // Check edges
    let mut stmt = db.prepare("SELECT from_rowid, to_rowid, level FROM test_embedding_hnsw_edges ORDER BY from_rowid, to_rowid").unwrap();
    let edges: Vec<(i64, i64, i64)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    println!("Total edges: {}", edges.len());
    println!("Edges (from → to @ level):");
    for (from, to, level) in &edges {
        println!("  {} → {} @ L{}", from, to, level);
    }

    // Check which nodes each node is connected to
    for node in 1..=5 {
        let neighbors: Vec<i64> = db.prepare(
            &format!("SELECT to_rowid FROM test_embedding_hnsw_edges WHERE from_rowid = {} AND level = 0", node)
        )
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        println!("Node {} neighbors at L0: {:?}", node, neighbors);
    }
}
