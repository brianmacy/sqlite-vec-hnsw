// Debug HNSW recall issue
use rusqlite::Connection;

#[test]
fn test_simple_hnsw_recall_4d() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    // Create table with small dimensions for easy debugging
    db.execute(
        "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[4])",
        [],
    )
    .unwrap();

    // Insert 10 simple vectors: [1,0,0,0], [2,0,0,0], ..., [10,0,0,0]
    println!("\n=== Inserting vectors ===");
    for i in 1..=10i64 {
        let vec: Vec<f32> = vec![i as f32, 0.0, 0.0, 0.0];
        let bytes: Vec<u8> = vec.iter().flat_map(|f| f.to_le_bytes()).collect();
        db.execute(
            "INSERT INTO vectors(rowid, embedding) VALUES (?, ?)",
            rusqlite::params![i, bytes],
        )
        .unwrap();
        println!("Inserted rowid {} with vec {:?}", i, vec);
    }

    // Check HNSW metadata (single-row schema)
    println!("\n=== HNSW Metadata ===");
    let (m, ef_construction, ef_search, entry_point_rowid, num_nodes): (i32, i32, i32, i64, i32) =
        db.query_row(
            "SELECT m, ef_construction, ef_search, entry_point_rowid, num_nodes \
             FROM vectors_embedding_hnsw_meta WHERE id = 1",
            [],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        )
        .unwrap();
    println!("  m: {}", m);
    println!("  ef_construction: {}", ef_construction);
    println!("  ef_search: {}", ef_search);
    println!("  entry_point_rowid: {}", entry_point_rowid);
    println!("  num_nodes: {}", num_nodes);

    // Check HNSW nodes
    println!("\n=== HNSW Nodes ===");
    let nodes: Vec<(i64, i32)> = db
        .prepare("SELECT rowid, level FROM vectors_embedding_hnsw_nodes")
        .unwrap()
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    println!("Total nodes: {}", nodes.len());
    for (rowid, level) in &nodes {
        println!("  rowid={}, level={}", rowid, level);
    }

    // Check HNSW edges at level 0 (distance column may be NULL)
    println!("\n=== HNSW Edges (level 0) ===");
    let edges: Vec<(i64, i64)> = db
        .prepare("SELECT from_rowid, to_rowid FROM vectors_embedding_hnsw_edges WHERE level = 0 ORDER BY from_rowid LIMIT 30")
        .unwrap()
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    println!("Total edges at level 0: {}", edges.len());
    for (from, to) in &edges {
        println!("  {} -> {}", from, to);
    }

    // Query for vector close to [5, 0, 0, 0]
    let query_vec: Vec<f32> = vec![5.0, 0.0, 0.0, 0.0];
    let query_bytes: Vec<u8> = query_vec.iter().flat_map(|f| f.to_le_bytes()).collect();

    println!("\n=== Query ===");
    println!("Query vector: {:?}", query_vec);

    // Search
    let mut stmt = db
        .prepare(
            "SELECT rowid, distance FROM vectors WHERE embedding MATCH ? AND k = 5 ORDER BY distance",
        )
        .unwrap();

    let results: Vec<(i64, f64)> = stmt
        .query_map(rusqlite::params![query_bytes], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();

    println!("\n=== HNSW Search Results ===");
    for (rowid, dist) in &results {
        println!("  rowid={}, distance={:.4}", rowid, dist);
    }

    // Calculate ground truth
    println!("\n=== Ground Truth (brute force) ===");
    let mut distances: Vec<(i64, f32)> = (1..=10i64)
        .map(|i| {
            let vec = [i as f32, 0.0, 0.0, 0.0];
            let dist: f32 = query_vec
                .iter()
                .zip(vec.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            (i, dist)
        })
        .collect();
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    for (rowid, dist) in distances.iter().take(5) {
        println!("  rowid={}, distance={:.4}", rowid, dist);
    }

    // Verify
    let ground_truth: Vec<i64> = distances.iter().take(5).map(|(r, _)| *r).collect();
    let hnsw_rowids: Vec<i64> = results.iter().map(|(r, _)| *r).collect();

    let matches = hnsw_rowids
        .iter()
        .filter(|r| ground_truth.contains(r))
        .count();
    let recall = matches as f64 / 5.0;

    println!("\n=== Recall ===");
    println!("Ground truth: {:?}", ground_truth);
    println!("HNSW results: {:?}", hnsw_rowids);
    println!("Matches: {}/5", matches);
    println!("Recall: {:.1}%", recall * 100.0);

    assert!(
        recall >= 0.95,
        "Recall should be at least 95%, got {:.1}%",
        recall * 100.0
    );
}

#[test]
fn test_hnsw_recall_128d_100v() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    const DIM: usize = 128;
    const NUM_VECTORS: i64 = 100;
    const K: i64 = 10;

    // Create table
    db.execute(
        "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[128])",
        [],
    )
    .unwrap();

    // Generate deterministic vectors
    let vectors: Vec<Vec<f32>> = (1..=NUM_VECTORS)
        .map(|i| {
            (0..DIM)
                .map(|j| ((i as usize * 7 + j * 13) % 100) as f32 / 100.0)
                .collect()
        })
        .collect();

    // Insert vectors
    db.execute("BEGIN", []).unwrap();
    for (idx, vec) in vectors.iter().enumerate() {
        let rowid = idx as i64 + 1;
        let bytes: Vec<u8> = vec.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Debug: check bytes size
        if idx == 0 {
            println!(
                "First vector bytes size: {} (expected: {})",
                bytes.len(),
                DIM * 4
            );
        }

        db.execute(
            "INSERT INTO vectors(rowid, embedding) VALUES (?, ?)",
            rusqlite::params![rowid, bytes],
        )
        .unwrap();

        // Check metadata after first insert
        if idx == 0 {
            let num_nodes: Result<i32, _> = db.query_row(
                "SELECT num_nodes FROM vectors_embedding_hnsw_meta WHERE id = 1",
                [],
                |row| row.get(0),
            );
            let hnsw_node_count: i64 = db
                .query_row(
                    "SELECT COUNT(*) FROM vectors_embedding_hnsw_nodes",
                    [],
                    |row| row.get(0),
                )
                .unwrap_or(0);
            println!(
                "After first insert: metadata.num_nodes={:?}, actual nodes={}",
                num_nodes, hnsw_node_count
            );
        }
    }
    db.execute("COMMIT", []).unwrap();

    // Check final state
    let hnsw_node_count: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM vectors_embedding_hnsw_nodes",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);
    let edge_count: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM vectors_embedding_hnsw_edges WHERE level = 0",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);
    let avg_edges: f64 = edge_count as f64 / hnsw_node_count as f64;
    println!("\nInserted {} vectors with {} dimensions", NUM_VECTORS, DIM);
    println!("HNSW nodes in table: {}", hnsw_node_count);
    println!(
        "HNSW edges at level 0: {} (avg {:.1} per node, expected ~{})",
        edge_count,
        avg_edges,
        32 * 2
    );

    // Check metadata (single-row schema)
    let (entry_point, entry_level, num_nodes): (i64, i32, i32) = db
        .query_row(
            "SELECT entry_point_rowid, entry_point_level, num_nodes \
             FROM vectors_embedding_hnsw_meta WHERE id = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap_or((-1, 0, 0));
    println!(
        "HNSW entry_point: {}, entry_level: {}, num_nodes: {}",
        entry_point, entry_level, num_nodes
    );

    // Query vector
    let query_vec: Vec<f32> = (0..DIM).map(|j| (j * 11 % 100) as f32 / 100.0).collect();
    let query_bytes: Vec<u8> = query_vec.iter().flat_map(|f| f.to_le_bytes()).collect();

    // HNSW search (use ef_search = 400 for better recall)
    db.execute(
        "UPDATE vectors_embedding_hnsw_meta SET ef_search = 400 WHERE id = 1",
        [],
    )
    .ok();
    let mut stmt = db
        .prepare("SELECT rowid, distance FROM vectors WHERE embedding MATCH ? AND k = ? ORDER BY distance")
        .unwrap();
    let hnsw_results: Vec<(i64, f64)> = stmt
        .query_map(rusqlite::params![query_bytes.clone(), K], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();

    println!("\nHNSW results ({} returned):", hnsw_results.len());
    for (rowid, dist) in &hnsw_results {
        println!("  rowid={}, distance={:.6}", rowid, dist);
    }

    // Brute force ground truth
    let mut distances: Vec<(i64, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(idx, vec)| {
            let rowid = idx as i64 + 1;
            let dist: f32 = query_vec
                .iter()
                .zip(vec.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            (rowid, dist)
        })
        .collect();
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let ground_truth: Vec<i64> = distances.iter().take(K as usize).map(|(r, _)| *r).collect();

    println!("\nGround truth:");
    for (rowid, dist) in distances.iter().take(K as usize) {
        println!("  rowid={}, distance={:.6}", rowid, dist);
    }

    // Calculate recall
    let hnsw_rowids: Vec<i64> = hnsw_results.iter().map(|(r, _)| *r).collect();
    let matches = hnsw_rowids
        .iter()
        .filter(|r| ground_truth.contains(r))
        .count();
    let recall = if ground_truth.is_empty() {
        0.0
    } else {
        matches as f64 / K as f64
    };

    println!("\n=== Recall ===");
    println!("Ground truth: {:?}", ground_truth);
    println!("HNSW results: {:?}", hnsw_rowids);
    println!("Matches: {}/{}", matches, K);
    println!("Recall: {:.1}%", recall * 100.0);

    assert!(
        recall >= 0.95,
        "Recall should be at least 95%, got {:.1}%",
        recall * 100.0
    );
}
