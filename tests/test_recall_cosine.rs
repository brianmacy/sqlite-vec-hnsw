use rusqlite::Connection;
use std::collections::HashSet;

/// Compute cosine distance: 1 - (a·b)/(|a||b|)
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }
    1.0 - (dot / (norm_a * norm_b))
}

#[test]
fn test_recall_with_cosine_distance() {
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    // Default is cosine distance (best for embeddings)
    db.execute(
        "CREATE VIRTUAL TABLE embeddings USING vec0(vector float[128] hnsw())",
        [],
    )
    .unwrap();

    let num_vectors = 100i64;

    for i in 0..num_vectors {
        let vector: Vec<f32> = (0..128)
            .map(|j| {
                let cluster = (i / 10) as f32;
                let noise = (i * 7 + j * 13) as f32 * 0.01;
                cluster + noise
            })
            .collect();

        let vector_str = format!(
            "[{}]",
            vector
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );

        db.execute(
            &format!(
                "INSERT INTO embeddings(rowid, vector) VALUES ({}, vec_f32('{}'))",
                i + 1,
                vector_str
            ),
            [],
        )
        .unwrap();
    }

    let query_vector: Vec<f32> = (0..128).map(|j| 5.0 + (j as f32 * 0.01)).collect();
    let query_bytes: Vec<u8> = query_vector.iter().flat_map(|f| f.to_le_bytes()).collect();

    println!("\nQuery first 5 values: {:?}", &query_vector[..5]);

    // HNSW search (uses cosine by default)
    let mut stmt = db
        .prepare(
            "SELECT rowid, distance FROM embeddings WHERE vector MATCH ? AND k = 10 ORDER BY distance",
        )
        .unwrap();
    let hnsw_results: Vec<(i64, f64)> = stmt
        .query_map(rusqlite::params![query_bytes], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();

    println!("\nHNSW top 10 (cosine):");
    for (rowid, dist) in &hnsw_results {
        let cluster = (rowid - 1) / 10;
        println!("  rowid={:3}, cluster={}, dist={:.6}", rowid, cluster, dist);
    }

    // Brute force with cosine distance
    let mut stmt = db
        .prepare("SELECT rowid, vec00 FROM embeddings_data")
        .unwrap();
    let rows: Vec<(i64, Vec<u8>)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();

    let mut brute_results: Vec<(i64, f32)> = rows
        .iter()
        .map(|(rowid, bytes)| {
            let floats: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let dist = cosine_distance(&query_vector, &floats);
            (*rowid, dist)
        })
        .collect();

    brute_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    println!("\nBrute force top 10 (cosine):");
    for (rowid, dist) in brute_results.iter().take(10) {
        let cluster = (rowid - 1) / 10;
        println!("  rowid={:3}, cluster={}, dist={:.6}", rowid, cluster, dist);
    }

    // Calculate recall
    let hnsw_set: HashSet<i64> = hnsw_results.iter().map(|(r, _)| *r).collect();
    let brute_set: HashSet<i64> = brute_results.iter().take(10).map(|(r, _)| *r).collect();
    let intersection = hnsw_set.intersection(&brute_set).count();

    println!("\nRecall: {}/10 = {}%", intersection, intersection * 10);

    assert!(
        intersection >= 9,
        "Recall should be at least 90%, got {}%",
        intersection * 10
    );
}
