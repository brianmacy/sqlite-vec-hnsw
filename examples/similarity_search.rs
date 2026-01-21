//! Similarity search example using HNSW index
//!
//! Demonstrates:
//! - Creating a larger dataset of vectors
//! - Using HNSW search API directly (workaround for MATCH operator limitation)
//! - Finding k-nearest neighbors
//! - Measuring search performance

use rusqlite::Connection;
use sqlite_vec_hnsw::distance::DistanceMetric;
use sqlite_vec_hnsw::hnsw::{HnswMetadata, search};
use sqlite_vec_hnsw::vector::{Vector, VectorType};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Open an in-memory database
    let db = Connection::open_in_memory()?;

    // Initialize the extension
    sqlite_vec_hnsw::init(&db)?;

    println!("🔍 Similarity Search Example\n");

    // Create a virtual table with a 128-dimensional float32 vector column
    db.execute(
        "CREATE VIRTUAL TABLE embeddings USING vec0(vector float[128])",
        [],
    )?;

    println!("✓ Created virtual table with 128D vectors");

    // Generate and insert synthetic vectors
    let num_vectors = 100;
    println!("⏳ Inserting {} vectors...", num_vectors);

    let start = Instant::now();
    for i in 0..num_vectors {
        // Generate a synthetic vector (in practice, these would be from a model)
        let vector: Vec<f32> = (0..128)
            .map(|j| {
                // Create some structure: cluster vectors based on i
                let cluster = (i / 10) as f32;
                let noise = (i * 7 + j * 13) as f32 * 0.01;
                cluster + noise
            })
            .collect();

        // Convert to JSON string for insertion
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
        )?;
    }

    let insert_duration = start.elapsed();
    println!(
        "✓ Inserted {} vectors in {:.2}s ({:.1} vectors/sec)",
        num_vectors,
        insert_duration.as_secs_f64(),
        num_vectors as f64 / insert_duration.as_secs_f64()
    );

    // Create a query vector (similar to vectors in cluster 5)
    let query_vector: Vec<f32> = (0..128).map(|j| 5.0 + (j as f32 * 0.01)).collect();

    // Convert to bytes for HNSW search
    let mut query_bytes = Vec::with_capacity(128 * 4);
    for &val in &query_vector {
        query_bytes.extend_from_slice(&val.to_le_bytes());
    }

    println!("\n🎯 Query vector: cluster pattern ≈ 5.0");

    // Load HNSW metadata
    let metadata = HnswMetadata::load_from_db(&db, "embeddings", "vector")?
        .expect("HNSW metadata should exist");

    println!("\n📊 HNSW Index Info:");
    println!("  Nodes: {}", metadata.num_nodes);
    println!("  M: {}", metadata.params.m);
    println!("  ef_construction: {}", metadata.params.ef_construction);
    println!("  ef_search: {}", metadata.params.ef_search);
    println!(
        "  Entry point: rowid {} at level {}",
        metadata.entry_point_rowid, metadata.entry_point_level
    );

    // Perform HNSW search for k=10 nearest neighbors
    let k = 10;
    println!("\n🔎 Searching for {} nearest neighbors...", k);

    let search_start = Instant::now();
    let results = search::search_hnsw(
        &db,
        &metadata,
        "embeddings",
        "vector",
        &query_bytes,
        k,
        None, // Use default ef_search from metadata
        None, // No statement cache in standalone example
    )?;
    let search_duration = search_start.elapsed();

    println!(
        "✓ Search completed in {:.2}ms",
        search_duration.as_secs_f64() * 1000.0
    );

    // Display results
    println!("\n📈 Top {} Results:", results.len());
    println!("{:<8} {:<12} {:<15}", "Rank", "Rowid", "Distance");
    println!("{}", "-".repeat(35));

    for (rank, (rowid, distance)) in results.iter().enumerate() {
        // Fetch the actual vector to show cluster info
        let cluster = (rowid - 1) / 10;
        println!(
            "{:<8} {:<12} {:<15.6} (cluster {})",
            rank + 1,
            rowid,
            distance,
            cluster
        );
    }

    // Verify the results are sorted by distance
    let is_sorted = results.windows(2).all(|w| w[0].1 <= w[1].1);
    assert!(is_sorted, "Results should be sorted by distance");
    println!("\n✓ Results correctly sorted by distance");

    // Compare with brute force search for accuracy verification
    println!("\n🔬 Verifying accuracy with brute force search...");

    let brute_start = Instant::now();
    let mut brute_results = Vec::new();

    // Query all vectors and calculate distances
    let mut stmt = db.prepare("SELECT rowid, vector FROM embeddings")?;
    let rows = stmt.query_map([], |row| {
        let rowid: i64 = row.get(0)?;
        let vector_bytes: Vec<u8> = row.get(1)?;
        Ok((rowid, vector_bytes))
    })?;

    for row in rows {
        let (rowid, vector_bytes) = row?;
        let vector = Vector::from_blob(&vector_bytes, VectorType::Float32, 128)?;
        let query_vec = Vector::from_blob(&query_bytes, VectorType::Float32, 128)?;
        let distance =
            sqlite_vec_hnsw::distance::distance(&query_vec, &vector, DistanceMetric::L2)?;
        brute_results.push((rowid, distance));
    }

    brute_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    brute_results.truncate(k);
    let brute_duration = brute_start.elapsed();

    println!(
        "✓ Brute force completed in {:.2}ms",
        brute_duration.as_secs_f64() * 1000.0
    );

    // Calculate recall (what percentage of true nearest neighbors were found)
    let brute_rowids: std::collections::HashSet<_> =
        brute_results.iter().map(|(rowid, _)| rowid).collect();
    let hnsw_rowids: std::collections::HashSet<_> =
        results.iter().map(|(rowid, _)| rowid).collect();
    let intersection = brute_rowids.intersection(&hnsw_rowids).count();
    let recall = (intersection as f64 / k as f64) * 100.0;

    println!("\n📊 Search Performance:");
    println!(
        "  HNSW search: {:.2}ms",
        search_duration.as_secs_f64() * 1000.0
    );
    println!(
        "  Brute force: {:.2}ms",
        brute_duration.as_secs_f64() * 1000.0
    );
    println!(
        "  Speedup: {:.1}x faster",
        brute_duration.as_secs_f64() / search_duration.as_secs_f64()
    );
    println!("  Recall@{}: {:.1}%", k, recall);

    if recall >= 80.0 {
        println!("\n✅ Excellent recall! HNSW is working correctly.");
    } else {
        println!("\n⚠️  Low recall - may need more vectors or tuning");
    }

    println!("\n💡 Note: MATCH operator support would enable KNN queries via SQL:");
    println!("   SELECT rowid, distance FROM embeddings");
    println!("   WHERE vector MATCH vec_f32('[...]') AND k = 10");
    println!("   ORDER BY distance");

    Ok(())
}
