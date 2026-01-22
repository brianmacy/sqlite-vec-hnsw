// Profile int8 cosine mode to find optimization opportunities
// Uses real 384D embeddings

use rusqlite::Connection;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;
use tempfile::TempDir;

const EMBEDDINGS_FILE: &str = "test_data/opensanctions_embeddings.jsonl";
const VECTOR_DIM: usize = 384;

fn load_embeddings(max_vectors: usize) -> Vec<Vec<f32>> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let path = format!("{}/{}", manifest_dir, EMBEDDINGS_FILE);
    let file = File::open(&path).unwrap();
    let reader = BufReader::new(file);

    reader
        .lines()
        .take(max_vectors)
        .map(|line| serde_json::from_str(&line.unwrap()).unwrap())
        .collect()
}

fn vector_to_json(vector: &[f32]) -> String {
    format!(
        "[{}]",
        vector
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

#[test]
#[ignore]
fn profile_single_thread_insert() {
    println!("\n=== Single-Thread Insert Profile (Int8 Cosine) ===\n");

    let vectors = load_embeddings(5000);
    println!("Loaded {} vectors of {}D\n", vectors.len(), VECTOR_DIM);

    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("profile.db");

    let db = Connection::open(&db_path).unwrap();
    db.execute_batch(
        "PRAGMA page_size=16384;
         PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         PRAGMA cache_size=10000;
         PRAGMA temp_store=MEMORY;",
    )
    .unwrap();

    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute(
        &format!("CREATE VIRTUAL TABLE vectors USING vec0(embedding float[{}] hnsw(index_quantization=int8))", VECTOR_DIM),
        [],
    ).unwrap();

    // Warm up
    for (i, v) in vectors.iter().take(100).enumerate() {
        let sql = format!(
            "INSERT INTO vectors(rowid, embedding) VALUES ({}, vec_f32('{}'))",
            i + 1,
            vector_to_json(v)
        );
        db.execute(&sql, []).unwrap();
    }

    // Profile batches to see how performance changes with index size
    let batch_sizes = [100, 500, 1000, 2000, 5000];
    let mut total_inserted = 100;

    println!(
        "{:>10} {:>10} {:>12} {:>10}",
        "Batch", "Total", "Time (ms)", "Vec/sec"
    );
    println!("{}", "-".repeat(50));

    for &batch_size in &batch_sizes {
        if total_inserted + batch_size > vectors.len() {
            break;
        }

        let start = Instant::now();
        for i in 0..batch_size {
            let rowid = total_inserted + i + 1;
            let v = &vectors[rowid - 1];
            let sql = format!(
                "INSERT INTO vectors(rowid, embedding) VALUES ({}, vec_f32('{}'))",
                rowid,
                vector_to_json(v)
            );
            db.execute(&sql, []).unwrap();
        }
        let elapsed = start.elapsed();

        total_inserted += batch_size;
        let rate = batch_size as f64 / elapsed.as_secs_f64();
        println!(
            "{:>10} {:>10} {:>12.1} {:>10.1}",
            batch_size,
            total_inserted,
            elapsed.as_millis(),
            rate
        );
    }

    // Check final stats
    let edge_count: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM vectors_embedding_hnsw_edges",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let node_count: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM vectors_embedding_hnsw_nodes",
            [],
            |r| r.get(0),
        )
        .unwrap();

    println!(
        "\nFinal: {} nodes, {} edges ({:.1} edges/node)",
        node_count,
        edge_count,
        edge_count as f64 / node_count as f64
    );
}

#[test]
#[ignore]
fn profile_insert_breakdown() {
    println!("\n=== Insert Time Breakdown (Int8 Cosine) ===\n");

    let vectors = load_embeddings(1000);
    println!("Loaded {} vectors of {}D\n", vectors.len(), VECTOR_DIM);

    // Pre-convert to JSON to isolate SQL parsing overhead
    let json_vectors: Vec<String> = vectors.iter().map(|v| vector_to_json(v)).collect();

    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("profile.db");

    let db = Connection::open(&db_path).unwrap();
    db.execute_batch(
        "PRAGMA page_size=16384;
         PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         PRAGMA cache_size=10000;
         PRAGMA temp_store=MEMORY;",
    )
    .unwrap();

    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute(
        &format!("CREATE VIRTUAL TABLE vectors USING vec0(embedding float[{}] hnsw(index_quantization=int8))", VECTOR_DIM),
        [],
    ).unwrap();

    // Seed with some data first
    for (i, json) in json_vectors.iter().take(500).enumerate() {
        let sql = format!(
            "INSERT INTO vectors(rowid, embedding) VALUES ({}, vec_f32('{}'))",
            i + 1,
            json
        );
        db.execute(&sql, []).unwrap();
    }
    println!("Seeded with 500 vectors\n");

    // Now measure next 500 inserts in detail
    let count = 500;

    let start = Instant::now();
    for i in 0..count {
        let rowid = 501 + i;
        let json = &json_vectors[rowid - 1];
        let sql = format!(
            "INSERT INTO vectors(rowid, embedding) VALUES ({}, vec_f32('{}'))",
            rowid, json
        );
        db.execute(&sql, []).unwrap();
    }
    let total_time = start.elapsed();

    let avg_us = total_time.as_micros() as f64 / count as f64;
    let rate = count as f64 / total_time.as_secs_f64();

    println!("Results for {} inserts (500-1000 vectors in index):", count);
    println!("  Total time:    {:.1} ms", total_time.as_millis());
    println!(
        "  Avg per insert: {:.1} µs ({:.1} ms)",
        avg_us,
        avg_us / 1000.0
    );
    println!("  Throughput:    {:.1} vec/sec", rate);

    // Compare with prepared statement
    println!("\n--- With Prepared Statement ---");

    let temp_dir2 = TempDir::new().unwrap();
    let db_path2 = temp_dir2.path().join("profile2.db");
    let db2 = Connection::open(&db_path2).unwrap();
    db2.execute_batch(
        "PRAGMA page_size=16384;
         PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         PRAGMA cache_size=10000;
         PRAGMA temp_store=MEMORY;",
    )
    .unwrap();
    sqlite_vec_hnsw::init(&db2).unwrap();
    db2.execute(
        &format!("CREATE VIRTUAL TABLE vectors USING vec0(embedding float[{}] hnsw(index_quantization=int8))", VECTOR_DIM),
        [],
    ).unwrap();

    // Seed
    for (i, json) in json_vectors.iter().take(500).enumerate() {
        let sql = format!(
            "INSERT INTO vectors(rowid, embedding) VALUES ({}, vec_f32('{}'))",
            i + 1,
            json
        );
        db2.execute(&sql, []).unwrap();
    }

    // Note: Can't use prepared statements with vec_f32() function easily
    // The real bottleneck is likely in the HNSW algorithm itself

    println!("\nNote: The bottleneck is likely in HNSW graph operations, not SQL parsing.");
}

#[test]
#[ignore]
fn profile_search_breakdown() {
    println!("\n=== Search Time Breakdown (Int8 Cosine) ===\n");

    let vectors = load_embeddings(10000);
    println!("Loaded {} vectors of {}D\n", vectors.len(), VECTOR_DIM);

    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("profile.db");

    let db = Connection::open(&db_path).unwrap();
    db.execute_batch(
        "PRAGMA page_size=16384;
         PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         PRAGMA cache_size=10000;
         PRAGMA temp_store=MEMORY;",
    )
    .unwrap();

    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute(
        &format!("CREATE VIRTUAL TABLE vectors USING vec0(embedding float[{}] hnsw(index_quantization=int8))", VECTOR_DIM),
        [],
    ).unwrap();

    // Insert all vectors
    println!("Inserting {} vectors...", vectors.len());
    let start = Instant::now();
    for (i, v) in vectors.iter().enumerate() {
        let sql = format!(
            "INSERT INTO vectors(rowid, embedding) VALUES ({}, vec_f32('{}'))",
            i + 1,
            vector_to_json(v)
        );
        db.execute(&sql, []).unwrap();
    }
    println!(
        "Insert time: {:.1}s ({:.1} vec/sec)\n",
        start.elapsed().as_secs_f64(),
        vectors.len() as f64 / start.elapsed().as_secs_f64()
    );

    // Profile searches at different k values
    println!(
        "{:>5} {:>10} {:>12} {:>10}",
        "k", "Queries", "Time (ms)", "QPS"
    );
    println!("{}", "-".repeat(45));

    for k in [1, 5, 10, 20, 50, 100] {
        let num_queries = 100;
        let start = Instant::now();

        for i in 0..num_queries {
            let query = &vectors[i * 37 % vectors.len()];
            let sql = format!(
                "SELECT rowid, distance FROM vectors WHERE embedding MATCH vec_f32('{}') AND k = {} ORDER BY distance",
                vector_to_json(query),
                k
            );
            let mut stmt = db.prepare(&sql).unwrap();
            let _results: Vec<(i64, f64)> = stmt
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
        }

        let elapsed = start.elapsed();
        let qps = num_queries as f64 / elapsed.as_secs_f64();
        println!(
            "{:>5} {:>10} {:>12.1} {:>10.1}",
            k,
            num_queries,
            elapsed.as_millis(),
            qps
        );
    }

    // Profile ef_search impact
    println!("\n--- ef_search Impact (k=10) ---");
    println!(
        "{:>10} {:>10} {:>12} {:>10}",
        "ef_search", "Queries", "Time (ms)", "QPS"
    );
    println!("{}", "-".repeat(50));

    for ef in [10, 20, 50, 100, 200, 400] {
        let num_queries = 100;

        // Set ef_search via PRAGMA or table option
        // For now, the default ef_search is used
        // TODO: Add way to configure ef_search per query

        let start = Instant::now();
        for i in 0..num_queries {
            let query = &vectors[i * 37 % vectors.len()];
            let sql = format!(
                "SELECT rowid, distance FROM vectors WHERE embedding MATCH vec_f32('{}') AND k = 10 ORDER BY distance",
                vector_to_json(query)
            );
            let mut stmt = db.prepare(&sql).unwrap();
            let _results: Vec<(i64, f64)> = stmt
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
        }
        let elapsed = start.elapsed();
        let qps = num_queries as f64 / elapsed.as_secs_f64();

        // Note: ef_search is currently fixed, showing same results
        println!(
            "{:>10} {:>10} {:>12.1} {:>10.1}",
            ef,
            num_queries,
            elapsed.as_millis(),
            qps
        );

        if ef == 10 {
            println!("  (Note: ef_search tuning not yet exposed - showing baseline)");
            break;
        }
    }
}

#[test]
#[ignore]
fn profile_distance_calculations() {
    println!("\n=== Distance Calculation Profile ===\n");

    use sqlite_vec_hnsw::distance::{DistanceMetric, distance};
    use sqlite_vec_hnsw::vector::Vector;

    let vectors = load_embeddings(1000);
    println!("Loaded {} vectors of {}D\n", vectors.len(), VECTOR_DIM);

    // Convert to Vector types
    let f32_vectors: Vec<Vector> = vectors.iter().map(|v| Vector::from_f32(v)).collect();

    let i8_vectors: Vec<Vector> = f32_vectors
        .iter()
        .map(|v| v.quantize_int8_for_index().unwrap())
        .collect();

    // Benchmark float32 L2 distance
    let num_pairs = 100000;

    println!("Benchmarking {} distance calculations:\n", num_pairs);

    // Float32 L2
    let start = Instant::now();
    for i in 0..num_pairs {
        let a = &f32_vectors[i % f32_vectors.len()];
        let b = &f32_vectors[(i * 7 + 1) % f32_vectors.len()];
        let _ = distance(a, b, DistanceMetric::L2).unwrap();
    }
    let f32_l2_time = start.elapsed();
    println!(
        "Float32 L2:     {:>8.1} ms ({:.0} ops/sec)",
        f32_l2_time.as_millis(),
        num_pairs as f64 / f32_l2_time.as_secs_f64()
    );

    // Float32 Cosine
    let start = Instant::now();
    for i in 0..num_pairs {
        let a = &f32_vectors[i % f32_vectors.len()];
        let b = &f32_vectors[(i * 7 + 1) % f32_vectors.len()];
        let _ = distance(a, b, DistanceMetric::Cosine).unwrap();
    }
    let f32_cos_time = start.elapsed();
    println!(
        "Float32 Cosine: {:>8.1} ms ({:.0} ops/sec)",
        f32_cos_time.as_millis(),
        num_pairs as f64 / f32_cos_time.as_secs_f64()
    );

    // Int8 L2
    let start = Instant::now();
    for i in 0..num_pairs {
        let a = &i8_vectors[i % i8_vectors.len()];
        let b = &i8_vectors[(i * 7 + 1) % i8_vectors.len()];
        let _ = distance(a, b, DistanceMetric::L2).unwrap();
    }
    let i8_l2_time = start.elapsed();
    println!(
        "Int8 L2:        {:>8.1} ms ({:.0} ops/sec) ({:.2}x vs f32)",
        i8_l2_time.as_millis(),
        num_pairs as f64 / i8_l2_time.as_secs_f64(),
        f32_l2_time.as_secs_f64() / i8_l2_time.as_secs_f64()
    );

    // Int8 Cosine
    let start = Instant::now();
    for i in 0..num_pairs {
        let a = &i8_vectors[i % i8_vectors.len()];
        let b = &i8_vectors[(i * 7 + 1) % i8_vectors.len()];
        let _ = distance(a, b, DistanceMetric::Cosine).unwrap();
    }
    let i8_cos_time = start.elapsed();
    println!(
        "Int8 Cosine:    {:>8.1} ms ({:.0} ops/sec) ({:.2}x vs f32)",
        i8_cos_time.as_millis(),
        num_pairs as f64 / i8_cos_time.as_secs_f64(),
        f32_cos_time.as_secs_f64() / i8_cos_time.as_secs_f64()
    );

    // Quantization overhead
    println!("\n--- Quantization Overhead ---");
    let start = Instant::now();
    for v in &f32_vectors {
        let _ = v.quantize_int8_for_index().unwrap();
    }
    let quant_time = start.elapsed();
    println!(
        "Quantize {} vectors: {:.1} ms ({:.1} µs/vec)",
        f32_vectors.len(),
        quant_time.as_millis(),
        quant_time.as_micros() as f64 / f32_vectors.len() as f64
    );

    // Normalization overhead
    println!("\n--- Normalization Overhead ---");
    let start = Instant::now();
    for v in &f32_vectors {
        let _ = v.normalize().unwrap();
    }
    let norm_time = start.elapsed();
    println!(
        "Normalize {} vectors: {:.1} ms ({:.1} µs/vec)",
        f32_vectors.len(),
        norm_time.as_millis(),
        norm_time.as_micros() as f64 / f32_vectors.len() as f64
    );
}

#[test]
#[ignore]
fn profile_hnsw_graph_operations() {
    println!("\n=== HNSW Graph Operations Profile ===\n");

    let vectors = load_embeddings(5000);
    println!(
        "Testing with {} vectors of {}D\n",
        vectors.len(),
        VECTOR_DIM
    );

    // Test different M values to see impact on performance
    println!(
        "{:>5} {:>12} {:>10} {:>10} {:>12}",
        "M", "Insert/sec", "Edges", "Edges/N", "Size KB"
    );
    println!("{}", "-".repeat(55));

    for m in [8, 16, 32, 64] {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("profile.db");

        let db = Connection::open(&db_path).unwrap();
        db.execute_batch(
            "PRAGMA page_size=16384;
             PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA cache_size=10000;
             PRAGMA temp_store=MEMORY;",
        )
        .unwrap();

        sqlite_vec_hnsw::init(&db).unwrap();

        db.execute(
            &format!("CREATE VIRTUAL TABLE vectors USING vec0(embedding float[{}] hnsw(index_quantization=int8, M={}))", VECTOR_DIM, m),
            [],
        ).unwrap();

        let start = Instant::now();
        for (i, v) in vectors.iter().enumerate() {
            let sql = format!(
                "INSERT INTO vectors(rowid, embedding) VALUES ({}, vec_f32('{}'))",
                i + 1,
                vector_to_json(v)
            );
            db.execute(&sql, []).unwrap();
        }
        let elapsed = start.elapsed();
        let rate = vectors.len() as f64 / elapsed.as_secs_f64();

        let edge_count: i64 = db
            .query_row(
                "SELECT COUNT(*) FROM vectors_embedding_hnsw_edges",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let db_size = std::fs::metadata(&db_path)
            .map(|m| m.len() / 1024)
            .unwrap_or(0);

        println!(
            "{:>5} {:>12.1} {:>10} {:>10.1} {:>12}",
            m,
            rate,
            edge_count,
            edge_count as f64 / vectors.len() as f64,
            db_size
        );
    }

    // Test different ef_construction values
    println!("\n--- ef_construction Impact (M=16) ---");
    println!(
        "{:>15} {:>12} {:>10}",
        "ef_construction", "Insert/sec", "Edges"
    );
    println!("{}", "-".repeat(40));

    for ef in [50, 100, 200, 400] {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("profile.db");

        let db = Connection::open(&db_path).unwrap();
        db.execute_batch(
            "PRAGMA page_size=16384;
             PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA cache_size=10000;
             PRAGMA temp_store=MEMORY;",
        )
        .unwrap();

        sqlite_vec_hnsw::init(&db).unwrap();

        db.execute(
            &format!("CREATE VIRTUAL TABLE vectors USING vec0(embedding float[{}] hnsw(index_quantization=int8, M=16, ef_construction={}))", VECTOR_DIM, ef),
            [],
        ).unwrap();

        let start = Instant::now();
        for (i, v) in vectors.iter().enumerate() {
            let sql = format!(
                "INSERT INTO vectors(rowid, embedding) VALUES ({}, vec_f32('{}'))",
                i + 1,
                vector_to_json(v)
            );
            db.execute(&sql, []).unwrap();
        }
        let elapsed = start.elapsed();
        let rate = vectors.len() as f64 / elapsed.as_secs_f64();

        let edge_count: i64 = db
            .query_row(
                "SELECT COUNT(*) FROM vectors_embedding_hnsw_edges",
                [],
                |r| r.get(0),
            )
            .unwrap();

        println!("{:>15} {:>12.1} {:>10}", ef, rate, edge_count);
    }
}
