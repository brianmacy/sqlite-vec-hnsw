/// Performance Report: Measures key metrics and compares to requirements
///
/// Run with: cargo run --release --example performance_report
///
/// This is an example, not a benchmark, to avoid Criterion overhead
/// and get real-world performance numbers.
use rusqlite::Connection;
use std::time::Instant;

fn random_vector(dimensions: usize) -> Vec<f32> {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hash, Hasher};

    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();

    let mut state = RandomState::new().build_hasher();
    seed.hash(&mut state);

    (0..dimensions)
        .map(|i| {
            (i as u64 ^ state.finish()).hash(&mut state);
            ((state.finish() % 10000) as f32) / 10000.0
        })
        .collect()
}

fn main() {
    println!("\n=== SQLite-Vec-HNSW Performance Report ===\n");
    println!("Testing Rust implementation (Release build)");
    println!(
        "Hardware: {} ({})",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    println!();

    // Test 1: Vector Insert Rate
    println!("## Test 1: Vector Insert Rate (with HNSW indexing)");
    for dimensions in [128, 384, 768] {
        let db = Connection::open_in_memory().unwrap();
        sqlite_vec_hnsw::init(&db).unwrap();

        db.execute(
            &format!(
                "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[{}])",
                dimensions
            ),
            [],
        )
        .unwrap();

        let num_vectors = 500; // Reduced for faster testing
        let start = Instant::now();

        for i in 0..num_vectors {
            let vector = random_vector(dimensions);
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
                    "INSERT INTO vectors(rowid, embedding) VALUES ({}, vec_f32('{}'))",
                    i + 1,
                    vector_json
                ),
                [],
            )
            .unwrap();
        }

        let elapsed = start.elapsed();
        let rate = num_vectors as f64 / elapsed.as_secs_f64();

        println!(
            "  {}D: {:.1} vectors/sec ({} vectors in {:.1}s)",
            dimensions,
            rate,
            num_vectors,
            elapsed.as_secs_f64()
        );
    }

    println!();

    // Test 2: KNN Query Latency
    println!("## Test 2: KNN Query Latency (128D float32)");
    for num_vectors in [1_000, 5_000, 10_000] {
        let dimensions = 128;
        print!("  Building index with {} vectors... ", num_vectors);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let db = Connection::open_in_memory().unwrap();
        sqlite_vec_hnsw::init(&db).unwrap();

        db.execute(
            &format!(
                "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[{}])",
                dimensions
            ),
            [],
        )
        .unwrap();

        // Insert vectors
        for i in 0..num_vectors {
            let vector = random_vector(dimensions);
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
                    "INSERT INTO vectors(rowid, embedding) VALUES ({}, vec_f32('{}'))",
                    i + 1,
                    vector_json
                ),
                [],
            )
            .unwrap();
        }

        println!("done");

        // Measure query time
        let query = random_vector(dimensions);
        let query_json = format!(
            "[{}]",
            query
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );

        // Warm up
        for _ in 0..10 {
            let mut stmt = db
                .prepare(&format!(
                    "SELECT rowid, distance FROM vectors \
                     WHERE embedding MATCH vec_f32('{}') AND k = 10 \
                     ORDER BY distance",
                    query_json
                ))
                .unwrap();

            let _: Vec<(i64, f64)> = stmt
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap();
        }

        // Measure
        let num_queries = 100;
        let start = Instant::now();

        for _ in 0..num_queries {
            let mut stmt = db
                .prepare(&format!(
                    "SELECT rowid, distance FROM vectors \
                     WHERE embedding MATCH vec_f32('{}') AND k = 10 \
                     ORDER BY distance",
                    query_json
                ))
                .unwrap();

            let results: Vec<(i64, f64)> = stmt
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap();

            assert_eq!(results.len(), 10);
        }

        let elapsed = start.elapsed();
        let avg_latency = elapsed.as_micros() as f64 / num_queries as f64 / 1000.0;

        println!(
            "    → {:.2}ms per query (k=10, {} queries)",
            avg_latency, num_queries
        );
    }

    println!();

    // Test 3: Distance Calculations (SIMD)
    println!("## Test 3: Distance Calculations (SIMD via simsimd)");
    for dimensions in [128, 384, 768] {
        let db = Connection::open_in_memory().unwrap();
        sqlite_vec_hnsw::init(&db).unwrap();

        let vec1 = random_vector(dimensions);
        let vec2 = random_vector(dimensions);

        let vec1_json = format!(
            "[{}]",
            vec1.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        let vec2_json = format!(
            "[{}]",
            vec2.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );

        let num_calcs = 10_000;
        let start = Instant::now();

        for _ in 0..num_calcs {
            let _dist: f64 = db
                .query_row(
                    &format!(
                        "SELECT vec_distance_l2(vec_f32('{}'), vec_f32('{}'))",
                        vec1_json, vec2_json
                    ),
                    [],
                    |row| row.get(0),
                )
                .unwrap();
        }

        let elapsed = start.elapsed();
        let per_calc = elapsed.as_nanos() as f64 / num_calcs as f64;

        println!("  {}D L2: {:.1}ns per calculation", dimensions, per_calc);
    }

    println!();
    println!("## Performance Requirements (from CLAUDE.md)");
    println!("  Insert rate: Within 20% of C version (~170 vec/sec at 768D int8)");
    println!("  Query latency: Within 20% of C version (~2.8ms/query)");
    println!("  Recall: >95% at k=10 for 100K+ vectors");
    println!();
    println!("Note: C benchmarks from ~/dev/G2/dev/libs/external/sqlite-vec");
    println!("      Run this in release mode for accurate numbers");
}
