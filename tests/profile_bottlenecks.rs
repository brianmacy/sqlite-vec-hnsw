// Detailed bottleneck analysis for HNSW insert operations
// Profiles: RNG heuristic, HashSet vs FxHashSet, batch fetching, blob parsing

use rusqlite::Connection;
use std::collections::{BinaryHeap, HashSet};
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

fn f32_to_bytes(vec: &[f32]) -> Vec<u8> {
    vec.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[test]
#[ignore]
fn profile_rng_heuristic_cost() {
    println!("\n{}", "=".repeat(80));
    println!("RNG HEURISTIC COST ANALYSIS");
    println!("Testing O(n²) distance calculations in pruning");
    println!("{}", "=".repeat(80));

    let vectors = load_embeddings(200);
    let vec_bytes: Vec<Vec<u8>> = vectors.iter().map(|v| f32_to_bytes(v)).collect();

    // Simulate prune with different neighbor counts
    for neighbor_count in [32, 64, 96, 128] {
        let max_connections = 64usize;

        // Build candidates (center = vec[0], candidates = vec[1..neighbor_count])
        let center = &vectors[0];
        let mut candidates: Vec<(usize, f32, &[u8])> = Vec::new();
        for i in 1..=neighbor_count {
            let dist = l2_distance(center, &vectors[i]);
            candidates.push((i, dist, &vec_bytes[i]));
        }
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Measure RNG heuristic cost
        let start = Instant::now();
        let iterations = 100;
        let mut total_distances = 0u64;

        for _ in 0..iterations {
            let mut selected: Vec<Vec<f32>> = Vec::with_capacity(max_connections);
            let mut distance_calcs = 0u64;

            for (_, dist_to_center, blob) in &candidates {
                if selected.len() >= max_connections {
                    break;
                }

                let candidate_vec = bytes_to_f32(blob);

                // RNG heuristic: compare to ALL selected
                let mut good = true;
                for selected_vec in &selected {
                    distance_calcs += 1;
                    let dist_to_selected = l2_distance(&candidate_vec, selected_vec);
                    if dist_to_selected < *dist_to_center {
                        good = false;
                        break;
                    }
                }

                if good {
                    selected.push(candidate_vec);
                }
            }
            total_distances += distance_calcs;
        }

        let elapsed = start.elapsed();
        let avg_distances = total_distances / iterations as u64;
        let us_per_prune = elapsed.as_micros() / iterations as u128;

        println!(
            "\nNeighbors={}, MaxConnections={}:",
            neighbor_count, max_connections
        );
        println!("  Avg distance calcs per prune: {}", avg_distances);
        println!("  Time per prune: {} µs", us_per_prune);
        println!(
            "  Theoretical max (n²): {}",
            neighbor_count * max_connections
        );
    }

    println!("\n{}", "=".repeat(80));
}

#[test]
#[ignore]
fn profile_hashset_vs_alternatives() {
    println!("\n{}", "=".repeat(80));
    println!("VISITED SET PERFORMANCE COMPARISON");
    println!("HashSet (SipHash) vs alternatives for visited tracking");
    println!("{}", "=".repeat(80));

    // Simulate search pattern: insert rowids, then check membership
    let num_ops = 100_000;
    let rowids: Vec<i64> = (1..=num_ops as i64).collect();

    // Standard HashSet (SipHash - cryptographic, slower)
    let start = Instant::now();
    let mut visited: HashSet<i64> = HashSet::new();
    for &rowid in &rowids {
        if !visited.contains(&rowid) {
            visited.insert(rowid);
        }
    }
    let hashset_time = start.elapsed();

    // HashSet with capacity pre-allocated
    let start = Instant::now();
    let mut visited_prealloc: HashSet<i64> = HashSet::with_capacity(num_ops);
    for &rowid in &rowids {
        if !visited_prealloc.contains(&rowid) {
            visited_prealloc.insert(rowid);
        }
    }
    let hashset_prealloc_time = start.elapsed();

    // Vec<bool> for dense rowids (if rowids are sequential)
    let start = Instant::now();
    let mut visited_vec: Vec<bool> = vec![false; num_ops + 1];
    for &rowid in &rowids {
        let idx = rowid as usize;
        if !visited_vec[idx] {
            visited_vec[idx] = true;
        }
    }
    let vec_time = start.elapsed();

    println!("\n{} operations:", num_ops);
    println!(
        "  HashSet (SipHash):       {:>8} µs",
        hashset_time.as_micros()
    );
    println!(
        "  HashSet (prealloc):      {:>8} µs",
        hashset_prealloc_time.as_micros()
    );
    println!("  Vec<bool> (dense):       {:>8} µs", vec_time.as_micros());
    println!("\nSpeedup:");
    println!(
        "  Prealloc vs default: {:.2}x",
        hashset_time.as_nanos() as f64 / hashset_prealloc_time.as_nanos() as f64
    );
    println!(
        "  Vec<bool> vs HashSet: {:.2}x",
        hashset_time.as_nanos() as f64 / vec_time.as_nanos() as f64
    );

    println!("\n{}", "=".repeat(80));
}

#[test]
#[ignore]
fn profile_blob_parsing_overhead() {
    println!("\n{}", "=".repeat(80));
    println!("BLOB PARSING OVERHEAD");
    println!("Measuring Vector::from_blob equivalent cost");
    println!("{}", "=".repeat(80));

    let vectors = load_embeddings(1000);
    let vec_bytes: Vec<Vec<u8>> = vectors.iter().map(|v| f32_to_bytes(v)).collect();

    let iterations = 100_000;

    // Parse f32 vectors from bytes
    let start = Instant::now();
    for i in 0..iterations {
        let blob = &vec_bytes[i % vec_bytes.len()];
        let _parsed = bytes_to_f32(blob);
    }
    let parse_time = start.elapsed();

    // Parse + immediate distance calculation (common pattern)
    let query_bytes = f32_to_bytes(&vectors[0]);
    let query_vec = bytes_to_f32(&query_bytes);
    let start = Instant::now();
    for i in 0..iterations {
        let blob = &vec_bytes[i % vec_bytes.len()];
        let parsed = bytes_to_f32(blob);
        let _dist = l2_distance(&query_vec, &parsed);
    }
    let parse_dist_time = start.elapsed();

    // Just distance calculation (no parsing)
    let start = Instant::now();
    for i in 0..iterations {
        let vec = &vectors[i % vectors.len()];
        let _dist = l2_distance(&vectors[0], vec);
    }
    let dist_only_time = start.elapsed();

    println!("\n{} iterations, {}D vectors:", iterations, VECTOR_DIM);
    println!(
        "  Parse only:          {:>8} µs ({:.2} µs/op)",
        parse_time.as_micros(),
        parse_time.as_micros() as f64 / iterations as f64
    );
    println!(
        "  Parse + distance:    {:>8} µs ({:.2} µs/op)",
        parse_dist_time.as_micros(),
        parse_dist_time.as_micros() as f64 / iterations as f64
    );
    println!(
        "  Distance only:       {:>8} µs ({:.2} µs/op)",
        dist_only_time.as_micros(),
        dist_only_time.as_micros() as f64 / iterations as f64
    );
    println!(
        "\nParsing overhead: {:.1}% of parse+distance",
        (parse_time.as_nanos() as f64 / parse_dist_time.as_nanos() as f64) * 100.0
    );

    println!("\n{}", "=".repeat(80));
}

#[test]
#[ignore]
fn profile_binary_heap_overhead() {
    println!("\n{}", "=".repeat(80));
    println!("BINARY HEAP OVERHEAD");
    println!("Measuring heap operations cost");
    println!("{}", "=".repeat(80));

    #[derive(Debug, Clone)]
    #[allow(dead_code)]
    struct Candidate {
        rowid: i64,
        distance: f32,
    }

    impl PartialEq for Candidate {
        fn eq(&self, other: &Self) -> bool {
            self.distance == other.distance
        }
    }
    impl Eq for Candidate {}
    impl PartialOrd for Candidate {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Candidate {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            other
                .distance
                .partial_cmp(&self.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        }
    }

    let iterations = 100_000;

    // Simulate search pattern: push candidates, pop best, trim to ef
    let ef = 200;
    let candidates_per_iter = 64;

    let start = Instant::now();
    for i in 0..iterations {
        let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(ef);

        for j in 0..candidates_per_iter {
            let distance = ((i * 37 + j * 17) % 1000) as f32 / 1000.0;
            heap.push(Candidate {
                rowid: j as i64,
                distance,
            });

            // Trim to ef
            while heap.len() > ef {
                heap.pop();
            }
        }
    }
    let heap_time = start.elapsed();

    // Compare to Vec with sort (alternative approach)
    let start = Instant::now();
    for i in 0..iterations {
        let mut vec: Vec<Candidate> = Vec::with_capacity(candidates_per_iter);

        for j in 0..candidates_per_iter {
            let distance = ((i * 37 + j * 17) % 1000) as f32 / 1000.0;
            vec.push(Candidate {
                rowid: j as i64,
                distance,
            });
        }
        vec.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        vec.truncate(ef);
    }
    let vec_sort_time = start.elapsed();

    println!(
        "\n{} iterations, {} candidates each, ef={}:",
        iterations, candidates_per_iter, ef
    );
    println!("  BinaryHeap (push+trim): {:>8} µs", heap_time.as_micros());
    println!(
        "  Vec + sort + truncate:  {:>8} µs",
        vec_sort_time.as_micros()
    );
    println!(
        "\nRatio: {:.2}x",
        vec_sort_time.as_nanos() as f64 / heap_time.as_nanos() as f64
    );

    println!("\n{}", "=".repeat(80));
}

#[test]
#[ignore]
#[allow(clippy::needless_range_loop)]
fn profile_full_insert_breakdown() {
    println!("\n{}", "=".repeat(80));
    println!("FULL INSERT BREAKDOWN WITH TIMING");
    println!("Config: int8, M=64, ef_construction=200");
    println!("{}", "=".repeat(80));

    let vectors = load_embeddings(5000);
    println!("\nLoaded {} vectors of {}D\n", vectors.len(), VECTOR_DIM);

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
        &format!(
            "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[{}] hnsw(index_quantization=int8, M=64, ef_construction=200))",
            VECTOR_DIM
        ),
        [],
    ).unwrap();

    // Insert with detailed timing at checkpoints
    let checkpoints = [100, 500, 1000, 2000, 3000, 4000, 5000];
    let mut last_checkpoint = 0;

    println!(
        "{:>8} {:>10} {:>12} {:>10}",
        "Vectors", "Batch", "Time (ms)", "Vec/sec"
    );
    println!("{}", "-".repeat(45));

    for &checkpoint in &checkpoints {
        if checkpoint > vectors.len() {
            break;
        }

        let batch_size = checkpoint - last_checkpoint;
        let start = Instant::now();

        for i in last_checkpoint..checkpoint {
            let sql = format!(
                "INSERT INTO vectors(rowid, embedding) VALUES ({}, vec_f32('{}'))",
                i + 1,
                vector_to_json(&vectors[i])
            );
            db.execute(&sql, []).unwrap();
        }

        let elapsed = start.elapsed();
        let rate = batch_size as f64 / elapsed.as_secs_f64();

        println!(
            "{:>8} {:>10} {:>12.1} {:>10.1}",
            checkpoint,
            batch_size,
            elapsed.as_millis(),
            rate
        );

        last_checkpoint = checkpoint;
    }

    // Print timing stats from the extension
    println!("\nInternal timing breakdown (call sqlite_vec_hnsw::print_insert_timing_stats):");
    println!("  (Enable timing counters in Rust code to see detailed breakdown)");

    // Get final stats
    let edges: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM vectors_embedding_hnsw_edges",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let db_size = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);

    println!("\nFinal stats:");
    println!("  Vectors: {}", last_checkpoint);
    println!(
        "  Edges: {} ({:.1} per node)",
        edges,
        edges as f64 / last_checkpoint as f64
    );
    println!("  DB size: {:.1} MB", db_size as f64 / 1024.0 / 1024.0);

    println!("\n{}", "=".repeat(80));
}

#[test]
#[ignore]
fn identify_optimization_opportunities() {
    println!("\n{}", "=".repeat(80));
    println!("OPTIMIZATION OPPORTUNITIES SUMMARY");
    println!("{}", "=".repeat(80));

    println!(
        "
1. RNG HEURISTIC O(n²) - CRITICAL
   Location: insert.rs:200-230
   Issue: Each prune compares candidates to ALL selected neighbors
   Impact: With M=64, up to 128*64 = 8192 distance calculations per prune
   Fix: Skip RNG heuristic, use simple distance-based selection
        OR: Cache parsed vectors, don't re-parse blob

2. FETCH_NODES_BATCH IN PRUNE - MAJOR
   Location: insert.rs:178
   Issue: Uses dynamic SQL (slow path), not cached statement
   Impact: SQL parsing overhead for every prune operation
   Fix: Use cached batch fetch statement like search does

3. DOUBLE BLOB PARSING - MODERATE
   Location: insert.rs:184 and insert.rs:209
   Issue: Same blob parsed twice in prune (once for sorting, once for RNG)
   Impact: 2x parsing overhead for 384D vectors
   Fix: Parse once, store Vec<f32> instead of blob reference

4. HASHSET OVERHEAD - MODERATE
   Location: search.rs:282
   Issue: HashSet uses SipHash (cryptographic, slower)
   Impact: Hash operations in hot path
   Fix: Use rustc_hash::FxHashSet or nohash-hasher for i64 keys

5. STRING FORMATTING - MODERATE
   Location: All storage functions (format!(\"{{}}_{{}}_hnsw_nodes\", ...))
   Issue: String allocation on every storage call
   Impact: Allocation overhead in hot path
   Fix: Pre-compute table names, pass as parameters

6. BATCH SIZE 16 - MINOR
   Location: search.rs:392
   Issue: Batch fetches in chunks of 16
   Impact: Multiple queries instead of one larger batch
   Fix: Use batch size 64 to match M parameter

RECOMMENDED PRIORITY:
1. Fix RNG heuristic (biggest CPU win)
2. Use cached batch fetch in prune
3. Avoid double blob parsing
4. Switch to FxHashSet
"
    );

    println!("{}", "=".repeat(80));
}
