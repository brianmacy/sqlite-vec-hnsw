// Detailed timing profile of HNSW insert to find bottlenecks
use rusqlite::Connection;
use std::time::Instant;

#[test]
fn profile_insert_timing() {
    println!("\n=== Detailed Insert Timing Profile ===");

    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute(
        "CREATE VIRTUAL TABLE test USING vec0(embedding float[128])",
        [],
    )
    .unwrap();

    // Warm up with a few inserts first
    println!("Warming up with 10 vectors...");
    for i in 1..=10 {
        let vec: Vec<f32> = (0..128).map(|j| ((i + j) as f32 / 100.0).sin()).collect();
        let bytes: Vec<u8> = vec.iter().flat_map(|f| f.to_le_bytes()).collect();
        db.execute(
            "INSERT INTO test(rowid, embedding) VALUES (?, ?)",
            rusqlite::params![i, bytes],
        )
        .unwrap();
    }

    // Profile inserts at different graph sizes
    println!("\nProfiling inserts at different graph sizes...");

    db.execute("BEGIN", []).unwrap();

    // Profile early inserts (11-110)
    let mut early_total = 0u128;
    for i in 11..=110 {
        let vec: Vec<f32> = (0..128).map(|j| ((i + j) as f32 / 100.0).sin()).collect();
        let bytes: Vec<u8> = vec.iter().flat_map(|f| f.to_le_bytes()).collect();

        let start = Instant::now();
        db.execute(
            "INSERT INTO test(rowid, embedding) VALUES (?, ?)",
            rusqlite::params![i, bytes],
        )
        .unwrap();
        early_total += start.elapsed().as_micros();
    }

    // Continue to 500 without timing
    for i in 111..=500 {
        let vec: Vec<f32> = (0..128).map(|j| ((i + j) as f32 / 100.0).sin()).collect();
        let bytes: Vec<u8> = vec.iter().flat_map(|f| f.to_le_bytes()).collect();
        db.execute(
            "INSERT INTO test(rowid, embedding) VALUES (?, ?)",
            rusqlite::params![i, bytes],
        )
        .unwrap();
    }

    // Profile mid inserts (501-600)
    let mut mid_total = 0u128;
    for i in 501..=600 {
        let vec: Vec<f32> = (0..128).map(|j| ((i + j) as f32 / 100.0).sin()).collect();
        let bytes: Vec<u8> = vec.iter().flat_map(|f| f.to_le_bytes()).collect();

        let start = Instant::now();
        db.execute(
            "INSERT INTO test(rowid, embedding) VALUES (?, ?)",
            rusqlite::params![i, bytes],
        )
        .unwrap();
        mid_total += start.elapsed().as_micros();
    }

    // Continue to 900
    for i in 601..=900 {
        let vec: Vec<f32> = (0..128).map(|j| ((i + j) as f32 / 100.0).sin()).collect();
        let bytes: Vec<u8> = vec.iter().flat_map(|f| f.to_le_bytes()).collect();
        db.execute(
            "INSERT INTO test(rowid, embedding) VALUES (?, ?)",
            rusqlite::params![i, bytes],
        )
        .unwrap();
    }

    // Profile late inserts (901-1000)
    let mut late_total = 0u128;
    for i in 901..=1000 {
        let vec: Vec<f32> = (0..128).map(|j| ((i + j) as f32 / 100.0).sin()).collect();
        let bytes: Vec<u8> = vec.iter().flat_map(|f| f.to_le_bytes()).collect();

        let start = Instant::now();
        db.execute(
            "INSERT INTO test(rowid, embedding) VALUES (?, ?)",
            rusqlite::params![i, bytes],
        )
        .unwrap();
        late_total += start.elapsed().as_micros();
    }

    db.execute("COMMIT", []).unwrap();

    let avg_early = early_total / 100;
    let avg_mid = mid_total / 100;
    let avg_late = late_total / 100;

    println!("\nResults by graph size:");
    println!(
        "  Early  (11-110):   {}μs ({:.2}ms)",
        avg_early,
        avg_early as f64 / 1000.0
    );
    println!(
        "  Mid    (501-600):  {}μs ({:.2}ms)",
        avg_mid,
        avg_mid as f64 / 1000.0
    );
    println!(
        "  Late   (901-1000): {}μs ({:.2}ms)",
        avg_late,
        avg_late as f64 / 1000.0
    );

    if avg_late > avg_early {
        println!("\n⚠️  Performance DEGRADES as graph grows!");
        println!(
            "  Degradation: {:.2}x slower at 1000 nodes vs 100 nodes",
            avg_late as f64 / avg_early as f64
        );
    }

    println!("\nC performance (constant): 2.43ms/vector");
    println!(
        "Rust early ratio: {:.2}x slower",
        avg_early as f64 / 1000.0 / 2.43
    );
    println!(
        "Rust late ratio:  {:.2}x slower",
        avg_late as f64 / 1000.0 / 2.43
    );

    // Also test search performance
    println!("\n=== Search Performance ===");

    let search_vec: Vec<f32> = (0..128).map(|j| (j as f32 / 100.0).sin()).collect();
    let search_bytes: Vec<u8> = search_vec.iter().flat_map(|f| f.to_le_bytes()).collect();

    let search_start = Instant::now();
    let mut stmt = db
        .prepare("SELECT rowid FROM test WHERE embedding MATCH ? AND k = 10 ORDER BY distance")
        .unwrap();
    let results: Vec<i64> = stmt
        .query_map([search_bytes], |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    let search_time = search_start.elapsed();

    println!(
        "Search returned {} results in {}μs ({:.2}ms)",
        results.len(),
        search_time.as_micros(),
        search_time.as_micros() as f64 / 1000.0
    );
    println!("C search: 1.35ms");
    println!(
        "Rust ratio: {:.2}x slower",
        search_time.as_micros() as f64 / 1000.0 / 1.35
    );
}
