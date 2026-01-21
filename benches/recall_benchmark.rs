// Benchmark HNSW recall test performance
// Same parameters as C test: 128D, 1000 vectors, k=10
use criterion::{Criterion, criterion_group, criterion_main};
use rusqlite::Connection;

fn benchmark_hnsw_insert_1000(c: &mut Criterion) {
    const DIMENSIONS: usize = 128;
    const NUM_VECTORS: i64 = 1000;

    c.bench_function("hnsw_insert_1000_vectors", |b| {
        b.iter(|| {
            let db = Connection::open_in_memory().unwrap();
            sqlite_vec_hnsw::init(&db).unwrap();

            db.execute(
                "CREATE VIRTUAL TABLE test_hnsw USING vec0(embedding float[128])",
                [],
            )
            .unwrap();

            db.execute("BEGIN", []).unwrap();
            for i in 0..NUM_VECTORS {
                let vec: Vec<f32> = (0..DIMENSIONS)
                    .map(|j| (i * 100 + j as i64) as f32 / 1000.0)
                    .collect();
                let bytes: Vec<u8> = vec.iter().flat_map(|f| f.to_le_bytes()).collect();
                let rowid = i + 1;
                db.execute(
                    "INSERT INTO test_hnsw(rowid, embedding) VALUES (?, ?)",
                    rusqlite::params![rowid, bytes],
                )
                .unwrap();
            }
            db.execute("COMMIT", []).unwrap();
        })
    });
}

fn benchmark_hnsw_search(c: &mut Criterion) {
    const DIMENSIONS: usize = 128;
    const NUM_VECTORS: i64 = 1000;
    const K: i64 = 10;

    // Setup: create and populate the database once
    let db = Connection::open_in_memory().unwrap();
    sqlite_vec_hnsw::init(&db).unwrap();

    db.execute(
        "CREATE VIRTUAL TABLE test_hnsw USING vec0(embedding float[128])",
        [],
    )
    .unwrap();

    db.execute("BEGIN", []).unwrap();
    for i in 0..NUM_VECTORS {
        let vec: Vec<f32> = (0..DIMENSIONS)
            .map(|j| (i * 100 + j as i64) as f32 / 1000.0)
            .collect();
        let bytes: Vec<u8> = vec.iter().flat_map(|f| f.to_le_bytes()).collect();
        let rowid = i + 1;
        db.execute(
            "INSERT INTO test_hnsw(rowid, embedding) VALUES (?, ?)",
            rusqlite::params![rowid, bytes],
        )
        .unwrap();
    }
    db.execute("COMMIT", []).unwrap();

    // Query vector
    let query_vec: Vec<f32> = vec![0.5; DIMENSIONS];
    let query_bytes: Vec<u8> = query_vec.iter().flat_map(|f| f.to_le_bytes()).collect();

    c.bench_function("hnsw_search_k10", |b| {
        b.iter(|| {
            let mut stmt = db
                .prepare("SELECT rowid FROM test_hnsw WHERE embedding MATCH ? AND k = ? ORDER BY distance")
                .unwrap();
            let results: Vec<i64> = stmt
                .query_map(rusqlite::params![query_bytes.clone(), K], |row| row.get(0))
                .unwrap()
                .collect::<Result<_, _>>()
                .unwrap();
            assert_eq!(results.len(), K as usize);
        })
    });
}

criterion_group!(benches, benchmark_hnsw_insert_1000, benchmark_hnsw_search);
criterion_main!(benches);
