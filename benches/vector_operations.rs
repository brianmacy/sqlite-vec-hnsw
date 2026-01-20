use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use rusqlite::Connection;

fn setup_db_with_vectors(num_vectors: usize, dimensions: usize) -> Connection {
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
        let vector: Vec<f32> = (0..dimensions)
            .map(|j| (i * dimensions + j) as f32 / 1000.0)
            .collect();
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

    db
}

fn bench_vector_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_insertion");

    for dimensions in [128, 384, 768].iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("insert_with_hnsw", dimensions),
            dimensions,
            |b, &dims| {
                b.iter_batched(
                    || {
                        let db = Connection::open_in_memory().unwrap();
                        sqlite_vec_hnsw::init(&db).unwrap();
                        db.execute(
                            &format!(
                                "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[{}])",
                                dims
                            ),
                            [],
                        )
                        .unwrap();
                        db
                    },
                    |db| {
                        let vector: Vec<f32> = (0..dims).map(|i| i as f32 / 1000.0).collect();
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
                                "INSERT INTO vectors(rowid, embedding) VALUES (1, vec_f32('{}'))",
                                vector_json
                            ),
                            [],
                        )
                        .unwrap();
                        black_box(db);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_knn_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_query");

    for num_vectors in [100, 1000, 10000].iter() {
        let db = setup_db_with_vectors(*num_vectors, 128);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("hnsw_search", num_vectors),
            &db,
            |b, db| {
                let query_vector = vec![0.5f32; 128];
                let query_json = format!(
                    "[{}]",
                    query_vector
                        .iter()
                        .map(|v| v.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                );

                b.iter(|| {
                    let mut stmt = db
                        .prepare(&format!(
                            "SELECT rowid, distance FROM vectors WHERE embedding MATCH vec_f32('{}') AND k = 10 ORDER BY distance",
                            query_json
                        ))
                        .unwrap();

                    let results: Vec<(i64, f64)> = stmt
                        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
                        .unwrap()
                        .collect::<rusqlite::Result<Vec<_>>>()
                        .unwrap();

                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

fn bench_distance_calculations(c: &mut Criterion) {
    use sqlite_vec_hnsw::distance::{DistanceMetric, distance};
    use sqlite_vec_hnsw::vector::Vector;

    let mut group = c.benchmark_group("distance_calculations");

    for dimensions in [128, 384, 768].iter() {
        let vec1 = Vector::from_f32(&vec![1.0f32; *dimensions]);
        let vec2 = Vector::from_f32(&vec![2.0f32; *dimensions]);

        group.throughput(Throughput::Elements(*dimensions as u64));

        group.bench_with_input(
            BenchmarkId::new("l2_distance", dimensions),
            &(vec1.clone(), vec2.clone()),
            |b, (v1, v2)| {
                b.iter(|| {
                    let dist = distance(v1, v2, DistanceMetric::L2).unwrap();
                    black_box(dist);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cosine_distance", dimensions),
            &(vec1.clone(), vec2.clone()),
            |b, (v1, v2)| {
                b.iter(|| {
                    let dist = distance(v1, v2, DistanceMetric::Cosine).unwrap();
                    black_box(dist);
                });
            },
        );
    }

    group.finish();
}

fn bench_batch_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_insert");
    group.sample_size(10); // Reduce samples for long-running benchmarks

    for batch_size in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("insert_batch", batch_size),
            batch_size,
            |b, &size| {
                b.iter_batched(
                    || {
                        let db = Connection::open_in_memory().unwrap();
                        sqlite_vec_hnsw::init(&db).unwrap();
                        db.execute(
                            "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[128])",
                            [],
                        )
                        .unwrap();
                        db
                    },
                    |db| {
                        for i in 0..size {
                            let vector: Vec<f32> = (0..128).map(|j| (i * 128 + j) as f32 / 1000.0).collect();
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
                        black_box(db);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_vector_insertion,
    bench_knn_query,
    bench_distance_calculations,
    bench_batch_insert
);
criterion_main!(benches);
