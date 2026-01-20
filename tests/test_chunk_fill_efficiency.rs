// Test storage efficiency as chunks fill up
use rusqlite::Connection;
use tempfile::TempDir;

#[test]
fn test_storage_efficiency_at_scale() {
    println!("\n=== Storage Efficiency vs Chunk Fill ===");
    println!("chunk_size=1024, each vector=768 float32 (3,072 bytes)\n");

    for &num_vectors in &[100, 500, 1000, 2000] {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join(format!("test_{}.db", num_vectors));

        let db = Connection::open(&db_path).unwrap();
        sqlite_vec_hnsw::init(&db).unwrap();

        db.execute("CREATE VIRTUAL TABLE v USING vec0(e float[768])", [])
            .unwrap();
        db.execute("BEGIN", []).unwrap();

        for i in 1..=num_vectors {
            let vec: Vec<f32> = (0..768).map(|j| ((i + j) % 100) as f32 / 100.0).collect();
            let bytes: Vec<u8> = vec.iter().flat_map(|f| f.to_le_bytes()).collect();
            db.execute(
                "INSERT INTO v(rowid, e) VALUES (?, ?)",
                rusqlite::params![i, bytes],
            )
            .unwrap();
        }

        db.execute("COMMIT", []).unwrap();

        let size = std::fs::metadata(&db_path).unwrap().len();
        let per_vec = size / num_vectors as u64;
        let num_chunks = (num_vectors + 1023) / 1024;
        let chunk_fill = (num_vectors % 1024) as f64 / 1024.0 * 100.0;

        println!(
            "{:5} vectors: {:8} bytes total, {:6} bytes/vec ({} chunks, {:.0}% fill)",
            num_vectors,
            size,
            per_vec,
            num_chunks,
            if num_vectors % 1024 == 0 {
                100.0
            } else {
                chunk_fill
            }
        );
    }

    println!("\n=== Analysis ===");
    println!("Expected pattern:");
    println!("- 100 vectors (1 chunk, 10% fill): HIGH cost/vector due to chunk allocation");
    println!("- 1000 vectors (1 chunk, 98% fill): LOW cost/vector, chunk almost full");
    println!("- 2000 vectors (2 chunks, 95% fill): Cost should stabilize");
    println!("\nC's benchmark uses 24,000 vectors (23 chunks), so chunks are efficiently filled!");
}
