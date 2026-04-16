# Insert Performance Optimization Plan

## Critical Findings

### 1. Massive Heap Allocation in Hot Path

**Every distance calculation allocates 3 times and copies 1536 bytes (128D):**

```rust
// vector.rs:99-106 - Called for every node in search
pub fn from_blob(blob: &[u8], ...) -> Result<Self> {
    Ok(Vector {
        data: blob.to_vec(),  // ALLOCATION #1 + COPY
    })
}

// vector.rs:124-137 - Called TWICE per distance calc
pub fn as_f32(&self) -> Result<Vec<f32>> {
    let mut result = Vec::with_capacity(self.dimensions);  // ALLOCATION #2 or #3
    for chunk in self.data.chunks_exact(4) {
        result.push(f32::from_le_bytes(bytes));  // COPY
    }
    Ok(result)
}

// distance/scalar.rs:11-18
pub fn distance_l2_f32_scalar(a: &Vector, b: &Vector) -> Result<f32> {
    let a_vals = a.as_f32()?;  // ALLOCATES + COPIES
    let b_vals = b.as_f32()?;  // ALLOCATES + COPIES
    ...
}
```

**Impact:** With ef_construction=400, each insert explores hundreds of candidates:
- ~1000 distance calculations per insert
- ~3000 heap allocations per insert
- ~1.5MB of memory copied per insert

### 2. Uncached Batch Fetch Statements

**`fetch_nodes_batch()` rebuilds SQL dynamically each call:**

```rust
// storage.rs:383-393 - Called in every search_layer iteration
let placeholders: Vec<&str> = (0..chunk.len()).map(|_| "?").collect();
let sql = format!(
    "SELECT ... WHERE rowid IN ({})",  // STRING ALLOCATION + FORMAT
    placeholders.join(",")
);
let mut stmt = db.prepare(&sql)?;  // SQL PARSING EVERY TIME
```

### 3. Summary: Per-Insert Overhead

For a single 128D vector insert:
| Operation | Allocations | Bytes Copied | SQL Parses |
|-----------|-------------|--------------|------------|
| search_layer calls | ~3000 | ~1.5MB | ~50 |
| prune operations | ~500 | ~250KB | ~10 |
| **Total** | **~3500** | **~1.75MB** | **~60** |

---

## Optimization Plan

### Phase 1: Zero-Copy Distance Calculation (HIGHEST IMPACT)

**Goal:** Eliminate all allocations in distance calculation hot path.

#### 1.1 Add Zero-Copy Slice Access to Vector

```rust
// vector.rs - NEW METHOD
impl Vector {
    /// Get raw data as f32 slice WITHOUT allocation (zero-copy)
    ///
    /// # Safety
    /// Caller must ensure vector type is Float32 and data is properly aligned.
    /// SQLite BLOB data from column binding is always aligned.
    #[inline]
    pub fn as_f32_slice(&self) -> &[f32] {
        debug_assert!(self.vec_type == VectorType::Float32);
        // SAFETY: SQLite BLOBs are 8-byte aligned, f32 requires 4-byte alignment
        // Data was originally written as f32 values in little-endian format
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const f32,
                self.dimensions,
            )
        }
    }

    /// Get raw data as i8 slice WITHOUT allocation (zero-copy)
    #[inline]
    pub fn as_i8_slice(&self) -> &[i8] {
        debug_assert!(self.vec_type == VectorType::Int8);
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const i8,
                self.dimensions,
            )
        }
    }
}
```

**Alternative using `bytemuck` crate (safer):**
```rust
use bytemuck::cast_slice;

pub fn as_f32_slice(&self) -> &[f32] {
    cast_slice(&self.data)  // Panics if misaligned (won't happen with SQLite)
}
```

#### 1.2 Update Distance Functions to Use Slices

```rust
// distance/scalar.rs - CHANGE
pub fn distance_l2_f32_scalar(a: &Vector, b: &Vector) -> Result<f32> {
    let a_vals = a.as_f32_slice();  // NO ALLOCATION
    let b_vals = b.as_f32_slice();  // NO ALLOCATION

    let distance = f32::sqeuclidean(a_vals, b_vals)
        .ok_or_else(|| Error::InvalidParameter("L2 distance failed".into()))?;

    Ok(distance.sqrt())
}
```

#### 1.3 Avoid from_blob Copy in Hot Path

**Option A: Store reference instead of owned data**
```rust
// For search operations, use a borrowed variant
pub struct VectorRef<'a> {
    vec_type: VectorType,
    dimensions: usize,
    data: &'a [u8],  // BORROWED, no copy
}

impl<'a> VectorRef<'a> {
    #[inline]
    pub fn as_f32_slice(&self) -> &[f32] {
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const f32,
                self.dimensions,
            )
        }
    }
}
```

**Option B: Work directly with &[u8] in distance functions**
```rust
// distance/scalar.rs - Add raw byte version
#[inline]
pub fn distance_l2_f32_raw(a: &[u8], b: &[u8]) -> f32 {
    let a_vals: &[f32] = bytemuck::cast_slice(a);
    let b_vals: &[f32] = bytemuck::cast_slice(b);
    f32::sqeuclidean(a_vals, b_vals).unwrap_or(f32::MAX).sqrt()
}
```

**Estimated Impact:** 50-70% reduction in search_layer time

---

### Phase 2: Statement Caching for Batch Fetches (HIGH IMPACT)

**Goal:** Eliminate SQL parsing in `fetch_nodes_batch()`.

#### 2.1 Pre-compile Statements for Power-of-2 Batch Sizes

```rust
// hnsw/storage.rs or new file: hnsw/batch_cache.rs

/// Cached prepared statements for batch node fetches
/// Uses power-of-2 sizes: 1, 2, 4, 8, 16, 32, 64
pub struct BatchFetchCache {
    stmts: [*mut ffi::sqlite3_stmt; 7],  // For sizes 1, 2, 4, 8, 16, 32, 64
}

impl BatchFetchCache {
    pub fn new(db: &Connection, table_name: &str, column_name: &str) -> Result<Self> {
        let nodes_table = format!("{}_{}_hnsw_nodes", table_name, column_name);
        let mut stmts = [std::ptr::null_mut(); 7];

        for (i, &size) in [1, 2, 4, 8, 16, 32, 64].iter().enumerate() {
            let placeholders = (0..size).map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!(
                "SELECT rowid, level, vector FROM \"{}\" WHERE rowid IN ({})",
                nodes_table, placeholders
            );
            // Prepare statement...
            stmts[i] = prepare_stmt(db, &sql)?;
        }

        Ok(Self { stmts })
    }

    /// Get cached statement for batch size, padding with -1 if needed
    pub fn get_stmt_for_size(&self, size: usize) -> (*mut ffi::sqlite3_stmt, usize) {
        let padded_size = size.next_power_of_two().min(64);
        let idx = padded_size.trailing_zeros() as usize;
        (self.stmts[idx], padded_size)
    }
}
```

#### 2.2 Update fetch_nodes_batch to Use Cache

```rust
pub fn fetch_nodes_batch_cached(
    db: &Connection,
    table_name: &str,
    column_name: &str,
    rowids: &[i64],
    cache: Option<&BatchFetchCache>,
) -> Result<Vec<HnswNode>> {
    if let Some(cache) = cache {
        // Fast path: use cached statement
        let (stmt, padded_size) = cache.get_stmt_for_size(rowids.len());
        unsafe {
            ffi::sqlite3_reset(stmt);

            // Bind actual rowids
            for (i, &rowid) in rowids.iter().enumerate() {
                ffi::sqlite3_bind_int64(stmt, (i + 1) as i32, rowid);
            }
            // Pad with -1 (won't match any rowid)
            for i in rowids.len()..padded_size {
                ffi::sqlite3_bind_int64(stmt, (i + 1) as i32, -1);
            }

            // Execute and collect results...
        }
    } else {
        // Fallback to dynamic SQL (existing code)
    }
}
```

**Estimated Impact:** 20-30% reduction in search_layer time

---

### Phase 3: Reduce Allocations in Search Loop (MEDIUM IMPACT)

#### 3.1 Reuse Result Vectors

```rust
// search.rs - Preallocate and reuse
pub fn search_layer(...) -> Result<Vec<(i64, f32)>> {
    // Preallocate with reasonable capacity
    let mut visited = HashSet::with_capacity(ef * 2);
    let mut candidates: BinaryHeap<MinCandidate> = BinaryHeap::with_capacity(ef);
    let mut results: BinaryHeap<MaxCandidate> = BinaryHeap::with_capacity(ef + 1);

    // ... rest of function
}
```

#### 3.2 Avoid Collecting Into Intermediate Vecs

```rust
// search.rs:226-236 - CURRENT (allocates)
let unvisited_neighbors: Vec<i64> = neighbors
    .into_iter()
    .filter(|&rowid| { ... })
    .collect();

// BETTER - Use SmallVec or inline buffer
use smallvec::SmallVec;
let unvisited_neighbors: SmallVec<[i64; 64]> = neighbors
    .into_iter()
    .filter(|&rowid| { ... })
    .collect();
```

**Estimated Impact:** 5-10% reduction

---

### Phase 4: Additional Optimizations (LOWER PRIORITY)

#### 4.1 Use Arena Allocator for Temporary Data

```rust
use bumpalo::Bump;

pub fn search_layer_with_arena(
    ctx: &SearchContext,
    arena: &Bump,  // Thread-local arena
    ...
) -> Result<Vec<(i64, f32)>> {
    // Allocate temporary structures in arena
    let visited = bumpalo::collections::HashSet::new_in(arena);
    // ...
}
```

#### 4.2 Cache Query Vector f32 Slice

```rust
// In SearchContext, cache the query vector's f32 slice
pub struct SearchContext<'a> {
    pub query_vec: &'a Vector,
    pub query_f32: &'a [f32],  // Pre-computed slice, no allocation on each distance
}
```

#### 4.3 SIMD-Friendly Memory Layout

Ensure vector data is 32-byte aligned for AVX2 or 64-byte aligned for AVX-512:
```rust
#[repr(C, align(64))]
struct AlignedVectorData {
    data: [u8; MAX_VECTOR_SIZE],
}
```

---

## Implementation Order

1. **Phase 1.1 + 1.2**: Zero-copy `as_f32_slice()` + update distance functions
   - Lowest risk, highest impact
   - Keep existing `as_f32()` for backward compatibility

2. **Phase 2**: Batch fetch statement caching
   - Medium complexity, high impact
   - Should have been done originally

3. **Phase 1.3**: VectorRef for hot path
   - Higher complexity, avoids from_blob copy

4. **Phase 3**: SmallVec and capacity hints
   - Easy, moderate impact

5. **Phase 4**: Arena allocator (if still needed after 1-3)

---

## Validation

After each phase:

1. Run existing tests: `cargo test`
2. Run clippy: `cargo clippy --all-targets --all-features -- -D warnings`
3. Run benchmark: `cargo run --release --example profile_insert`
4. Run stress test: `cargo test test_multithread_long_running -- --ignored --nocapture`

**Target metrics:**
- Single-threaded: 600+ vec/sec (vs current ~300)
- Multi-threaded (16): 300+ vec/sec (vs current ~162)

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/vector.rs` | Add `as_f32_slice()`, `as_i8_slice()` |
| `src/distance/scalar.rs` | Use slices instead of `as_f32()` |
| `src/hnsw/storage.rs` | Add `BatchFetchCache`, `fetch_nodes_batch_cached()` |
| `src/hnsw/search.rs` | Use cached batch fetch, capacity hints |
| `src/hnsw/insert.rs` | Wire up batch cache through stmt_cache |
| `src/vtab.rs` | Initialize batch cache in HnswStmtCache |
| `Cargo.toml` | Add `bytemuck` dependency (optional) |

---

## Risk Assessment

| Change | Risk | Mitigation |
|--------|------|------------|
| Zero-copy cast | Alignment issues | Use bytemuck for safe casting |
| Statement cache | Memory leak | Proper Drop impl with finalize |
| SmallVec | Stack overflow if too large | Use reasonable inline size (64) |

---

## Appendix: Quick Test for Zero-Copy Impact

```rust
// Can add this test to measure impact before full implementation
#[test]
fn bench_distance_allocation() {
    let data: Vec<u8> = (0..512).map(|i| i as u8).collect();  // 128 floats

    // Current: allocates
    let start = std::time::Instant::now();
    for _ in 0..100_000 {
        let vec = Vector::from_blob(&data, VectorType::Float32, 128).unwrap();
        let _ = vec.as_f32().unwrap();
    }
    let alloc_time = start.elapsed();

    // Zero-copy (simulated)
    let start = std::time::Instant::now();
    for _ in 0..100_000 {
        let slice: &[f32] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const f32, 128)
        };
        let _ = slice[0];  // Force use
    }
    let zerocopy_time = start.elapsed();

    println!("Allocating: {:?}", alloc_time);
    println!("Zero-copy:  {:?}", zerocopy_time);
    println!("Speedup:    {:.1}x", alloc_time.as_nanos() as f64 / zerocopy_time.as_nanos() as f64);
}
```
