# Compatibility & Performance Investigation Findings

## Executive Summary

**Compatibility Status:** ✅ **SCHEMA COMPATIBLE**
- All shadow tables match C schema
- Rust can read C-created databases
- C can read Rust-created databases
- Int8 quantization fully functional

**Storage Status:** ❌ **27% BLOAT (NOT ACCEPTABLE)**
- Root cause: Simple closest-N pruning creates dense graphs
- Rust: ALL nodes have 64 edges (max capacity)
- C: Natural distribution 11-64 edges (avg 32.2)
- Fix required: Implement RNG heuristic pruning

**Performance Status:** ❌ **6-7x SLOWER THAN C**
- Primary cause: Missing prepared statement caching
- Secondary cause: Dense graph (2x more edges to traverse)
- Fix required: Implement HnswStatementCache + RNG heuristic

---

## Cross-Compatibility Testing

### ✅ Rust Reads C Database
- Opened C-created database successfully
- Read shadow tables (chunks, rowids, vector_chunks, info)
- Read vectors from BLOB storage
- All data intact and readable

### ✅ C Reads Rust Database
- Opened Rust-created database successfully
- Read shadow tables (all 9 tables)
- Read vectors from BLOB storage using C's BLOB API
- HNSW index readable (50 nodes verified)
- All data intact and readable

**Conclusion:** Binary format 100% compatible

---

## Storage Analysis

### Initial Confusion: "4-9x Bloat"

**Problem:** C's benchmark reports:
```
Storage: 70.3 MB for 24K float32 vectors = 2,929 bytes/vector
```

**Reality:** This is ONLY HNSW node storage, calculated as:
```cpp
result.storage_bytes = SUM(length(vector)) FROM hnsw_nodes
```

**Missing from C's metric:**
- vector_chunks tables (~75 MB)
- hnsw_edges tables (~60 MB with M=64)
- Other shadow tables (~5 MB)
- **C's actual full database: ~210 MB = ~8,750 bytes/vector**

### CORRECTED Storage Comparison (1000 vectors, M=32, ef=400)

| Table | C (bytes) | Rust (bytes) | Difference |
|-------|-----------|--------------|------------|
| chunks | 12,288 | 12,288 | ✓ Match |
| **hnsw_edges** | **765,952** | **2,236,416** | ❌ **+192% bloat** |
| **hnsw_levels** | 4,096 | 16,384 | ❌ **+300% bloat** |
| hnsw_meta | 4,096 | 4,096 | ✓ Match |
| hnsw_nodes | 4,108,288 | 4,108,288 | ✓ Match |
| info | 4,096 | 4,096 | ✓ Match |
| rowids | 16,384 | 16,384 | ✓ Match |
| vector_chunks00 | 3,149,824 | 3,149,824 | ✓ Match |
| **TOTAL** | **9,633,792** | **12,328,960** | ❌ **+27% bloat** |

**Per vector:** C = 9,634 bytes/vec, Rust = 12,329 bytes/vec

**Root cause: Dense graph from simple pruning**

| Metric | C | Rust |
|--------|---|------|
| Total edges | 32,235 | 65,088 |
| Avg edges/node (L0) | 32.2 | 64.0 |
| Min edges (L0) | 11 | 64 |
| Max edges (L0) | 64 | 64 |
| Edge distribution | Natural (11-64) | **ALL nodes = 64** |

**Rust creates 2x more edges because:**
- Simple closest-N pruning → dense, uniform graph
- Every node connects to same popular neighbors
- All nodes reach maximum capacity (64 edges)

**C creates fewer edges because:**
- RNG heuristic → sparse, diverse graph
- Promotes diversity in neighbor selection
- Natural distribution (11-64 edges, avg 32)

**Impact:**
- Storage: +1.47 MB in edges table (+192%)
- Performance: More edges to traverse during search
- Quality: Possibly lower recall (dense graph ≠ small-world property)

---

## Performance Analysis

### Measured Performance (In-Memory + Transactions)

| Implementation | Float32 (768D) | Int8 (768D) |
|----------------|----------------|-------------|
| C (M=64, ef=200) | 162 vec/sec | 184 vec/sec |
| Rust (M=32, ef=400) | 21.9 vec/sec | 26.4 vec/sec |
| **Gap** | **7.4x slower** | **7.0x slower** |

### Bottleneck Identified: Statement Preparation Overhead

**C Implementation:**
```c
struct HnswStatementCache {
    sqlite3_stmt *get_node_data;
    sqlite3_stmt *get_edges;
    sqlite3_stmt *insert_node;
    sqlite3_stmt *insert_edge;
    sqlite3_stmt *delete_edges_from;
    // ... 10+ prepared statements
};

// Prepared ONCE per connection
hnsw_prepare_statements(db, table, column, &stmts);

// Reused hundreds of times
hnsw_fetch_node_data(stmts, rowid, ...);  // Uses stmts->get_node_data
```

**Rust Implementation:**
```rust
// Called EVERY time
pub fn fetch_node_data(...) {
    let query = format!("SELECT ... FROM hnsw_nodes WHERE ...");
    let mut stmt = db.prepare(&query)?;  // ❌ Parse SQL every call!
    stmt.query_row(...)
}
```

**Impact:**
- With ef_construction=400, each insert searches ~400 nodes
- Each search fetches neighbors (~32 edges with M=32)
- Each fetch calls `db.prepare()` → parses SQL from scratch
- **Thousands of SQL parses per insert!**

### What Was Tested (All Had Minimal Impact):

✅ Transaction batching (BEGIN...COMMIT) - Same speed
✅ Prepared statements at SQL level - Same speed
✅ In-memory vs disk - Same speed
✅ WAL mode - Same speed
✅ PRAGMA synchronous modes - Same speed
✅ Binary vectors vs JSON - Same speed

**None of these helped** because the bottleneck is INSIDE the virtual table implementation, not at the SQL layer.

---

## Bugs Fixed

### 1. Missing `_info` Shadow Table
**Before:** Rust didn't create `{table}_info` table
**After:** Now creates and populates with version metadata
**Impact:** C compatibility

### 2. Wrong Default Chunk Size
**Before:** DEFAULT_CHUNK_SIZE = 256
**After:** DEFAULT_CHUNK_SIZE = 1024
**Impact:** Storage efficiency (4x improvement at scale)

### 3. Incorrect Edge Pruning
**Before:** Set all edge distances to 0.0 during neighbor pruning
**After:** Fetch stored distances from edges table
**Impact:** Correctness (graph quality), no performance change

---

## Int8 Quantization Status

### ✅ Fully Functional
- `vec_int8()` function works
- `vec_quantize_int8()` converts float32 → int8
- HNSW indexing works with int8 vectors
- KNN search returns correct results

### Storage Savings
- Float32: 2.5 MB (200 vectors)
- Int8: 1.3 MB (200 vectors)
- **Ratio: 1.88x smaller**

### Performance
- Float32: 27.0 vec/sec (200 vectors, in-memory)
- Int8: 29.5 vec/sec (200 vectors, in-memory)
- **Ratio: 1.09x faster** (minimal, as expected)

---

## Path to Performance Parity

### Required: Implement Prepared Statement Caching

**Architecture:**
```rust
struct HnswStatementCache<'conn> {
    get_node_data: Statement<'conn>,
    get_node_level: Statement<'conn>,
    get_edges: Statement<'conn>,
    get_edges_with_dist: Statement<'conn>,
    insert_node: Statement<'conn>,
    insert_edge: Statement<'conn>,
    delete_edges_from: Statement<'conn>,
    get_entry_point: Statement<'conn>,
    get_meta_value: Statement<'conn>,
    update_meta: Statement<'conn>,
}

impl HnswStatementCache<'_> {
    fn new(db: &Connection, table: &str, column: &str) -> Result<Self> {
        // Prepare all statements once
    }
}
```

**Expected improvement:** 5-10x faster (would match or exceed C's 162 vec/sec)

**Challenges:**
- Statement lifetimes tied to Connection
- Need per-connection caching (not per-table)
- Must integrate with virtual table lifetime management

**Effort:** Medium (2-3 hours of focused work)

---

## Test Coverage Added

1. `test_disk_persistence.rs` - Cross-connection persistence
2. `test_int8_quantization.rs` - Int8 functionality and performance
3. `test_transaction_batching.rs` - Transaction overhead testing
4. `test_inmemory_batching.rs` - In-memory performance
5. `test_prepared_statements.rs` - Statement preparation overhead
6. `test_storage_format.rs` - Storage breakdown analysis
7. `test_chunk_fill_efficiency.rs` - Chunk allocation efficiency
8. `test_c_compatibility.rs` - Schema validation
9. `test_read_c_database.rs` - Cross-compatibility (Rust reads C)
10. `examples/create_test_db.rs` - Create Rust DB for C testing
11. `/tmp/read_rust_db.cpp` - Cross-compatibility (C reads Rust)
12. `/tmp/create_c_testdb.cpp` - Create C DB for Rust testing

---

## Recommendations

### Immediate (to reach performance parity):
1. **Implement HnswStatementCache** - Pre-prepare all statements
2. **Test at scale** - Verify performance with 10K+ vectors
3. **Benchmark with M=64** - Match C's configuration

### Future (nice to have):
4. Reduce 1.4x storage overhead
5. Implement RNG heuristic pruning (instead of simple closest-N)
6. Add prepared statement caching for vector_chunks operations too

---

## Date
January 20, 2026

---

## CRITICAL BUGS FOUND (Must Fix for Parity)

### 1. Dense Graph from Simple Pruning (STORAGE BLOAT)

**Bug:** Rust uses simple closest-N edge selection
**Impact:** 
- 2x more edges than C (65,088 vs 32,235)
- ALL nodes have max edges (64), none have natural distribution
- +27% storage bloat (+2.6 MB for 1000 vectors)

**Evidence:**
```
C edge distribution (level 0):  11-64 edges, avg 32.2, natural curve
Rust edge distribution (level 0): ALL 1000 nodes have exactly 64 edges
```

**Fix Required:** Implement RNG (Relative Neighborhood Graph) heuristic pruning
- Source: sqlite-vec.c lines 13048-13152 (`hnsw_select_neighbors_by_heuristic`)
- Algorithm: HNSWlib's `getNeighborsByHeuristic2()`
- Promotes diversity: only keep neighbor if closer to center than to existing selected neighbors
- Expected result: Natural edge distribution matching C

### 2. Missing Prepared Statement Cache (PERFORMANCE)

**Bug:** Every DB operation parses SQL from scratch
**Impact:** 6-7x slower than C (23 vs 162 vec/sec)

**Fix Required:** Implement HnswStatementCache (like C lines 11889-11964)
- Pre-prepare all statements once per connection
- Reuse for all operations (hundreds per insert)
- Expected improvement: 5-10x faster

### 3. Syntax Incompatibility (USABILITY)

**Bug:** Rust doesn't parse C's syntax for HNSW parameters

**C syntax:**
```sql
CREATE VIRTUAL TABLE t USING szvec(v float[768] hnsw(M=32, ef_construction=400))
```

**Rust syntax:**
```sql
CREATE VIRTUAL TABLE t USING vec0(v float[768], type=hnsw)
-- M and ef_construction are hardcoded defaults, not configurable
```

**Fix Required:** Parse `hnsw(...)` clause in column definition
- Extract M, ef_construction, ef_search, distance metric
- Store per-column configuration
- Match C's syntax exactly

---

## Summary: Path to Full Parity

| Issue | Status | Priority | Effort | Expected Gain |
|-------|--------|----------|--------|---------------|
| RNG heuristic pruning | ❌ Required | **CRITICAL** | Medium (4-6h) | -27% storage, better recall |
| Prepared statement cache | ❌ Required | **CRITICAL** | Medium (3-4h) | 5-10x performance |
| Syntax compatibility | ❌ Required | High | Small (1-2h) | User experience |
| Schema compatibility | ✅ Done | - | - | - |
| Int8 support | ✅ Done | - | - | - |
| Cross-reading | ✅ Done | - | - | - |

**Estimated total effort:** 8-12 hours of focused work
**Expected outcome:** Full parity with C (storage, performance, syntax, compatibility)

**Current state:** Schema-compatible but with algorithmic differences that cause storage bloat and performance degradation.

