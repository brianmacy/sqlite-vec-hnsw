# Comprehensive Fix Plan: Match C Implementation Performance

## Executive Summary

The Rust implementation has **9 critical deviations** from the proven C implementation. The analysis shows:

1. **Code doesn't compile** - 5 missing functions
2. **#1 Performance Bug**: `fetch_neighbors()` prepares statement on every call (5-10x slowdown)
3. **#2 Performance Bug**: No `sqlite3_reset()` after reads (holds WAL locks)
4. **Correctness Issues**: Missing multi-connection validation, incorrect algorithms

**Expected outcome**: Match C performance (<1ms/vector) by following C patterns exactly.

---

## Phase 1: Make Code Compile (Critical Priority)

### Issue 1.1: Missing `fetch_neighbors_cached()` Function

**Problem**: Search code calls non-existent function
**File**: `src/hnsw/storage.rs`
**Impact**: Code doesn't compile

**Solution**: Rename existing `fetch_neighbors()` to `fetch_neighbors_cached()` and add cached statement parameter.

**Implementation**:
```rust
// Change signature from:
pub fn fetch_neighbors(
    db: &Connection,
    table_name: &str,
    column_name: &str,
    from_rowid: i64,
    level: i32,
) -> Result<Vec<i64>>

// To:
pub fn fetch_neighbors_cached(
    db: &Connection,
    table_name: &str,
    column_name: &str,
    from_rowid: i64,
    level: i32,
    cached_stmt: Option<*mut ffi::sqlite3_stmt>,
) -> Result<Vec<i64>>

// Implementation:
pub fn fetch_neighbors_cached(...) -> Result<Vec<i64>> {
    // Fast path: use cached statement (like C implementation)
    if let Some(stmt) = cached_stmt {
        unsafe {
            ffi::sqlite3_reset(stmt);
            ffi::sqlite3_bind_int64(stmt, 1, from_rowid);
            ffi::sqlite3_bind_int(stmt, 2, level);

            let mut neighbors = Vec::new();
            loop {
                let rc = ffi::sqlite3_step(stmt);
                if rc == ffi::SQLITE_ROW {
                    let to_rowid = ffi::sqlite3_column_int64(stmt, 0);
                    neighbors.push(to_rowid);
                } else if rc == ffi::SQLITE_DONE {
                    break;
                } else {
                    return Err(Error::Sqlite(rusqlite::Error::SqliteFailure(
                        rusqlite::ffi::Error::new(rc),
                        None,
                    )));
                }
            }

            // CRITICAL: Reset immediately to release WAL lock
            ffi::sqlite3_reset(stmt);
            return Ok(neighbors);
        }
    }

    // Slow path fallback: prepare statement each time (only for testing/fallback)
    let edges_table = format!("{}_{}_hnsw_edges", table_name, column_name);
    let query = format!(
        "SELECT to_rowid FROM \"{}\" WHERE from_rowid = ? AND level = ? ORDER BY distance",
        edges_table
    );

    let mut stmt = db.prepare(&query).map_err(Error::Sqlite)?;
    let neighbors = stmt
        .query_map([from_rowid, level as i64], |row| row.get::<_, i64>(0))
        .map_err(Error::Sqlite)?
        .collect::<std::result::Result<Vec<i64>, _>>()
        .map_err(Error::Sqlite)?;

    Ok(neighbors)
}
```

**Files to modify**:
- `src/hnsw/storage.rs`: Implement function
- `src/hnsw/search.rs`: Already calls this (no change needed)
- `src/hnsw/insert.rs`: Update calls in `find_closest_at_level()` to pass cached statement

---

### Issue 1.2: Missing Batch Insert/Delete Functions

**Problem**: Code calls `insert_edges_batch()` and `delete_edges_batch()` which don't exist
**File**: `src/hnsw/storage.rs`, `src/hnsw/insert.rs`
**Impact**: Code doesn't compile

**Decision Point**: Batch operations OR single-row operations?

**Recommendation**: **Use single-row operations like C** (proven pattern)

**Rationale**:
- C uses single-row with immediate reset (proven to work)
- Batching requires holding statement open longer (potential lock issues)
- Simpler to implement and debug
- Can optimize later if needed

**Solution A: Replace batch calls with single-row operations**

In `src/hnsw/insert.rs`:

```rust
// REMOVE THIS (lines 210-221):
let edges: Vec<(i64, i64, i32, f32)> = selected
    .iter()
    .flat_map(...)
    .collect();
storage::insert_edges_batch(db, table_name, column_name, &edges)?;

// REPLACE WITH (like C):
let insert_edge_stmt = stmt_cache.map(|c| c.insert_edge);

for (neighbor_rowid, dist) in selected.iter() {
    // Edge: new_node -> neighbor
    storage::insert_edge(db, table_name, column_name,
                        rowid, *neighbor_rowid, lv, *dist,
                        insert_edge_stmt)?;

    // Edge: neighbor -> new_node (bidirectional)
    storage::insert_edge(db, table_name, column_name,
                        *neighbor_rowid, rowid, lv, *dist,
                        insert_edge_stmt)?;
}
```

In `src/hnsw/insert.rs` prune function:

```rust
// REMOVE THIS (line 88):
storage::delete_edges_batch(db, table_name, column_name, neighbor_rowid, level, &edges_to_remove)?;

// REPLACE WITH (like C - delete ALL, then re-insert):
let delete_stmt = stmt_cache.map(|c| c.delete_edges_from);
let insert_stmt = stmt_cache.map(|c| c.insert_edge);

// Delete all edges at this level
storage::delete_edges_from_level(db, table_name, column_name,
                                  neighbor_rowid, level, delete_stmt)?;

// Re-insert the edges we're keeping (first max_connections from sorted list)
for (to_rowid, dist) in sorted_edges.iter().take(max_connections) {
    storage::insert_edge(db, table_name, column_name,
                        neighbor_rowid, *to_rowid, level, *dist,
                        insert_stmt)?;
}
```

**Files to modify**:
- `src/hnsw/insert.rs`: Replace batch calls with loops

---

### Issue 1.3: Fix HnswStmtCache Missing Field

**Problem**: `HnswStmtCache` missing `get_edges` field
**File**: `src/hnsw/insert.rs`, `src/vtab.rs`
**Impact**: Code doesn't compile

**Solution**: `get_edges` field already exists (line 98), but vtab.rs isn't passing it.

In `src/vtab.rs` around lines 903 and 1081:

```rust
Some(hnsw::insert::HnswStmtCache {
    get_node_data: ...,
    get_edges: get_edges_stmt_ptr,          // ADD THIS LINE
    get_edges_with_dist: ...,
    insert_node: ...,
    insert_edge: ...,
    delete_edges_from: ...,
})
```

**Files to modify**:
- `src/vtab.rs`: Add `get_edges` field to struct initialization

---

### Issue 1.4: Fix `save_dynamic_to_db()` Function Name

**Problem**: Function doesn't exist
**File**: `src/hnsw/insert.rs` lines 145, 260
**Impact**: Code doesn't compile

**Solution**: Use existing `save_to_db()` or create lighter version.

**Better approach** (match C): Save only changed fields.

Add to `src/hnsw/mod.rs`:

```rust
impl HnswMetadata {
    /// Save only dynamic fields (entry_point, num_nodes, version)
    /// This matches C implementation which updates fields individually
    pub fn save_dynamic_to_db(&self, db: &Connection, table_name: &str, column_name: &str) -> Result<()> {
        let meta_table = format!("{}_{}_hnsw_meta", table_name, column_name);
        let update_sql = format!(
            "INSERT OR REPLACE INTO \"{}\" (key, value) VALUES (?, ?)",
            meta_table
        );

        // Only update fields that change during insert/delete
        db.execute(&update_sql, ["entry_point_rowid", &self.entry_point_rowid.to_string()])?;
        db.execute(&update_sql, ["entry_point_level", &self.entry_point_level.to_string()])?;
        db.execute(&update_sql, ["num_nodes", &self.num_nodes.to_string()])?;
        db.execute(&update_sql, ["hnsw_version", &self.hnsw_version.to_string()])?;

        Ok(())
    }
}
```

**Files to modify**:
- `src/hnsw/mod.rs`: Add `save_dynamic_to_db()` method

---

### Issue 1.5: Fix Cache Module Syntax Errors

**Problem**: `retain()` closure syntax error
**File**: `src/hnsw/cache.rs` lines 174-175
**Impact**: Code doesn't compile

**Solution**: Fix closure parameter syntax.

```rust
// CHANGE FROM:
self.neighbors.retain(|(r, _)| *r != rowid);
self.neighbors_with_dist.retain(|(r, _)| *r != rowid);

// TO:
self.neighbors.retain(|&(r, _), _| r != rowid);
self.neighbors_with_dist.retain(|&(r, _), _| r != rowid);
```

**Files to modify**:
- `src/hnsw/cache.rs`: Fix closure syntax

---

## Phase 2: Fix Critical Performance Bugs

### Issue 2.1: Add sqlite3_reset() After All Reads

**Problem**: Statements not reset after reading, holding WAL locks
**File**: `src/hnsw/storage.rs` multiple functions
**Impact**: 2-5x performance degradation, lock contention

**Solution**: Add `ffi::sqlite3_reset(stmt)` before EVERY return in cached statement paths.

**Pattern to follow**:
```rust
unsafe {
    ffi::sqlite3_reset(stmt);
    ffi::sqlite3_bind_XXX(stmt, ...);

    let rc = ffi::sqlite3_step(stmt);
    if rc == ffi::SQLITE_ROW {
        // Copy all data to owned structures FIRST
        let data = copy_data_from_statement(stmt);

        // THEN reset immediately
        ffi::sqlite3_reset(stmt);

        // THEN return
        return Ok(Some(data));
    } else if rc == ffi::SQLITE_DONE {
        ffi::sqlite3_reset(stmt);  // Reset before return
        return Ok(None);
    } else {
        ffi::sqlite3_reset(stmt);  // Reset even on error
        return Err(...);
    }
}
```

**Functions to fix** in `src/hnsw/storage.rs`:
1. `fetch_node_data()` (lines 36-64) - Add reset before returns
2. `fetch_neighbors_cached()` (after implementing) - Add reset after loop
3. `fetch_neighbors_with_distances()` (lines 143-166) - Add reset before return
4. `insert_node()` (lines 200-213) - Already has reset? Verify
5. `insert_edge()` (lines 252-267) - Already has reset? Verify
6. `delete_edges_from_level()` (lines 296-309) - Already has reset? Verify

**Files to modify**:
- `src/hnsw/storage.rs`: Add `ffi::sqlite3_reset(stmt)` before all returns

---

### Issue 2.2: Remove or Optimize Levels Table Inserts

**Problem**: Extra INSERT statements on every node insertion
**File**: `src/hnsw/storage.rs` lines 226-235
**Impact**: Unnecessary overhead (up to level+1 extra INSERTs per node)

**Solution**: Verify if levels table is actually needed, remove if not.

**Check usage**:
```bash
grep -r "hnsw_levels" src/
```

If only used in `get_nodes_at_level()` for rebuild operations, consider:
- Option A: Remove entirely, rebuild can query nodes table directly
- Option B: Make it optional (only insert if needed)
- Option C: Batch inserts in a single statement

**Recommendation**: Remove entirely for now, optimize rebuild if needed later.

```rust
// REMOVE lines 226-235 from insert_node():
// Also insert into levels table for efficient level queries
let levels_table = format!("{}_{}_hnsw_levels", table_name, column_name);
for lv in 0..=level {
    // ... DELETE THIS LOOP
}
```

Update `get_nodes_at_level()` to query nodes table directly:

```rust
pub fn get_nodes_at_level(
    db: &Connection,
    table_name: &str,
    column_name: &str,
    level: i32,
) -> Result<Vec<i64>> {
    let nodes_table = format!("{}_{}_hnsw_nodes", table_name, column_name);
    let query = format!(
        "SELECT rowid FROM \"{}\" WHERE level >= ? ORDER BY rowid",
        nodes_table
    );

    let mut stmt = db.prepare(&query).map_err(Error::Sqlite)?;
    let rowids = stmt
        .query_map([level as i64], |row| row.get::<_, i64>(0))
        .map_err(Error::Sqlite)?
        .collect::<std::result::Result<Vec<i64>, _>>()
        .map_err(Error::Sqlite)?;

    Ok(rowids)
}
```

**Files to modify**:
- `src/hnsw/storage.rs`: Remove levels table inserts
- `src/shadow.rs`: Consider removing levels table creation (check if used elsewhere)

---

## Phase 3: Fix Algorithm Correctness

### Issue 3.1: Add Multi-Connection Validation

**Problem**: No validation before insert/search operations
**File**: `src/hnsw/insert.rs`, `src/hnsw/search.rs`
**Impact**: Potential index corruption with concurrent connections

**Solution**: Add validation function like C implementation.

Add to `src/hnsw/mod.rs`:

```rust
impl HnswMetadata {
    /// Validate metadata is current and reload if changed by another connection
    /// Matches C implementation: hnsw_validate_and_refresh_caches()
    pub fn validate_and_refresh(
        &mut self,
        db: &Connection,
        table_name: &str,
        column_name: &str,
    ) -> Result<bool> {
        // Load current metadata from database
        let current = HnswMetadata::load_from_db(db, table_name, column_name)?;

        let current = match current {
            Some(m) => m,
            None => return Ok(false), // Index not initialized
        };

        // Check if hnsw_version changed (another connection modified index)
        if current.hnsw_version != self.hnsw_version {
            // Reload metadata
            *self = current;
            return Ok(false); // Metadata was stale
        }

        Ok(true) // Metadata is current
    }
}
```

Call at start of insert:

```rust
pub fn insert_hnsw(
    db: &Connection,
    metadata: &mut HnswMetadata,
    table_name: &str,
    column_name: &str,
    rowid: i64,
    vector: &[u8],
    stmt_cache: Option<&HnswStmtCache>,
) -> Result<()> {
    // CRITICAL: Validate metadata before insert (multi-connection safety)
    metadata.validate_and_refresh(db, table_name, column_name)?;

    // ... rest of insert
}
```

**Files to modify**:
- `src/hnsw/mod.rs`: Add `validate_and_refresh()` method
- `src/hnsw/insert.rs`: Call at start of `insert_hnsw()`
- `src/hnsw/search.rs`: Call at start of `search_hnsw()`

---

### Issue 3.2: Consider RNG Heuristic for Pruning

**Problem**: Greedy pruning may not maintain recall at scale
**File**: `src/hnsw/insert.rs`
**Impact**: Potentially degraded recall (needs testing to confirm)

**Decision**: Defer to Phase 4 (test with greedy first)

**Rationale**:
- Greedy pruning is simpler and may work for moderate scale
- RNG heuristic is complex (requires fetching all candidate vectors)
- Test recall with greedy first, implement RNG if recall is insufficient
- C comment says RNG "maintains small-world property and ensures high recall"

**If needed later**, implement in `prune_neighbor_if_needed()`:
1. Fetch neighbor's vector (center point)
2. Build candidate pool (existing + new node)
3. Compute distances to center for all candidates
4. Sort by distance to center
5. Apply RNG heuristic: keep candidates that are closer to center than to already-selected neighbors
6. Delete all edges, re-insert selected edges

**Files to modify** (if needed):
- `src/hnsw/insert.rs`: Implement full RNG heuristic in `prune_neighbor_if_needed()`

---

### Issue 3.3: Use search_layer for Greedy Search

**Problem**: Custom greedy search implementation
**File**: `src/hnsw/insert.rs` `find_closest_at_level()`
**Impact**: Less efficient than heap-based search, doesn't use cached statements

**Solution**: Replace with call to `search_layer()` with ef=1 (like C).

```rust
// REMOVE find_closest_at_level() function entirely

// In insert_hnsw(), REPLACE (lines 156-168):
for lv in (level + 1..=metadata.entry_point_level).rev() {
    let nearest = find_closest_at_level(
        db,
        metadata,
        table_name,
        column_name,
        &new_vec,
        current_nearest,
        lv,
        stmt_cache,
    )?;
    current_nearest = nearest;
}

// WITH:
for lv in (level + 1..=metadata.entry_point_level).rev() {
    // Create search context
    let search_stmt_cache = stmt_cache.map(|c| search::SearchStmtCache {
        get_node_data: Some(c.get_node_data),
        get_edges: Some(c.get_edges),
    });

    let ctx = search::SearchContext {
        db,
        metadata,
        table_name,
        column_name,
        query_vec: &new_vec,
        stmt_cache: search_stmt_cache.as_ref(),
    };

    // Greedy search: ef=1 returns single nearest neighbor
    let results = search::search_layer(&ctx, current_nearest, 1, lv)?;

    if let Some((nearest_rowid, _dist)) = results.first() {
        current_nearest = *nearest_rowid;
    }
}
```

**Files to modify**:
- `src/hnsw/insert.rs`: Remove `find_closest_at_level()`, use `search_layer()`

---

## Phase 4: Testing and Validation

### Test 4.1: Compilation

```bash
cargo clean
cargo build --release
```

Expected: Clean build with no errors.

---

### Test 4.2: Unit Tests

```bash
cargo test
```

Expected: All tests pass.

---

### Test 4.3: Recall Test

Use existing test in `tests/test_recall_accuracy.rs`:

```bash
cargo test --release test_recall_accuracy -- --nocapture
```

Expected: Recall >= 95% at 100K vectors, 768D.

---

### Test 4.4: Performance Benchmark

Create benchmark test:

```rust
// In tests/bench_insert.rs
#[test]
fn bench_insert_10k_vectors() {
    let db = Connection::open_in_memory().unwrap();
    // ... setup

    let start = std::time::Instant::now();
    for i in 0..10_000 {
        insert_hnsw(&db, &mut meta, "test", "vec", i, &random_vector(), None).unwrap();
    }
    let elapsed = start.elapsed();

    let per_vector = elapsed.as_micros() / 10_000;
    println!("Average insert time: {}μs per vector", per_vector);

    // Target: < 1000μs (1ms) per vector on average
    assert!(per_vector < 1000, "Insert too slow: {}μs > 1000μs", per_vector);
}
```

Expected: < 1ms per vector (match C performance).

---

### Test 4.5: Multi-Connection Safety

```rust
#[test]
fn test_concurrent_inserts() {
    // Open same database from multiple connections
    // Insert from each connection
    // Verify index integrity
    // Verify no corruption
}
```

Expected: No corruption, metadata stays consistent.

---

## Phase 5: Optimization (If Needed)

### If Performance Still Below C:

1. **Profile with `perf`**:
   ```bash
   cargo build --release
   perf record --call-graph=dwarf target/release/your_test
   perf report
   ```

2. **Check for**:
   - Excessive memory allocations
   - Redundant vector copies
   - Inefficient distance calculations
   - Statement preparation in hot paths

3. **Consider**:
   - SIMD distance calculations (if not already)
   - Arena allocator for temporary allocations
   - Batch operations (if lock contention not an issue)

---

## Implementation Checklist

### Phase 1: Compilation Fixes
- [ ] 1.1: Implement `fetch_neighbors_cached()` with cached statement
- [ ] 1.2: Replace batch operations with single-row (like C)
- [ ] 1.3: Fix `HnswStmtCache` missing `get_edges` field in vtab.rs
- [ ] 1.4: Add `save_dynamic_to_db()` method
- [ ] 1.5: Fix cache `retain()` syntax errors
- [ ] ✅ Verify compilation: `cargo build`

### Phase 2: Performance Fixes
- [ ] 2.1: Add `sqlite3_reset()` after all reads in storage.rs
- [ ] 2.2: Remove levels table inserts
- [ ] ✅ Run clippy: `cargo clippy --all-targets --all-features -- -D warnings`

### Phase 3: Correctness
- [ ] 3.1: Add multi-connection validation
- [ ] 3.2: (Defer) Consider RNG heuristic for pruning
- [ ] 3.3: Use `search_layer()` for greedy search
- [ ] ✅ Verify correctness: `cargo test`

### Phase 4: Testing
- [ ] 4.1: Verify compilation
- [ ] 4.2: Run all unit tests
- [ ] 4.3: Run recall accuracy test (>= 95%)
- [ ] 4.4: Run performance benchmark (< 1ms/vector)
- [ ] 4.5: Test multi-connection safety

### Phase 5: Optimization
- [ ] Profile if performance below target
- [ ] Address hotspots found in profiling

---

## Success Criteria

1. ✅ **Code compiles** with no errors
2. ✅ **All tests pass**
3. ✅ **Recall >= 95%** at 100K vectors, 768D
4. ✅ **Insert performance < 1ms per vector** (match C)
5. ✅ **Multi-connection safe** (no corruption)
6. ✅ **Clippy clean** with `-D warnings`

---

## Risk Mitigation

### Risk: Performance still below C after fixes
- **Mitigation**: Profile with `perf`, identify actual bottleneck
- **Fallback**: Consider algorithm differences (greedy vs RNG)

### Risk: Recall degraded with greedy pruning
- **Mitigation**: Test recall on large dataset first
- **Fallback**: Implement RNG heuristic (Phase 3.2)

### Risk: Batch operations cause lock issues
- **Mitigation**: Use single-row operations (proven C pattern)
- **Future**: Can optimize to batching later if needed

### Risk: Multi-connection issues persist
- **Mitigation**: Comprehensive validation tests
- **Fallback**: Add transaction isolation if needed

---

## Timeline Estimate

- **Phase 1** (Compilation): 2-4 hours
- **Phase 2** (Performance): 1-2 hours
- **Phase 3** (Correctness): 2-3 hours
- **Phase 4** (Testing): 1-2 hours
- **Phase 5** (Optimization): Variable (if needed)

**Total**: 6-11 hours for Phases 1-4

---

## Notes

- Follow C implementation patterns exactly (proven to work)
- Test frequently (after each phase)
- Don't optimize prematurely (fix bugs first)
- The #1 issue is `fetch_neighbors()` statement preparation
- Trust SQLite's page cache (don't need custom cache)
