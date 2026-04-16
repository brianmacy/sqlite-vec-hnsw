# Detailed Comparison: Rust vs C Implementation

## Compilation Status

**CRITICAL**: The Rust code does NOT currently compile. Multiple missing functions:
- `storage::insert_edges_batch()` - called but not implemented
- `storage::delete_edges_batch()` - called but not implemented
- `storage::fetch_neighbors_cached()` - called but not implemented
- Cache module has syntax errors

## High-Level Comparison

### What the C Code Does (From Pseudo-Code)

1. **Prepared Statements**: Uses ONE set of prepared statements per connection
2. **Single-Row Operations**: Each edge insertion is ONE SQL statement
3. **Immediate sqlite3_reset()**: Releases WAL locks after each operation
4. **Cache**: DISABLED by default (24-42% slower when enabled)
5. **Relies on SQLite Page Cache**: Trusts SQLite to cache hot pages
6. **Minimal Memory Copies**: Only copies when necessary

### What the Rust Code Attempts (From Source)

1. **Prepared Statements**: Uses cached statements (GOOD)
2. **Batch Operations**: Tries to use batch insert/delete (NOT IMPLEMENTED)
3. **No Cache**: No HnswNodeCache equivalent (GOOD - matches C default)
4. **Different Pruning Algorithm**: Rust uses simpler pruning vs C's RNG heuristic

---

## Detailed Deviation Analysis

### DEVIATION 1: Batch Operations (NOT IMPLEMENTED)

**C Code**:
```c
// Insert edges ONE AT A TIME
for (i32 i = 0; i < selected_count; i++) {
    // Edge 1: new_node -> neighbor
    rc = hnsw_insert_edge_data(stmts->insert_edge, rowid, neighbor_rowid, lc);

    // Edge 2: neighbor -> new_node
    rc = hnsw_insert_edge_data(stmts->insert_edge, neighbor_rowid, rowid, lc);

    // Each call:
    sqlite3_reset(stmt);
    sqlite3_bind_int64(stmt, 1, from_rowid);
    sqlite3_bind_int64(stmt, 2, to_rowid);
    sqlite3_bind_int(stmt, 3, level);
    rc = sqlite3_step(stmt);
    sqlite3_reset(stmt);  // IMMEDIATE release of WAL lock
}
```

**Rust Code** (insert.rs:210-221):
```rust
// Collect all bidirectional edges
let edges: Vec<(i64, i64, i32, f32)> = selected
    .iter()
    .flat_map(|(neighbor_rowid, dist)| {
        vec![
            (rowid, *neighbor_rowid, lv, *dist),           // new node -> neighbor
            (*neighbor_rowid, rowid, lv, *dist),           // neighbor -> new node
        ]
    })
    .collect();

// Batch insert all edges in one SQL statement
storage::insert_edges_batch(db, table_name, column_name, &edges)?;
//        ^^^^^^^^^^^^^^^^^^^ FUNCTION DOES NOT EXIST
```

**Impact**:
- Rust tries to batch insert edges (more efficient IF implemented correctly)
- But function doesn't exist, so code doesn't compile
- C does single INSERTs with immediate reset - trusted pattern

**Why This Matters**:
- Batching COULD be faster (fewer round-trips)
- BUT: holding statement open longer = holding WAL locks longer
- C's immediate reset pattern is proven to work well with SQLite's page cache
- **RISK**: Batching might cause lock contention issues

---

### DEVIATION 2: Missing `fetch_neighbors_cached()` Function

**C Code** (sqlite-vec.c:12189-12286):
```c
static int hnsw_fetch_neighbors(struct HnswStatementCache *stmts, i64 rowid, i32 level,
                                 i64 **out_neighbors, i32 *out_count) {
    // Try cache first (if enabled, which it's not by default)
    if (stmts->node_cache) {
        // ... cache lookup (usually DISABLED)
    }

    // Cache miss - fetch from database using PREPARED STATEMENT
    sqlite3_stmt *stmt = stmts->get_edges;

    // Query: SELECT to_rowid FROM edges WHERE from_rowid=? AND level=?
    sqlite3_reset(stmt);
    sqlite3_bind_int64(stmt, 1, rowid);
    sqlite3_bind_int(stmt, 2, level);

    // Accumulate all rows
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        neighbors[count++] = sqlite3_column_int64(stmt, 0);
    }

    sqlite3_reset(stmt);  // IMMEDIATE reset
    *out_neighbors = neighbors;
    *out_count = count;

    return SQLITE_OK;
}
```

**Rust Code** (search.rs:211):
```rust
let neighbors = storage::fetch_neighbors_cached(
    //              ^^^^^^^^^^^^^^^^^^^^^^^ FUNCTION DOES NOT EXIST
    ctx.db,
    ctx.table_name,
    ctx.column_name,
    candidate.rowid,
    level,
    cached_edges_stmt,
)?;
```

**Actual Rust Implementation** (storage.rs:104-125):
```rust
pub fn fetch_neighbors(
    db: &Connection,
    table_name: &str,
    column_name: &str,
    from_rowid: i64,
    level: i32,
) -> Result<Vec<i64>> {
    let edges_table = format!("{}_{}_hnsw_edges", table_name, column_name);
    let query = format!(
        "SELECT to_rowid FROM \"{}\" WHERE from_rowid = ? AND level = ? ORDER BY distance",
        edges_table
    );

    // PROBLEM: This prepares a NEW statement every time!
    let mut stmt = db.prepare(&query).map_err(Error::Sqlite)?;
    //              ^^^^^^^^^^^ EXPENSIVE - parses SQL every call

    let neighbors = stmt
        .query_map([from_rowid, level as i64], |row| row.get::<_, i64>(0))
        .map_err(Error::Sqlite)?
        .collect::<std::result::Result<Vec<i64>, _>>()
        .map_err(Error::Sqlite)?;

    Ok(neighbors)
}
```

**Impact**:
- **CRITICAL PERFORMANCE BUG**: Rust prepares a new statement on EVERY fetch_neighbors call
- C uses cached prepared statement (no SQL parsing)
- During insert, fetch_neighbors is called HUNDREDS of times
- Each call: SQL parsing + query planning overhead
- **This alone could account for 5-10x slowdown**

**Evidence from Storage.rs**:
- `fetch_node_data()` at line 28 DOES use cached statements correctly
- `fetch_neighbors_with_distances()` at line 134 DOES use cached statements correctly
- But `fetch_neighbors()` does NOT use cached statements
- The search code tries to call `fetch_neighbors_cached()` which doesn't exist

---

### DEVIATION 3: Pruning Algorithm

**C Code** (sqlite-vec.c:13048-13180):
```c
static int hnsw_select_neighbors_by_heuristic(
    i64 *candidates, f32 *distances, i32 candidate_count,
    const void *center_vector,
    i32 max_neighbors,
    struct HnswStatementCache *stmts,
    struct HnswMetadata *meta,
    i64 **out_selected, i32 *out_selected_count) {

    // Build hash map for O(1) lookups
    // ... (rowid → candidate index mapping)

    // Process candidates from closest to farthest
    for (i32 i = 0; i < candidate_count && selected_count < max_neighbors; i++) {
        f32 dist_to_center = distances[i];
        int good = 1;

        // Check if candidate is too close to already-selected neighbors
        for (i32 j = 0; j < selected_count; j++) {
            i64 selected_rowid = selected[j];

            // Find candidate vector (lazy fetch with caching)
            const void *cand_vec = cached_vectors[i];
            if (!cand_vec) {
                hnsw_fetch_node_data(stmts, candidates[i], &level, &cand_vec, &size);
                cached_vectors[i] = cand_vec;
            }

            // Find selected vector
            const void *sel_vec = ...;

            // RNG heuristic: if dist(candidate, selected) < dist(candidate, center),
            // then candidate is redundant (selected node covers this area better)
            f32 dist_cand_to_sel = hnsw_calc_distance(meta, cand_vec, sel_vec);
            if (dist_cand_to_sel < dist_to_center) {
                good = 0;  // Discard candidate - it's redundant
                break;
            }
        }

        if (good) {
            selected[selected_count++] = candidates[i];
        }
    }

    // ... cleanup
    return SQLITE_OK;
}
```

**Rust Code** (insert.rs:204-206):
```rust
// Select M closest neighbors from search results (already sorted by distance)
// No need to re-fetch vectors - distances are computed during search
let selected: Vec<(i64, f32)> = neighbors.into_iter().take(max_connections).collect();
```

**Impact**:
- **C uses RNG heuristic**: Ensures diverse neighbors (maintains "small world" property)
- **Rust uses greedy selection**: Just takes M closest neighbors
- C's algorithm is proven to maintain >95% recall at scale
- Rust's simpler algorithm MIGHT work for small graphs but degrades recall on larger graphs

**Why RNG Heuristic Matters**:
- Prevents "hub" nodes that connect to everything (bad for search quality)
- Creates diverse paths through the graph (good for recall)
- The C implementation comment says: "This maintains the small-world property and ensures high recall"

---

### DEVIATION 4: Prune Function Differences

**C Code** (sqlite-vec.c:13165-13315):
```c
static void hnsw_prune_neighbor_connections(
    sqlite3 *db,
    struct HnswStatementCache *stmts,
    struct HnswMetadata *meta,
    i64 neighbor_rowid,
    i64 new_node_rowid,
    i32 level,
    i32 max_connections) {

    // 1. Fetch existing neighbors
    i64 *existing = NULL;
    i32 count = 0;
    hnsw_fetch_neighbors(stmts, neighbor_rowid, level, &existing, &count);

    // Early return if under limit
    if (count <= max_connections) {
        sqlite3_free(existing);
        return;
    }

    // 2. Fetch neighbor's vector (center point)
    const void *center_vec = NULL;
    hnsw_fetch_node_data(stmts, neighbor_rowid, &level_out, &center_vec, &size);

    // 3. Build candidate pool: existing + new node
    i64 *pool = malloc((count + 1) * sizeof(i64));
    memcpy(pool, existing, count * sizeof(i64));
    pool[count] = new_node_rowid;

    // 4. Compute distances to center for ALL candidates
    f32 *distances = malloc((count + 1) * sizeof(f32));
    for (i32 i = 0; i < count + 1; i++) {
        const void *cand_vec = NULL;
        hnsw_fetch_node_data(stmts, pool[i], &level, &cand_vec, &size);
        distances[i] = hnsw_calc_distance(meta, center_vec, cand_vec);
        sqlite3_free(cand_vec);
    }

    // 5. Sort by distance to center
    // ... bubble sort

    // 6. Apply RNG heuristic to select best max_connections
    i64 *selected = NULL;
    i32 selected_count = 0;
    hnsw_select_neighbors_by_heuristic(pool, distances, count + 1,
                                        center_vec, max_connections,
                                        stmts, meta, &selected, &selected_count);

    // 7. DELETE all edges at this level
    sqlite3_reset(stmts->delete_edges_from);
    sqlite3_bind_int64(stmts->delete_edges_from, 1, neighbor_rowid);
    sqlite3_bind_int(stmts->delete_edges_from, 2, level);
    rc = sqlite3_step(stmts->delete_edges_from);
    sqlite3_reset(stmts->delete_edges_from);

    // 8. Re-INSERT selected edges
    for (i32 i = 0; i < selected_count; i++) {
        sqlite3_reset(stmts->insert_edge);
        sqlite3_bind_int64(stmts->insert_edge, 1, neighbor_rowid);
        sqlite3_bind_int64(stmts->insert_edge, 2, selected[i]);
        sqlite3_bind_int(stmts->insert_edge, 3, level);
        rc = sqlite3_step(stmts->insert_edge);
        sqlite3_reset(stmts->insert_edge);
    }

    // 9. Invalidate cache (if enabled)
    if (stmts->node_cache) {
        hnsw_cache_invalidate_neighbors(stmts->node_cache, neighbor_rowid, level);
    }

    // Cleanup
    sqlite3_free(pool);
    sqlite3_free(distances);
    sqlite3_free(selected);
}
```

**Rust Code** (insert.rs:46-91):
```rust
fn prune_neighbor_if_needed(
    db: &Connection,
    table_name: &str,
    column_name: &str,
    neighbor_rowid: i64,
    _new_node_rowid: i64,
    _new_edge_dist: f32,
    level: i32,
    max_connections: usize,
    _metadata: &HnswMetadata,
    _new_vec: &Vector,
    _stmt_cache: Option<&HnswStmtCache>,
    get_edges_stmt: Option<*mut rusqlite::ffi::sqlite3_stmt>,
    _insert_edge_stmt: Option<*mut rusqlite::ffi::sqlite3_stmt>,
) -> Result<()> {
    // 1. Fetch current neighbor edges WITH DISTANCES
    let neighbor_edges = storage::fetch_neighbors_with_distances(
        db,
        table_name,
        column_name,
        neighbor_rowid,
        level,
        get_edges_stmt,
    )?;

    // 2. Early return if under limit
    if neighbor_edges.len() <= max_connections {
        return Ok(());
    }

    // 3. Sort by distance (GREEDY - just keep closest)
    let mut sorted_edges = neighbor_edges;
    sorted_edges.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // 4. Collect edges to DELETE (those beyond max_connections)
    let edges_to_remove: Vec<i64> = sorted_edges
        .into_iter()
        .skip(max_connections)
        .map(|(to_rowid, _dist)| to_rowid)
        .collect();

    // 5. Batch delete only the edges we need to remove
    storage::delete_edges_batch(db, table_name, column_name, neighbor_rowid, level, &edges_to_remove)?;
    //        ^^^^^^^^^^^^^^^^^ FUNCTION DOES NOT EXIST

    Ok(())
}
```

**Key Differences**:

1. **Rust fetches distances WITH edges** (one query)
   - C fetches edges, then fetches vectors to compute distances (multiple queries)
   - Rust approach is potentially more efficient IF it works

2. **Rust uses greedy pruning** (keep closest to query)
   - C uses RNG heuristic (keep diverse neighbors)
   - C's approach is proven to maintain recall

3. **Rust tries to batch delete**
   - C deletes ALL edges, then re-inserts selected ones
   - Rust tries to selectively delete only excess edges
   - Rust's approach COULD be more efficient but function doesn't exist

4. **Rust doesn't fetch neighbor vector or new node vector**
   - Many unused parameters suggest incomplete implementation
   - C needs these for RNG heuristic distance calculations

---

### DEVIATION 5: Greedy Search Algorithm (find_closest_at_level)

**C Code** (used in Phase 4 of insert):
```c
// Phase 4: Find insertion point
for (i32 lc = meta->entry_point_level; lc > level; lc--) {
    i64 *layer_neighbors = NULL;
    f32 *layer_distances = NULL;
    i32 layer_count = 0;

    // Search this layer with ef=1 (greedy)
    rc = hnsw_search_layer_query(db, stmts, meta, current_nearest, vector, lc, 1,
                                  &layer_neighbors, &layer_distances, &layer_count);

    if (rc == SQLITE_OK && layer_count > 0) {
        current_nearest = layer_neighbors[0];  // Move to nearest
        sqlite3_free(layer_neighbors);
        sqlite3_free(layer_distances);
    }
}
```

**Rust Code** (insert.rs:265-324):
```rust
fn find_closest_at_level(
    db: &Connection,
    metadata: &HnswMetadata,
    table_name: &str,
    column_name: &str,
    query_vec: &Vector,
    start_rowid: i64,
    level: i32,
    stmt_cache: Option<&HnswStmtCache>,
) -> Result<i64> {
    let mut current = start_rowid;
    let mut changed = true;

    while changed {
        changed = false;

        // Get current node's distance
        let current_node = storage::fetch_node_data(db, table_name, column_name, current, get_node_stmt)?
            .ok_or_else(|| Error::InvalidParameter(format!("Node {} not found", current)))?;

        let current_vec = Vector::from_blob(&current_node.vector, ...)?;
        let current_dist = distance::distance(query_vec, &current_vec, metadata.distance_metric)?;

        // Check all neighbors
        let neighbors = storage::fetch_neighbors(db, table_name, column_name, current, level)?;
        //                      ^^^^^^^^^^^^^^^^ USES SLOW PATH (no cached statement!)

        for neighbor_rowid in neighbors {
            let neighbor_node = storage::fetch_node_data(db, table_name, column_name, neighbor_rowid, get_node_stmt)?;
            let neighbor_node = match neighbor_node {
                Some(n) => n,
                None => continue,
            };

            let neighbor_vec = Vector::from_blob(&neighbor_node.vector, ...)?;
            let neighbor_dist = distance::distance(query_vec, &neighbor_vec, metadata.distance_metric)?;

            if neighbor_dist < current_dist {
                current = neighbor_rowid;
                changed = true;
                break;
            }
        }
    }

    Ok(current)
}
```

**Key Differences**:

1. **C calls search_layer with ef=1** (uses full HNSW search with heaps, visited set)
   - Proven algorithm from HNSW paper
   - Uses priority queues for efficient traversal

2. **Rust implements custom greedy search** (simple while loop)
   - Simpler but not the standard HNSW algorithm
   - No visited set - could revisit nodes (infinite loop risk?)
   - Less efficient than heap-based search

3. **Rust calls fetch_neighbors WITHOUT cached statement**
   - **CRITICAL**: Uses slow path that prepares statement every call
   - Called in a loop, so statement preparation happens many times

---

### DEVIATION 6: Statement Reset Patterns

**C Code** (consistent pattern everywhere):
```c
// Every statement follows this pattern:
sqlite3_reset(stmt);
sqlite3_bind_XXX(stmt, ...);
rc = sqlite3_step(stmt);
sqlite3_reset(stmt);  // IMMEDIATE reset

// OR for queries that return multiple rows:
sqlite3_reset(stmt);
sqlite3_bind_XXX(stmt, ...);
while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    // Extract data into malloc'd buffers
}
sqlite3_reset(stmt);  // IMMEDIATE reset after loop
```

**Rust Code** (storage.rs:36-64, using ffi):
```rust
unsafe {
    ffi::sqlite3_reset(stmt);
    ffi::sqlite3_bind_int64(stmt, 1, rowid);

    let rc = ffi::sqlite3_step(stmt);
    if rc == ffi::SQLITE_ROW {
        let blob_ptr = ffi::sqlite3_column_blob(stmt, 2);
        let blob_len = ffi::sqlite3_column_bytes(stmt, 2);
        let vector = std::slice::from_raw_parts(blob_ptr as *const u8, blob_len as usize).to_vec();
        //           ^^^^^^^^^^^ COPIES data while statement is still active

        return Ok(Some(HnswNode { ... }));
        //     ^^^^^^^^^^^ RETURNS without reset!
    } else if rc == ffi::SQLITE_DONE {
        return Ok(None);
        //     ^^^^^^^^^^^ RETURNS without reset!
    } else {
        return Err(...);
        //     ^^^^^^^^^^^ RETURNS without reset!
    }
}
```

**Impact**:
- **CRITICAL BUG**: Rust doesn't reset statement after successful read
- Statement remains active, holding WAL read lock
- In C, the comment says: "BUG FIX: Reset statement to release WAL read lock"
- This could cause lock contention and performance issues

**Correct Pattern** (from C):
```c
// Copy data BEFORE reset
const void *blob_ptr = sqlite3_column_blob(stmt, 2);
int blob_size = sqlite3_column_bytes(stmt, 2);

void *vector_copy = sqlite3_malloc64(blob_size);
memcpy(vector_copy, blob_ptr, blob_size);

// Reset IMMEDIATELY to release lock
sqlite3_reset(stmt);

*out_vector = vector_copy;  // Return the copy
return SQLITE_OK;
```

---

### DEVIATION 7: Insert Levels Table

**C Code**: No "levels" table. Entry point is stored in metadata.

**Rust Code** (storage.rs:226-235):
```rust
// Also insert into levels table for efficient level queries
let levels_table = format!("{}_{}_hnsw_levels", table_name, column_name);
for lv in 0..=level {
    let insert_level_sql = format!(
        "INSERT OR IGNORE INTO \"{}\" (level, rowid) VALUES (?, ?)",
        levels_table
    );
    db.execute(&insert_level_sql, [lv as i64, rowid])
        .map_err(Error::Sqlite)?;
}
```

**Impact**:
- Rust inserts into levels table for EVERY node insertion
- If node has level=5, that's 6 extra INSERT statements
- C doesn't have this table at all
- Unclear if this table is actually used or needed
- **Extra overhead on every insert**

---

### DEVIATION 8: Missing Validation and Error Handling

**C Code** (sqlite-vec.c:13329-13333):
```c
// CRITICAL: Validate caches before INSERT (handles multi-connection race conditions)
rc = hnsw_validate_and_refresh_caches(db, stmts, meta, "INSERT");
if (rc != SQLITE_OK) {
    return rc;
}
```

Where `hnsw_validate_and_refresh_caches()` does:
1. Reads `PRAGMA data_version` (SQLite's DB change counter)
2. Reads `hnsw_version` from meta table
3. If changed: clears cache, reloads metadata
4. Handles multi-connection/multi-process safety

**Rust Code**: No equivalent validation at start of insert.

**Impact**:
- Rust doesn't check if another connection modified the index
- Could lead to corrupted index in multi-connection scenarios
- C has this specifically for "multi-connection race conditions"

---

### DEVIATION 9: Metadata Save Pattern

**C Code** (saves incrementally):
```c
// Update just num_nodes
sprintf(value_buf, "%d", meta->num_nodes);
rc = hnsw_update_metadata_value(stmts->update_meta, "num_nodes", value_buf);

// Later, update just entry_point if needed
if (level > meta->entry_point_level) {
    sprintf(value_buf, "%lld", rowid);
    rc = hnsw_update_metadata_value(stmts->update_meta, "entry_point_rowid", value_buf);

    sprintf(value_buf, "%d", level);
    rc = hnsw_update_metadata_value(stmts->update_meta, "entry_point_level", value_buf);
}

// At end, increment hnsw_version
hnsw_set_hnsw_version(stmts->update_meta, new_version);
```

**Rust Code** (insert.rs:145, 260):
```rust
metadata.save_dynamic_to_db(db, table_name, column_name)?;
//        ^^^^^^^^^^^^^^^^^^^ FUNCTION DOES NOT EXIST

// Should be:
metadata.save_to_db(db, table_name, column_name)?;
// But this saves ALL metadata fields (overkill)
```

**Impact**:
- C saves only changed fields (minimal SQL)
- Rust tries to call non-existent function
- When fixed, Rust might save all fields every time (wasteful)

---

## Summary of Critical Issues

### Issues Preventing Compilation:
1. ❌ `storage::insert_edges_batch()` - NOT IMPLEMENTED
2. ❌ `storage::delete_edges_batch()` - NOT IMPLEMENTED
3. ❌ `storage::fetch_neighbors_cached()` - NOT IMPLEMENTED
4. ❌ `metadata.save_dynamic_to_db()` - NOT IMPLEMENTED
5. ❌ Cache module syntax errors in `retain()` calls

### Critical Performance Issues:
1. 🔥 **`fetch_neighbors()` prepares statement every call** - 5-10x slowdown
2. 🔥 **No `sqlite3_reset()` after reads** - holds WAL locks
3. 🔥 **Greedy pruning instead of RNG heuristic** - degrades recall
4. ⚠️ **Extra levels table inserts** - unnecessary overhead
5. ⚠️ **Custom greedy search vs search_layer** - less efficient

### Correctness Issues:
1. 🐛 **No multi-connection validation** - could corrupt index
2. 🐛 **Statement not reset after reads** - resource leaks
3. 🐛 **Simplified pruning algorithm** - may not maintain recall at scale

### Design Differences:
1. 📝 **Batch operations** - C uses single-row, Rust attempts batching
2. 📝 **Levels table** - Rust has extra table C doesn't have
3. 📝 **Simpler heuristics** - Rust uses greedy, C uses RNG

---

## Recommended Fix Priority

### Phase 1: Make it Compile
1. Implement `fetch_neighbors_cached()` using cached statement
2. Implement `insert_edges_batch()` OR change to single-row inserts like C
3. Implement `delete_edges_batch()` OR change to delete-all-then-reinsert like C
4. Fix `save_dynamic_to_db()` → `save_to_db()`
5. Fix cache `retain()` syntax errors

### Phase 2: Fix Critical Performance Bugs
1. **Fix `fetch_neighbors()` to use cached statement** - BIGGEST WIN
2. **Add `sqlite3_reset()` after all statement reads** - Fix lock holding
3. Remove levels table inserts (or make optional) - Reduce overhead

### Phase 3: Algorithm Correctness
1. Add multi-connection validation like C
2. Implement RNG heuristic for pruning (or verify greedy works)
3. Use `search_layer()` for greedy search instead of custom loop

### Phase 4: Optimization
1. Profile actual performance after fixes
2. Consider batch operations (if done correctly)
3. Benchmark against C implementation

---

## Key Insight

**The #1 performance issue is likely `fetch_neighbors()` preparing a new statement on every call.**

During a single insert operation:
- `search_layer()` is called multiple times
- Each `search_layer()` calls `fetch_neighbors()` hundreds of times
- Each call prepares a new statement (SQL parsing + query planning)
- This happens thousands of times per insert

Fix this first, and performance should improve dramatically.

The C implementation is fast because:
1. ✅ Uses prepared statements (avoids SQL parsing)
2. ✅ Resets statements immediately (releases WAL locks)
3. ✅ Relies on SQLite's page cache (don't need custom cache)
4. ✅ Proven algorithms (RNG heuristic, proper search_layer)

The Rust implementation should do the same!
