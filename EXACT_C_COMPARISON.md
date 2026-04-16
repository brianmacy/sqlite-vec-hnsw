# Exact C vs Rust Implementation Comparison

## Current Performance Gap

**Rust (latest):**
- Early inserts (11-110): 3.53 ms/vector (1.45x slower)
- Late inserts (901-1000): 8.02 ms/vector (3.30x slower)
- **Degrades 2.27x as graph grows**

**C:**
- Constant: 2.43 ms/vector (no degradation)

---

## Prepared SQL Statements

### C Statements (sqlite-vec.c lines 11905-11969)

```c
// get_node_data
SELECT rowid, level, vector FROM hnsw_nodes WHERE rowid = ?

// get_edges
SELECT to_rowid FROM hnsw_edges WHERE from_rowid = ? AND level = ?
// NO ORDER BY, NO distance

// insert_node
INSERT INTO hnsw_nodes (rowid, level, vector) VALUES (?, ?, ?)

// insert_edge
INSERT OR IGNORE INTO hnsw_edges (from_rowid, to_rowid, level) VALUES (?, ?, ?)
// 3 parameters, NO distance

// delete_edges_from
DELETE FROM hnsw_edges WHERE from_rowid = ? AND level = ?

// update_meta
INSERT OR REPLACE INTO hnsw_meta (key, value) VALUES (?, ?)
```

### Rust Statements (AFTER our fixes)

```rust
// get_node_data - MATCHES ✅
SELECT rowid, level, vector FROM hnsw_nodes WHERE rowid = ?

// get_edges - MATCHES ✅ (we fixed this)
SELECT to_rowid FROM hnsw_edges WHERE from_rowid = ? AND level = ?

// get_edges_with_dist - EXTRA (not in C)
SELECT to_rowid, distance FROM hnsw_edges WHERE from_rowid = ? AND level = ?
// We prepare this but C doesn't have it

// insert_node - MATCHES ✅
INSERT OR REPLACE INTO hnsw_nodes (rowid, level, vector) VALUES (?, ?, ?)

// insert_edge - MATCHES ✅ (we fixed this)
INSERT OR IGNORE INTO hnsw_edges (from_rowid, to_rowid, level) VALUES (?, ?, ?)

// delete_edges_from - MATCHES ✅
DELETE FROM hnsw_edges WHERE from_rowid = ? AND level = ?

// update_meta - MATCHES ✅
INSERT OR REPLACE INTO hnsw_meta (key, value) VALUES (?, ?)
```

**Extra statement:** get_edges_with_dist (not needed)

---

## Insert Algorithm Comparison

### Phase 1: Validate & Insert Node

**C (lines 13329-13362):**
1. Validate caches (check versions)
2. Generate level
3. Insert node with retry logic
4. Cache node if cache enabled (disabled by default)

**Rust:**
1. ✅ Validate metadata (we do this)
2. ✅ Generate level (we do this)
3. ✅ Insert node (we do this)
4. ❌ NO retry logic for SQLITE_BUSY

**Deviation:** No retry logic

---

### Phase 2: Greedy Search (ef=1)

**C (lines 13401-13417):**
```c
for (i32 lc = meta->entry_point_level; lc > level; lc--) {
    rc = hnsw_search_layer_query(db, stmts, meta, current_nearest, vector, lc, 1,
                                  &layer_neighbors, &layer_distances, &layer_count);
    if (rc == SQLITE_OK && layer_count > 0) {
        current_nearest = layer_neighbors[0];
        sqlite3_free(layer_neighbors);
        sqlite3_free(layer_distances);
    }
}
```

**Rust:**
```rust
for lv in (level + 1..=metadata.entry_point_level).rev() {
    let results = search::search_layer(&ctx, current_nearest, 1, lv)?;
    if let Some((nearest_rowid, _dist)) = results.first() {
        current_nearest = *nearest_rowid;
    }
}
```

**MATCHES ✅**

---

### Phase 3: Insert at Each Level

**C (lines 13419-13478):**
```c
for (i32 lc = level; lc >= 0; lc--) {
    // Search layer with ef_construction
    rc = hnsw_search_layer_query(db, stmts, meta, current_nearest, vector, lc,
                                  meta->params.ef_construction,
                                  &candidates, &distances, &candidate_count);

    // Select M neighbors (bubble sort all candidates, take first M)
    i32 M = (lc == 0) ? meta->params.max_M0 : meta->params.M;
    i64 *selected = hnsw_select_neighbors(candidates, distances, candidate_count, M, &selected_count);

    // For each selected neighbor
    for (i32 i = 0; i < selected_count; i++) {
        i64 neighbor_rowid = selected[i];

        // Insert edge: new -> neighbor
        rc = hnsw_insert_edge_data(stmts->insert_edge, rowid, neighbor_rowid, lc);

        // Insert edge: neighbor -> new
        rc = hnsw_insert_edge_data(stmts->insert_edge, neighbor_rowid, rowid, lc);

        // Prune neighbor's connections
        hnsw_prune_neighbor_connections(db, stmts, meta, neighbor_rowid, rowid, lc, M);
    }

    sqlite3_free(selected);
    sqlite3_free(candidates);
    sqlite3_free(distances);
}
```

**Rust:**
```rust
for lv in (0..=level).rev() {
    // Search layer with ef_construction
    let neighbors = search::search_layer(&ctx, current_nearest,
                                         metadata.params.ef_construction as usize, lv)?;

    // Select M neighbors (take first M from heap results)
    let max_connections = if lv == 0 { max_m0 } else { m };
    let selected: Vec<(i64, f32)> = neighbors.into_iter().take(max_connections).collect();

    // Insert all edges first (LOOP 1)
    for (neighbor_rowid, _dist) in selected.iter() {
        storage::insert_edge(db, ..., rowid, *neighbor_rowid, lv, ...)?;
        storage::insert_edge(db, ..., *neighbor_rowid, rowid, lv, ...)?;
    }

    // Then prune all neighbors (LOOP 2)
    for (neighbor_rowid, dist) in selected.iter() {
        prune_neighbor_if_needed(db, ..., *neighbor_rowid, ...)?;
    }
}
```

**DEVIATION:** We have TWO loops (insert all, then prune all) vs C's ONE loop (insert+prune each)

**Impact:** Shouldn't affect performance significantly

---

## Key Finding: No Retry Logic

C has EXTENSIVE retry logic with exponential backoff for SQLITE_BUSY and SQLITE_BUSY_SNAPSHOT.

**C pattern (in EVERY database operation):**
```c
int retries = 0;
const int max_retries = 5;
while (retries < max_retries) {
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_DONE) return SQLITE_OK;
    if ((rc == SQLITE_BUSY || rc == SQLITE_BUSY_SNAPSHOT) && retries < max_retries - 1) {
        retries++;
        // Re-bind parameters
        // Randomized exponential backoff
        unsigned int random_val;
        sqlite3_randomness(sizeof(random_val), &random_val);
        int sleep_ms = (random_val % (retries * 100)) + 1;
        sqlite3_sleep(sleep_ms);
        continue;
    }
    return rc;
}
```

**Rust:** NO retry logic at all

**Impact:** If we're getting BUSY errors, we fail immediately instead of retrying. This could cause transaction conflicts or errors that slow things down.

---

## Current Status

**Fixed:**
✅ Lazy statement preparation on correct connection
✅ Removed ORDER BY from SQL queries
✅ Removed distance from INSERT (3 params like C)
✅ Statement caching working (verified with debug output)

**Still Different:**
1. ❌ No SQLITE_BUSY retry logic
2. ❌ Two loops (insert then prune) vs one loop
3. ❌ Broken pruning (relies on NULL distances)
4. ❌ Extra get_edges_with_dist statement

**Still Degrading:**
- 3.53ms → 8.02ms (2.27x) as graph grows
- C stays constant at 2.43ms

**Next Steps:**
1. Add proper retry logic like C
2. Fix pruning to compute distances (not fetch NULL)
3. Test if that fixes the degradation

---

## Hypothesis

The degradation might be caused by:
1. Failed operations due to SQLITE_BUSY (no retries) → errors accumulate
2. Broken pruning → edges accumulate → graph gets bloated → slower searches

To test: Check actual edge counts after 1000 inserts.
