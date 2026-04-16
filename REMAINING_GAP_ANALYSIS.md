# Remaining 4.5x Performance Gap Analysis

## Current Performance

**Rust (with Phase 1-3 fixes):**
- Insert: 10.95 ms/vector
- Search: 4.05 ms/search

**C (actual measured):**
- Insert: 2.43 ms/vector
- Search: 1.35 ms/search

**Gap: 4.5x slower on insert, 3x slower on search**

---

## Fixes Already Applied

✅ Phase 1: `fetch_neighbors_cached()` uses cached statements (no more SQL parsing)
✅ Phase 2: `sqlite3_reset()` releases WAL locks immediately
✅ Phase 2: Removed levels table overhead
✅ Phase 3: Using `search_layer()` instead of custom greedy search
✅ Phase 3: Added multi-connection validation

---

## Detailed Insert Path Comparison

### DEVIATION 1: validate_and_refresh() on EVERY Insert

**C Code** (C_INSERT_PSEUDOCODE.md Phase 0):
```c
// PHASE 0: VALIDATE CACHES
rc = hnsw_validate_and_refresh_caches(db, stmts, meta, "INSERT");
// This function:
//   1. Reads PRAGMA data_version (SQLite's DB change counter)
//   2. Reads hnsw_version from meta table
//   3. If either changed: clears cache, reloads metadata
//   4. Returns SQLITE_OK or error
```

**Rust Code** (src/hnsw/insert.rs:145-146):
```rust
// CRITICAL: Validate metadata before insert
metadata.validate_and_refresh(db, table_name, column_name)?;
```

**Rust Implementation** (src/hnsw/mod.rs:313-335):
```rust
pub fn validate_and_refresh(&mut self, db: &Connection, table_name: &str, column_name: &str) -> Result<bool> {
    // Load current metadata from database
    let current = HnswMetadata::load_from_db(db, table_name, column_name)?;
    // This calls: SELECT value FROM meta WHERE key=? for EVERY metadata field (14 queries!)

    let current = match current {
        Some(m) => m,
        None => return Ok(false),
    };

    // Check if hnsw_version changed
    if current.hnsw_version != self.hnsw_version {
        *self = current;
        return Ok(false);
    }

    Ok(true)
}
```

**Key Difference:**

C only checks TWO things:
1. `PRAGMA data_version` (1 query)
2. `hnsw_version` from meta (1 query)
3. **Total: 2 queries**, only reloads if version mismatch

Rust calls `load_from_db()` which:
- Queries **14 metadata fields** (M, max_M0, ef_construction, ef_search, max_level, level_factor, entry_point_rowid, entry_point_level, num_nodes, dimensions, element_type, distance_metric, rng_seed, hnsw_version)
- **Total: 14 queries EVERY insert**

**Impact:** 14 extra queries per insert vs 2 in C

**Fix:** Only query hnsw_version, reload full metadata only if changed:

```rust
pub fn validate_and_refresh(&mut self, db: &Connection, table_name: &str, column_name: &str) -> Result<bool> {
    let meta_table = format!("{}_{}_hnsw_meta", table_name, column_name);
    let query = format!("SELECT value FROM \"{}\" WHERE key = 'hnsw_version'", meta_table);

    // Just check version first
    let current_version: Option<i64> = db
        .query_row(&query, [], |row| row.get::<_, String>(0))
        .optional()?
        .and_then(|s| s.parse().ok());

    if let Some(curr_ver) = current_version {
        if curr_ver != self.hnsw_version {
            // Version changed - reload full metadata
            let current = HnswMetadata::load_from_db(db, table_name, column_name)?;
            if let Some(m) = current {
                *self = m;
                return Ok(false);
            }
        }
    }

    Ok(true) // Version unchanged
}
```

---

### DEVIATION 2: save_dynamic_to_db() on EVERY Insert

**C Code** (C_INSERT_PSEUDOCODE.md Phase 3, 7):
```c
// Update just num_nodes (1 query)
sprintf(value_buf, "%d", meta->num_nodes);
rc = hnsw_update_metadata_value(stmts->update_meta, "num_nodes", value_buf);
// Uses PREPARED STATEMENT: stmts->update_meta

// Later...
// Increment hnsw_version (1 query)
hnsw_set_hnsw_version(stmts->update_meta, new_version);
// Uses SAME PREPARED STATEMENT

// Total: 2 queries using PREPARED statement
```

**Rust Code** (src/hnsw/insert.rs:291-293):
```rust
metadata.num_nodes += 1;
metadata.hnsw_version += 1;
metadata.save_dynamic_to_db(db, table_name, column_name)?;
```

**Rust Implementation** (src/hnsw/mod.rs:279-307):
```rust
pub fn save_dynamic_to_db(&self, db: &Connection, table_name: &str, column_name: &str) -> Result<()> {
    let meta_table = format!("{}_{}_hnsw_meta", table_name, column_name);
    let update_sql = format!(
        "INSERT OR REPLACE INTO \"{}\" (key, value) VALUES (?, ?)",
        meta_table
    );

    // PROBLEM: Prepares statement 4 times!
    db.execute(&update_sql, ["entry_point_rowid", &self.entry_point_rowid.to_string()])?;
    db.execute(&update_sql, ["entry_point_level", &self.entry_point_level.to_string()])?;
    db.execute(&update_sql, ["num_nodes", &self.num_nodes.to_string()])?;
    db.execute(&update_sql, ["hnsw_version", &self.hnsw_version.to_string()])?;

    Ok(())
}
```

**Key Differences:**

1. **C uses ONE prepared statement** (`stmts->update_meta`) reused for all metadata updates
2. **Rust calls `db.execute()` 4 times**, each prepares a NEW statement
3. **C only updates changed fields** (num_nodes, hnsw_version typically)
4. **Rust updates 4 fields every time** (entry_point_rowid, entry_point_level, num_nodes, hnsw_version)

**Impact:** 4 statement preparations per insert vs 0 in C (using cached statement)

**Fix:** Use cached prepared statement from HnswStmtCache:

```rust
// In insert.rs, use cached statement:
if let Some(cache) = stmt_cache {
    // Use cache.update_meta prepared statement
    // Just update num_nodes and hnsw_version
}
```

But we don't have `update_meta` in our HnswStmtCache yet!

---

### DEVIATION 3: Missing update_meta Cached Statement

**C Code** (sqlite-vec.c HnswStatementCache):
```c
struct HnswStatementCache {
    sqlite3_stmt *get_node_data;
    sqlite3_stmt *get_edges;
    sqlite3_stmt *insert_node;
    sqlite3_stmt *insert_edge;
    sqlite3_stmt *delete_edges_from;
    sqlite3_stmt *update_meta;      // ← CRITICAL: Used for ALL metadata updates
    // ...
}
```

**Rust Code** (src/hnsw/insert.rs:95-102):
```rust
pub struct HnswStmtCache {
    pub get_node_data: *mut rusqlite::ffi::sqlite3_stmt,
    pub get_edges: *mut rusqlite::ffi::sqlite3_stmt,
    pub get_edges_with_dist: *mut rusqlite::ffi::sqlite3_stmt,
    pub insert_node: *mut rusqlite::ffi::sqlite3_stmt,
    pub insert_edge: *mut rusqlite::ffi::sqlite3_stmt,
    pub delete_edges_from: *mut rusqlite::ffi::sqlite3_stmt,
    // MISSING: update_meta
}
```

**Impact:** Every metadata update prepares a new statement instead of reusing cached one

**Fix:**
1. Add `update_meta` to HnswStmtCache
2. Prepare it in vtab.rs
3. Use it in save_dynamic_to_db()

---

### DEVIATION 4: Metadata Updates Pattern

**C Code** - Incremental updates during insert:
```c
// Early in insert:
meta->num_nodes++;
sprintf(value_buf, "%d", meta->num_nodes);
rc = hnsw_update_metadata_value(stmts->update_meta, "num_nodes", value_buf);
// Updates DB immediately with cached statement

// Later if entry point changes:
if (level > meta->entry_point_level) {
    meta->entry_point_rowid = rowid;
    meta->entry_point_level = level;

    sprintf(value_buf, "%lld", rowid);
    rc = hnsw_update_metadata_value(stmts->update_meta, "entry_point_rowid", value_buf);

    sprintf(value_buf, "%d", level);
    rc = hnsw_update_metadata_value(stmts->update_meta, "entry_point_level", value_buf);
}

// At very end:
rc = hnsw_set_hnsw_version(stmts->update_meta, new_version);
```

**Rust Code** - Batch update at end:
```rust
// At end of insert (line 291-293):
metadata.num_nodes += 1;
metadata.hnsw_version += 1;
metadata.save_dynamic_to_db(db, table_name, column_name)?;
// Saves 4 fields with 4 unprepared statements
```

**Difference:**
- C: Updates metadata fields incrementally, each with cached statement (2-4 uses of ONE cached statement)
- Rust: Batches all updates at end, prepares NEW statement for EACH field (4 preparations)

---

### DEVIATION 5: Pruning Algorithm Complexity

Already documented but worth repeating:

**C**: RNG heuristic - fetches ALL candidate vectors, computes pairwise distances, applies diversity selection
**Rust**: Greedy - sorts by distance, keeps closest, no extra fetches

**C's own benchmark says**: RNG pruning makes it 30% SLOWER (206 → 137 vec/sec)

**Therefore:** Our greedy pruning should be FASTER, not slower. This is NOT the bottleneck.

---

## Summary of Remaining Issues

### CRITICAL: Statement Preparation Overhead

**Issue 1: validate_and_refresh() - 14 queries per insert**
- C: 2 queries (data_version + hnsw_version)
- Rust: 14 queries (loads all metadata fields)
- **Impact: 7x more queries**

**Issue 2: save_dynamic_to_db() - 4 statement preparations per insert**
- C: 0 preparations (uses cached statement)
- Rust: 4 preparations (db.execute() each time)
- **Impact: 4 unprepared queries per insert**

**Issue 3: Missing update_meta in cache**
- C: Has update_meta cached statement
- Rust: Missing from HnswStmtCache
- **Impact: Can't use cached statement for metadata updates**

### Total Extra Overhead Per Insert

**Rust does:**
- 14 queries in validate_and_refresh() (vs 2 in C) = **+12 extra queries**
- 4 unprepared queries in save_dynamic_to_db() (vs 0 in C) = **+4 unprepared queries**
- **Total: +16 extra queries per insert, 4 of which are unprepared**

**At 1000 inserts:**
- 16,000 extra queries
- 4,000 extra statement preparations

This could EASILY account for 4.5x slowdown!

---

## Fix Priority

### CRITICAL FIX 1: Optimize validate_and_refresh()

Only query hnsw_version, not all metadata:

```rust
pub fn validate_and_refresh(&mut self, db: &Connection, table_name: &str, column_name: &str) -> Result<bool> {
    let meta_table = format!("{}_{}_hnsw_meta", table_name, column_name);

    // Just check version (1 query)
    let current_version: Option<i64> = db
        .query_row(
            &format!("SELECT value FROM \"{}\" WHERE key = 'hnsw_version'", meta_table),
            [],
            |row| row.get::<_, String>(0)
        )
        .optional()?
        .and_then(|s| s.parse().ok());

    if let Some(curr_ver) = current_version {
        if curr_ver != self.hnsw_version {
            // Version changed - reload full metadata
            if let Some(current) = HnswMetadata::load_from_db(db, table_name, column_name)? {
                *self = current;
                return Ok(false);
            }
        }
    }

    Ok(true)
}
```

**Impact:** 14 queries → 1 query per insert (13 queries saved)

---

### CRITICAL FIX 2: Add update_meta to Statement Cache

**Add to HnswStmtCache** (src/hnsw/insert.rs):
```rust
pub struct HnswStmtCache {
    pub get_node_data: *mut rusqlite::ffi::sqlite3_stmt,
    pub get_edges: *mut rusqlite::ffi::sqlite3_stmt,
    pub get_edges_with_dist: *mut rusqlite::ffi::sqlite3_stmt,
    pub insert_node: *mut rusqlite::ffi::sqlite3_stmt,
    pub insert_edge: *mut rusqlite::ffi::sqlite3_stmt,
    pub delete_edges_from: *mut rusqlite::ffi::sqlite3_stmt,
    pub update_meta: *mut rusqlite::ffi::sqlite3_stmt,  // ADD THIS
}
```

**Prepare in vtab.rs**:
```rust
// SQL: INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)
let sql = CString::new(format!(
    "INSERT OR REPLACE INTO \"{}_{}_hnsw_meta\" (key, value) VALUES (?, ?)",
    table_name, column_name
)).unwrap();

let rc = ffi::sqlite3_prepare_v2(
    db,
    sql.as_ptr(),
    -1,
    &mut self.update_meta,
    std::ptr::null_mut(),
);
```

**Use in save_dynamic_to_db()**:
```rust
pub fn save_dynamic_to_db_cached(
    &self,
    stmt: *mut ffi::sqlite3_stmt,
) -> Result<()> {
    unsafe {
        // Update num_nodes
        ffi::sqlite3_reset(stmt);
        ffi::sqlite3_bind_text(stmt, 1, b"num_nodes\0".as_ptr() as *const _, -1, ffi::SQLITE_STATIC());
        let num_str = self.num_nodes.to_string();
        ffi::sqlite3_bind_text(stmt, 2, num_str.as_ptr() as *const _, num_str.len() as i32, ffi::SQLITE_TRANSIENT());
        ffi::sqlite3_step(stmt);
        ffi::sqlite3_reset(stmt);

        // Update hnsw_version
        ffi::sqlite3_reset(stmt);
        ffi::sqlite3_bind_text(stmt, 1, b"hnsw_version\0".as_ptr() as *const _, -1, ffi::SQLITE_STATIC());
        let ver_str = self.hnsw_version.to_string();
        ffi::sqlite3_bind_text(stmt, 2, ver_str.as_ptr() as *const _, ver_str.len() as i32, ffi::SQLITE_TRANSIENT());
        ffi::sqlite3_step(stmt);
        ffi::sqlite3_reset(stmt);

        // Only update entry point if it changed (check metadata.entry_point_dirty flag)
    }
    Ok(())
}
```

**Impact:** 4 statement preparations → 0 (uses cached statement)

---

### DEVIATION 6: Entry Point Update Logic

**C Code**: Only updates entry_point in metadata table if it actually changed:
```c
if (level > meta->entry_point_level) {
    meta->entry_point_rowid = rowid;
    meta->entry_point_level = level;

    // Update in DB (2 queries with cached statement)
    rc = hnsw_update_metadata_value(stmts->update_meta, "entry_point_rowid", ...);
    rc = hnsw_update_metadata_value(stmts->update_meta, "entry_point_level", ...);
}
```

**Rust Code**: Always saves entry_point in save_dynamic_to_db() even if unchanged:
```rust
metadata.save_dynamic_to_db(db, table_name, column_name)?;
// This always updates entry_point_rowid and entry_point_level
// Even if they didn't change!
```

**Impact:** 2 extra unnecessary updates per insert (most inserts don't change entry point)

---

## Expected Performance After Fixes

**Current overhead per insert:**
- validate_and_refresh(): 14 queries → should be 1
- save_dynamic_to_db(): 4 unprepared queries → should be 2 with cached statement
- Unnecessary entry_point updates: 2 → should be 0 (conditional)

**Total reduction:**
- 14 + 4 = 18 queries → 1 + 2 = 3 queries
- 4 unprepared statements → 0 unprepared statements
- **Reduction: 15 queries + 4 preparations eliminated**

**Expected improvement:**
- Statement preparation is expensive (SQL parsing + query planning)
- Extra queries add round-trip overhead
- Eliminating 4 preparations + 15 queries could give us 2-3x speedup
- **Target: 10.95ms → 3.5-5.5ms** (close to C's 2.43ms)

---

## Implementation Plan

### Fix 1: Optimize validate_and_refresh()
- Change to only query hnsw_version (1 query)
- Only reload full metadata if version changed
- **Files:** src/hnsw/mod.rs

### Fix 2: Add update_meta to Statement Cache
- Add field to HnswStmtCache struct
- Prepare statement in vtab.rs prepare() function
- Pass to insert_hnsw()
- **Files:** src/hnsw/insert.rs, src/vtab.rs

### Fix 3: Create save_dynamic_to_db_cached()
- Takes cached statement as parameter
- Uses ffi calls with cached statement
- Only updates changed fields (track dirty flags)
- **Files:** src/hnsw/mod.rs

### Fix 4: Track Entry Point Changes
- Add dirty flag to know if entry point changed
- Only update in DB if changed
- **Files:** src/hnsw/insert.rs

---

## Other Potential Issues (Lower Priority)

### Possible Issue: Vector Parsing Overhead

**C Code**: Works directly with `const void *vector` blob (no parsing until distance calc)

**Rust Code**: Calls `Vector::from_blob()` multiple times
- Line 174: Parse once for new_vec (used in all distance calcs) - OK
- In search_layer: Parses every neighbor vector for distance calc

May not be significant, but worth checking if we parse vectors more than C does.

### Possible Issue: Different Distance Calculation

Need to verify our SIMD distance functions match C's performance.

### Possible Issue: BinaryHeap vs C's Manual Heap

Rust uses std::collections::BinaryHeap, C uses manual heap operations.
Unlikely to be significant, but worth profiling.

---

## Next Steps

1. ✅ Optimize validate_and_refresh() - only query version
2. ✅ Add update_meta to HnswStmtCache
3. ✅ Use cached statement in save_dynamic_to_db()
4. ✅ Track and conditionally update entry_point
5. 📊 Run benchmark again
6. 🔍 Profile if still >3ms per vector

Expected result: **3-5ms per vector** (within 2x of C's 2.43ms)
