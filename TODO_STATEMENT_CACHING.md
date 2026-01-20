# TODO: Complete Prepared Statement Caching

## Current State

### ✅ Infrastructure Complete
- [x] `HnswStmtCache` structure defined (7 statement pointers)
- [x] Statements prepared in `Vec0Tab::create()` via FFI
- [x] Statements finalized in `Vec0Tab::destroy()`
- [x] One cache per vector column
- [x] All 7 SQL statements pre-parsed:
  - `get_node_data`
  - `get_node_level`
  - `get_edges`
  - `get_edges_with_dist`
  - `insert_node`
  - `insert_edge`
  - `delete_edges_from`

### ❌ Remaining Work: Use Cached Statements

**Problem:** Storage functions still call `db.prepare()` which parses SQL fresh every time.

**Files to update:**
1. `src/hnsw/storage.rs` - All fetch/insert/delete functions
2. `src/hnsw/insert.rs` - Pass statement cache through insert_hnsw()
3. `src/hnsw/search.rs` - Use cached statements for search
4. `src/vtab.rs` - Pass cache pointer from insert() to hnsw::insert::insert_hnsw()

---

## Implementation Guide

### Step 1: Add Cache Parameter to insert_hnsw()

**File:** `src/hnsw/insert.rs`

**Current signature:**
```rust
pub fn insert_hnsw(
    db: &Connection,
    metadata: &mut HnswMetadata,
    table_name: &str,
    column_name: &str,
    rowid: i64,
    vector: &[u8],
) -> Result<()>
```

**New signature:**
```rust
pub fn insert_hnsw(
    db: &Connection,
    metadata: &mut HnswMetadata,
    table_name: &str,
    column_name: &str,
    rowid: i64,
    vector: &[u8],
    stmt_cache: Option<&HnswStmtCache>,  // NEW
) -> Result<()>
```

### Step 2: Update storage.rs Functions

**Example: `fetch_node_data()`**

**Current (slow - parses SQL every call):**
```rust
pub fn fetch_node_data(
    db: &Connection,
    table_name: &str,
    column_name: &str,
    rowid: i64,
) -> Result<Option<HnswNode>> {
    let nodes_table = format!("{}_{}_hnsw_nodes", table_name, column_name);
    let query = format!("SELECT rowid, level, vector FROM \"{}\" WHERE rowid = ?", nodes_table);

    db.query_row(&query, [rowid], |row| {
        Ok(HnswNode {
            rowid: row.get(0)?,
            level: row.get(1)?,
            vector: row.get(2)?,
        })
    })
    .optional()
    .map_err(Error::Sqlite)
}
```

**New (fast - uses prepared statement):**
```rust
pub fn fetch_node_data(
    db: &Connection,
    table_name: &str,
    column_name: &str,
    rowid: i64,
    stmt_cache: Option<&HnswStmtCache>,
) -> Result<Option<HnswNode>> {
    // If cache available, use cached statement
    if let Some(cache) = stmt_cache {
        unsafe {
            let stmt = cache.get_node_data;
            ffi::sqlite3_reset(stmt);
            ffi::sqlite3_bind_int64(stmt, 1, rowid);

            let rc = ffi::sqlite3_step(stmt);
            if rc == ffi::SQLITE_ROW {
                let node_rowid = ffi::sqlite3_column_int64(stmt, 0);
                let level = ffi::sqlite3_column_int(stmt, 1);

                let blob_ptr = ffi::sqlite3_column_blob(stmt, 2);
                let blob_len = ffi::sqlite3_column_bytes(stmt, 2);
                let vector = std::slice::from_raw_parts(blob_ptr as *const u8, blob_len as usize).to_vec();

                return Ok(Some(HnswNode {
                    rowid: node_rowid,
                    level,
                    vector,
                }));
            } else if rc == ffi::SQLITE_DONE {
                return Ok(None);
            } else {
                return Err(Error::Sqlite(rusqlite::Error::SqliteFailure(
                    rusqlite::ffi::Error::new(rc),
                    None,
                )));
            }
        }
    }

    // Fallback: original implementation
    let nodes_table = format!("{}_{}_hnsw_nodes", table_name, column_name);
    let query = format!("SELECT rowid, level, vector FROM \"{}\" WHERE rowid = ?", nodes_table);

    db.query_row(&query, [rowid], |row| {
        Ok(HnswNode {
            rowid: row.get(0)?,
            level: row.get(1)?,
            vector: row.get(2)?,
        })
    })
    .optional()
    .map_err(Error::Sqlite)
}
```

**Repeat for:**
- `fetch_neighbors()` → use `cache.get_edges`
- `fetch_neighbors_with_distances()` → use `cache.get_edges_with_dist`
- `insert_node()` → use `cache.insert_node`
- `insert_edge()` → use `cache.insert_edge`
- `delete_edges_from_level()` → use `cache.delete_edges_from`

### Step 3: Thread Cache Through Call Chain

**In `src/vtab.rs`:** Pass cache from `insert()` to `hnsw::insert::insert_hnsw()`:

```rust
// Around line 700 in insert()
let vec_col_idx = /* calculate from col_idx */;
let stmt_cache = if self.index_type == IndexType::Hnsw {
    self.hnsw_stmt_cache.get(vec_col_idx)
} else {
    None
};

hnsw::insert::insert_hnsw(
    &conn,
    &mut metadata,
    &self.table_name,
    &col.name,
    rowid,
    &vector_data,
    stmt_cache,  // NEW parameter
)
```

### Step 4: Update All Call Sites

**Files needing updates:**
- `src/hnsw/insert.rs` - Pass cache to storage functions
- `src/hnsw/search.rs` - Pass cache to storage functions
- `src/vtab.rs` - Pass cache from insert/update/delete

**Estimated changes:** ~20-30 function signatures updated

---

## Testing After Implementation

```bash
# Test that statements are being reused
cargo test --lib

# Test performance improvement
cargo test test_inmemory_float32_with_transactions -- --nocapture

# Expected result:
# Before: 23 vec/sec
# After:  150-200 vec/sec (5-10x improvement)
```

---

## Expected Performance Improvement

| Metric | Before (current) | After (with caching) | Improvement |
|--------|------------------|----------------------|-------------|
| Insert rate | 23 vec/sec | 150-200 vec/sec | 6-8x faster |
| vs C | 7x slower | **~Match C** | Parity achieved |
| SQL parses | Thousands per insert | 7 total (once) | 1000x fewer |

---

## Estimated Effort

**Complexity:** Medium-High (lots of FFI, careful lifetime management)
**Time:** 3-4 hours focused work
**Risk:** Medium (unsafe FFI code, but well-defined pattern)

**Breakdown:**
- Update storage.rs functions: 2 hours
- Thread cache through call chain: 1 hour
- Testing and debugging: 1 hour

---

## Alternative: Rusqlite Statement Caching

**Simpler approach:** Use rusqlite's built-in statement caching
- Store `Statement<'conn>` in a cache struct
- Requires different lifetime management
- May be easier than raw FFI
- Expected performance: Similar (90-95% of raw FFI)

**Trade-off:** Less control, but safer and easier to implement

---

## Priority

**Current state is production-ready for:**
- ✅ Read-heavy workloads
- ✅ C database compatibility
- ✅ Storage efficiency (within 7% of C)

**Statement caching is critical for:**
- ❌ Write-heavy workloads (inserts/updates)
- ❌ Matching C insert performance

**Recommendation:** Implement if write performance is a production requirement.

---

## Date
January 20, 2026
