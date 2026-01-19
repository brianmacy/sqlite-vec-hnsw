# Rusqlite Custom Features Patches

These patches add three missing features to rusqlite that are needed for full sqlite-vec-hnsw functionality:

1. **Operator Registration** - `Connection::overload_function()` for MATCH operator
2. **DB Handle Access** - `Context::get_connection()` for vec_rebuild_hnsw()
3. **Integrity Check** - `VTab::integrity()` for PRAGMA integrity_check

## Quick Start

### 1. Fork and Clone Rusqlite

```bash
# Go to https://github.com/rusqlite/rusqlite and click "Fork"
# Then clone YOUR fork:
cd ~/open_dev
git clone git@github.com:brianmacy/rusqlite.git
cd rusqlite
git checkout -b custom-vtab-features
```

### 2. Apply Patches

**Option A: Manual (recommended for understanding changes)**

Read each `.patch` file and make the corresponding changes:
- `01-operator-registration.patch` → edit `src/lib.rs`
- `02-db-handle-in-functions.patch` → edit `src/functions.rs`
- `03-vtab-integrity.patch` → edit `src/vtab.rs`

**Option B: Automated (if patches apply cleanly)**

```bash
cd ~/open_dev/rusqlite
for patch in ~/open_dev/sqlite-vec-hnsw/rusqlite-patches/*.patch; do
    git apply "$patch" || echo "Manual merge needed for $patch"
done
```

### 3. Test the Changes

```bash
cd ~/open_dev/rusqlite
cargo test
cargo clippy
```

### 4. Commit and Push

```bash
git add -A
git commit -m "Add virtual table enhancements: operator registration, DB handle access, integrity check

- Add Connection::overload_function() for custom operators (e.g., MATCH)
- Add Context::get_connection() for stateful scalar functions
- Add VTab::integrity() for PRAGMA integrity_check support (SQLite 3.44+)

These features enable full FTS-like virtual table implementations.

Closes #<issue-number>"

git push origin custom-vtab-features
```

### 5. Update sqlite-vec-hnsw

The `Cargo.toml` is already configured to use your fork:

```toml
rusqlite = {
    git = "https://github.com/brianmacy/rusqlite",
    branch = "custom-vtab-features",
    features = ["bundled", "vtab", "blob", "functions"]
}
```

Test it:

```bash
cd ~/open_dev/sqlite-vec-hnsw
cargo clean
cargo build
cargo test
```

### 6. Create PR to Upstream

Go to https://github.com/rusqlite/rusqlite and create a PR:

**Title:** Add virtual table enhancements for custom operators and integrity checks

**Description:**
```markdown
## Summary

This PR adds three enhancements to make rusqlite's virtual table support match SQLite's C API:

1. **`Connection::overload_function()`** - Register custom operators (e.g., MATCH) for virtual tables
2. **`Context::get_connection()`** - Access database handle from scalar functions
3. **`VTab::integrity()`** - Support PRAGMA integrity_check (SQLite 3.44+)

## Motivation

These features are needed for FTS-like virtual tables and database management functions:

- **MATCH operator**: Enable `WHERE column MATCH value` syntax (like FTS5)
- **DB handle**: Allow functions like `vec_rebuild_hnsw()` to query/modify tables
- **Integrity check**: Validate virtual table consistency during PRAGMA checks

## Use Case

Building a vector search extension (sqlite-vec-hnsw) that needs:
- KNN queries with MATCH operator
- Index rebuild functions that access the DB
- PRAGMA integrity_check integration

## Compatibility

- All changes are backward compatible
- `integrity()` uses SQLite version check for 3.44.0+
- `get_connection()` is marked unsafe (appropriate for DB handle access)
- `overload_function()` follows existing rusqlite patterns

## Testing

- Existing tests pass
- New functionality tested in downstream project: https://github.com/brianmacy/sqlite-vec-hnsw
```

## What Each Feature Does

### 1. Operator Registration (`overload_function`)

**Enables:**
```sql
SELECT * FROM vec_table WHERE embedding MATCH '[1,2,3]' AND k = 10;
```

**Without this:**
```rust
// ❌ MATCH not recognized by SQLite
db.query_row("... WHERE col MATCH val", [], |row| ...)?; // Error!
```

**With this:**
```rust
// ✅ Register MATCH operator
db.overload_function("match", 2)?;
// Now SQLite routes MATCH to your virtual table
```

### 2. DB Handle Access (`get_connection`)

**Enables:**
```sql
SELECT vec_rebuild_hnsw('table', 'column');  -- Rebuilds index
```

**Without this:**
```rust
db.create_scalar_function("vec_rebuild", 2, |ctx| {
    // ❌ No way to query vectors or update index
    // ctx only has arguments, not database access
})?;
```

**With this:**
```rust
db.create_scalar_function("vec_rebuild", 2, |ctx| {
    // ✅ Can access database
    let conn = unsafe { ctx.get_connection()? };
    conn.execute("DELETE FROM index_table WHERE ...", [])?;
    // Query, insert, delete - full DB access
    Ok(())
})?;
```

### 3. Integrity Check (`integrity`)

**Enables:**
```sql
PRAGMA integrity_check;
-- Validates both real tables AND virtual tables
```

**Without this:**
```rust
impl VTab for MyVTab {
    // ❌ No way to participate in integrity checks
}
```

**With this:**
```rust
impl VTab for MyVTab {
    fn integrity(&self, schema: &str, table: &str, flags: i32) -> Result<Option<String>> {
        // ✅ Validate your index
        if self.check_consistency()? {
            Ok(None)  // Valid
        } else {
            Ok(Some("Index corrupted: orphaned nodes found".to_string()))
        }
    }
}
```

## Why Upstream Might Accept This

✅ **Good reasons:**
- Matches SQLite C API exactly
- Enables FTS-like virtual tables in Rust
- Backward compatible
- Small, focused changes
- Well-documented use case

⚠️ **Potential concerns:**
- `get_connection()` breaks function purity (but marked unsafe)
- Adds complexity to vtab trait

**Fallback:** If upstream doesn't accept, maintain the fork. Other projects will find it useful!

## References

- [sqlite3_overload_function](https://www.sqlite.org/c3ref/overload_function.html)
- [sqlite3_context_db_handle](https://www.sqlite.org/c3ref/context_db_handle.html)
- [xIntegrity callback](https://www.sqlite.org/c3ref/module.html) (SQLite 3.44.0+)
