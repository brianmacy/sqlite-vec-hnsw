# Rusqlite Fork Patch for Loadable Extensions

## Issue

The rusqlite fork's `rusqlite_extension_init2()` function performs overly strict version checking that prevents loadable extensions from working with SQLite versions different from the bundled version.

**Original behavior:**
- Extension built with bundled SQLite 3.51.2
- System SQLite is 3.50.4
- Version check: `if SQLITE_VERSION_NUMBER > version` (i.e., if 3051002 > 3050004)
- Result: Extension rejected with "Failed to initialize SQLite extension API"

**Problem:** The check prevents loading the extension into OLDER SQLite versions, even when those versions have all required APIs (system SQLite 3.50.4 has xIntegrity support which requires 3.44.0+).

## Fix

Modified `/home/parallels/.cargo/git/checkouts/rusqlite-96d599330613ead4/dfba8b0/libsqlite3-sys/sqlite3/bindgen_bundled_version_ext.rs`:

Changed the version check from:
```rust
if SQLITE_VERSION_NUMBER > version {
    return Err(crate::InitError::VersionMismatch {
        compile_time: SQLITE_VERSION_NUMBER,
        runtime: version,
    });
}
```

To:
```rust
const MIN_REQUIRED_VERSION: i32 = 3044000; // SQLite 3.44.0 for xIntegrity support
if version < MIN_REQUIRED_VERSION {
    return Err(crate::InitError::VersionMismatch {
        compile_time: SQLITE_VERSION_NUMBER,
        runtime: version,
    });
}
```

**New behavior:**
- Extension works with any SQLite >= 3.44.0
- Compatible with both older (3.50.4) and newer (3.51.2+) versions
- Properly rejects SQLite < 3.44.0 which lacks required xIntegrity API

## Files Modified

1. **rusqlite fork (git checkout):** `/home/parallels/.cargo/git/checkouts/rusqlite-96d599330613ead4/dfba8b0/libsqlite3-sys/sqlite3/bindgen_bundled_version_ext.rs`
   - Line 6897: Changed version check from `SQLITE_VERSION_NUMBER > version` to `version < MIN_REQUIRED_VERSION`
   - Added MIN_REQUIRED_VERSION constant (3044000)

2. **sqlite-vec-hnsw:** `src/lib.rs`
   - Simplified error handling (removed complex version mismatch handling since it's now in rusqlite)
   - Added better error messages with `set_error()` helper

## Testing

Test: `test_SQLiteErrorSeverity` (libs/db/test_SQLiteErrorSeverity.cpp)
- **Before fix:** FAILED - Extension failed to load with version mismatch error
- **After fix:** PASSED - Extension loads successfully and all 8 test cases pass

## Long-term Solution

This patch should be submitted to the rusqlite fork maintainer (brianmacy/rusqlite branch custom-vtab-features) as it makes loadable extensions more flexible and compatible with a range of SQLite versions, which is the expected behavior for loadable extensions.

## Note

The cargo git checkout is at:
- `/home/parallels/.cargo/git/checkouts/rusqlite-96d599330613ead4/dfba8b0/`
- This directory contains the forked rusqlite from https://github.com/brianmacy/rusqlite (branch: custom-vtab-features)
- The patch is applied to the checkout, not the original repository
- Future `cargo clean` may require re-applying this patch
