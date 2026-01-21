//! sqlite-vec-hnsw: SQLite extension for vector search with HNSW indexing
//!
//! This is a Rust port of sqlite-vec with HNSW (Hierarchical Navigable Small World)
//! indexing for fast approximate nearest neighbor search.

pub mod connection_ext;
pub mod distance;
pub mod error;
pub mod hnsw;
pub mod shadow;
pub mod sql_functions;
pub mod vector;
pub mod vtab;

pub use error::{Error, Result};
pub use vector::{Vector, VectorType};

use rusqlite::Connection;
use rusqlite::ffi;

/// Initialize the sqlite-vec-hnsw extension
///
/// This function registers all SQL functions and virtual table modules.
pub fn init(db: &Connection) -> Result<()> {
    // Register scalar SQL functions
    sql_functions::register_all(db)?;

    // Register virtual table module
    vtab::register_vec0_module(db)?;

    Ok(())
}

/// Set SQLite pragmas optimized for HNSW performance on disk-based databases
///
/// These pragmas match the C sqlite-vec implementation for optimal performance.
/// Call this after opening the database connection for best results.
///
/// NOTE: For in-memory databases, these pragmas may hurt performance.
/// Use set_inmemory_pragmas() for in-memory databases instead.
///
/// # Pragmas set:
/// - `journal_mode = WAL` - Write-Ahead Logging for better concurrency
/// - `synchronous = NORMAL` - Balance between safety and performance
/// - `cache_size = -64000` - 64MB page cache (negative = KB)
/// - `temp_store = MEMORY` - Store temp tables in memory
/// - `mmap_size = 268435456` - 256MB memory-mapped I/O
pub fn set_performance_pragmas(db: &Connection) -> Result<()> {
    // Check if this is an in-memory database
    let is_memory: bool = db
        .query_row("PRAGMA database_list", [], |row| {
            let file: String = row.get(2)?;
            Ok(file.is_empty() || file == ":memory:")
        })
        .unwrap_or(false);

    if is_memory {
        // For in-memory databases, only set cache_size and temp_store
        return set_inmemory_pragmas(db);
    }

    // WAL mode for better concurrency during reads/writes
    db.execute_batch("PRAGMA journal_mode = WAL;")
        .map_err(Error::Sqlite)?;

    // NORMAL synchronous is safe with WAL and much faster than FULL
    db.execute_batch("PRAGMA synchronous = NORMAL;")
        .map_err(Error::Sqlite)?;

    // 64MB page cache (negative value = KB)
    db.execute_batch("PRAGMA cache_size = -64000;")
        .map_err(Error::Sqlite)?;

    // Store temporary tables in memory
    db.execute_batch("PRAGMA temp_store = MEMORY;")
        .map_err(Error::Sqlite)?;

    // 256MB memory-mapped I/O
    db.execute_batch("PRAGMA mmap_size = 268435456;")
        .map_err(Error::Sqlite)?;

    Ok(())
}

/// Set SQLite pragmas optimized for in-memory databases
///
/// Minimal pragmas that don't add overhead for in-memory operations.
pub fn set_inmemory_pragmas(db: &Connection) -> Result<()> {
    // Larger cache for in-memory operations
    db.execute_batch("PRAGMA cache_size = -64000;")
        .map_err(Error::Sqlite)?;

    // Store temporary tables in memory (already the default for :memory:)
    db.execute_batch("PRAGMA temp_store = MEMORY;")
        .map_err(Error::Sqlite)?;

    Ok(())
}

/// Initialize with performance pragmas (convenience function)
///
/// Equivalent to calling set_performance_pragmas() followed by init()
pub fn init_with_pragmas(db: &Connection) -> Result<()> {
    set_performance_pragmas(db)?;
    init(db)
}

/// Extension entry point for SQLite to load this as a shared library
///
/// # Safety
///
/// This function is called by SQLite's extension loading mechanism.
/// It must follow SQLite's extension initialization conventions.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sqlite3_sqlitevechnsw_init(
    db: *mut ffi::sqlite3,
    _err_msg: *mut *mut std::os::raw::c_char,
    p_api: *mut ffi::sqlite3_api_routines,
) -> std::os::raw::c_int {
    // Initialize the SQLite API routines pointer (required for loadable extensions)
    if unsafe { ffi::rusqlite_extension_init2(p_api) }.is_err() {
        return ffi::SQLITE_ERROR;
    }

    // SAFETY: We're being called by SQLite's extension loader with a valid db handle
    match std::panic::catch_unwind(|| {
        // Create a connection wrapper from the raw handle
        // SAFETY: db is a valid sqlite3 handle provided by SQLite
        let conn = match unsafe { Connection::from_handle(db) } {
            Ok(c) => c,
            Err(_) => return ffi::SQLITE_ERROR,
        };

        // Initialize the extension
        match init(&conn) {
            Ok(()) => {
                // Don't drop the connection - SQLite owns it
                std::mem::forget(conn);
                ffi::SQLITE_OK
            }
            Err(_) => ffi::SQLITE_ERROR,
        }
    }) {
        Ok(result) => result,
        Err(_) => ffi::SQLITE_ERROR,
    }
}

// Include additional entry point alias if compiled with loadable_extension_alias feature
// and SQLITE_VEC_ENTRY_POINT_ALIAS environment variable was set at build time
#[cfg(feature = "loadable_extension_alias")]
include!(concat!(env!("OUT_DIR"), "/entry_point_alias.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    #[test]
    fn test_extension_init() {
        let db = Connection::open_in_memory().unwrap();
        let result = init(&db);
        assert!(result.is_ok(), "Extension init should succeed");
    }
}
