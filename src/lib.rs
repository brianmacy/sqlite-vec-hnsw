//! sqlite-vec-hnsw: SQLite extension for vector search with HNSW indexing
//!
//! This is a Rust port of sqlite-vec with HNSW (Hierarchical Navigable Small World)
//! indexing for fast approximate nearest neighbor search.

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

/// Extension entry point for SQLite to load this as a shared library
///
/// # Safety
///
/// This function is called by SQLite's extension loading mechanism.
/// It must follow SQLite's extension initialization conventions.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sqlite3_sqlitevechnsw_init(
    _db: *mut ffi::sqlite3,
    _err_msg: *mut *mut std::os::raw::c_char,
    api: *mut ffi::sqlite3_api_routines,
) -> std::os::raw::c_int {
    // Initialize SQLite API
    if api.is_null() {
        return ffi::SQLITE_ERROR;
    }

    // TODO: Implement full extension initialization
    // For now, return unimplemented
    ffi::SQLITE_ERROR
}

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
