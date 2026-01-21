//! Extension trait for rusqlite::Connection to support lazy statement preparation
//!
//! Adds prepare_or_reuse() method for vtab use cases where statements need
//! lazy initialization without requiring rusqlite fork modifications.

use crate::error::Result;
use rusqlite::{Connection, ffi};
use std::ffi::CString;

/// Extension trait adding lazy statement preparation to Connection
pub trait ConnectionExt {
    /// Prepare a statement if null, reuse if already prepared
    ///
    /// This helper enables lazy statement preparation for virtual table implementations
    /// where statements can't be prepared until shadow tables are created.
    ///
    /// # Safety
    /// - stmt_ptr must be either null or a valid prepared statement for this connection
    /// - If stmt_ptr is non-null, it must have been prepared on THIS connection
    /// - Caller is responsible for finalizing the statement when done
    ///
    /// # Arguments
    /// * `stmt_ptr` - Mutable reference to statement pointer (will be set if null)
    /// * `sql` - SQL string to prepare
    ///
    /// # Returns
    /// Ok(()) if statement is ready (either reused or newly prepared)
    unsafe fn prepare_or_reuse(
        &self,
        stmt_ptr: &mut *mut ffi::sqlite3_stmt,
        sql: &str,
    ) -> Result<()>;
}

impl ConnectionExt for Connection {
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn prepare_or_reuse(
        &self,
        stmt_ptr: &mut *mut ffi::sqlite3_stmt,
        sql: &str,
    ) -> Result<()> {
        // If already prepared, just reuse it
        if !stmt_ptr.is_null() {
            return Ok(());
        }

        // Prepare the statement on this connection
        let c_sql = CString::new(sql)
            .map_err(|e| crate::error::Error::InvalidParameter(format!("Invalid SQL: {}", e)))?;

        let rc = ffi::sqlite3_prepare_v2(
            self.handle(),
            c_sql.as_ptr(),
            -1,
            stmt_ptr,
            std::ptr::null_mut(),
        );

        if rc != ffi::SQLITE_OK {
            return Err(crate::error::Error::Sqlite(rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rc),
                None,
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prepare_or_reuse() {
        let db = Connection::open_in_memory().unwrap();

        let mut stmt: *mut ffi::sqlite3_stmt = std::ptr::null_mut();

        // First call: prepares the statement
        unsafe {
            db.prepare_or_reuse(&mut stmt, "SELECT 1").unwrap();
            assert!(!stmt.is_null());

            // Second call: reuses the statement
            let stmt_before = stmt;
            db.prepare_or_reuse(&mut stmt, "SELECT 1").unwrap();
            assert_eq!(stmt, stmt_before, "Should reuse same statement");

            // Cleanup
            ffi::sqlite3_finalize(stmt);
        }
    }
}
