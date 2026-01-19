# Installation Instructions

Clone your fork and make these 3 simple changes:

## Setup

```bash
cd ~/open_dev
git clone git@github.com:brianmacy/rusqlite.git
cd rusqlite
git checkout -b custom-vtab-features
```

## Change 1: Add to `src/lib.rs`

Find the `impl Connection` block (around line 500), add this method:

```rust
    /// Register a custom operator for virtual tables
    ///
    /// This is typically used to enable MATCH operator for custom virtual tables.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use rusqlite::{Connection, Result};
    /// # fn main() -> Result<()> {
    /// let db = Connection::open_in_memory()?;
    /// db.overload_function("match", 2)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn overload_function(&self, name: &str, n_arg: c_int) -> Result<()> {
        let c_name = std::ffi::CString::new(name)?;
        unsafe {
            self.db.borrow_mut().check(ffi::sqlite3_overload_function(
                self.db.borrow().db(),
                c_name.as_ptr(),
                n_arg,
            ))
        }
    }
```

## Change 2: Add to `src/functions.rs`

Find the `impl<'a> Context<'a>` block (around line 200), add this method:

```rust
    /// Get the database connection handle
    ///
    /// This allows scalar functions to execute queries or modify the database.
    ///
    /// # Safety
    /// The returned connection borrows from this Context and must not outlive it.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use rusqlite::{Connection, Result, functions::FunctionFlags};
    /// # fn main() -> Result<()> {
    /// # let db = Connection::open_in_memory()?;
    /// db.create_scalar_function("my_func", 1, FunctionFlags::default(), |ctx| {
    ///     let conn = unsafe { ctx.get_connection()? };
    ///     Ok(())
    /// })?;
    /// # Ok(())
    /// # }
    /// ```
    pub unsafe fn get_connection(&self) -> Result<Connection> {
        Connection::from_handle(ffi::sqlite3_context_db_handle(self.ctx))
    }
```

## Change 3: Add to `src/vtab.rs`

Find the `pub trait VTab` definition (around line 150), add this method:

```rust
    /// Called during PRAGMA integrity_check to validate virtual table
    ///
    /// Return Ok(None) if valid, or Ok(Some(error_message)) if corrupted.
    fn integrity(&self, _schema: &str, _table_name: &str, _flags: c_int) -> Result<Option<String>> {
        Ok(None)
    }
```

Then find the `create_module` function (around line 600) and add the xIntegrity callback to the `ffi::sqlite3_module` struct:

```rust
// In create_module function, find the sqlite3_module struct initialization
// Add this field (after xRollbackTo, before the closing brace):
        #[cfg(sqlite_version_check(3, 44, 0))]
        xIntegrity: Some(rust_integrity::<T>),
        #[cfg(not(sqlite_version_check(3, 44, 0)))]
        xIntegrity: None,
```

And add this function before `create_module`:

```rust
#[cfg(feature = "vtab")]
unsafe extern "C" fn rust_integrity<T: CreateVTab>(
    vtab: *mut ffi::sqlite3_vtab,
    schema: *const c_char,
    tab_name: *const c_char,
    flags: c_int,
    pz_err: *mut *mut c_char,
) -> c_int {
    use std::ffi::CStr;

    let vtab = vtab as *mut T::VTab;
    let schema_str = match CStr::from_ptr(schema).to_str() {
        Ok(s) => s,
        Err(_) => return ffi::SQLITE_ERROR,
    };
    let table_str = match CStr::from_ptr(tab_name).to_str() {
        Ok(s) => s,
        Err(_) => return ffi::SQLITE_ERROR,
    };

    match (*vtab).integrity(schema_str, table_str, flags) {
        Ok(None) => ffi::SQLITE_OK,
        Ok(Some(err_msg)) => {
            let c_err = match std::ffi::CString::new(err_msg) {
                Ok(s) => s,
                Err(_) => return ffi::SQLITE_ERROR,
            };
            *pz_err = ffi::sqlite3_mprintf(c_str!("%s"), c_err.as_ptr());
            ffi::SQLITE_ERROR
        }
        Err(_) => ffi::SQLITE_ERROR,
    }
}
```

## Test

```bash
cargo build --features "bundled vtab blob functions"
cargo test --features "bundled vtab blob functions"
cargo clippy --features "bundled vtab blob functions"
```

## Commit and Push

```bash
git add -A
git commit -m "Add virtual table enhancements: operator registration, DB handle access, integrity check

- Add Connection::overload_function() for custom operators (e.g., MATCH)
- Add Context::get_connection() for stateful scalar functions
- Add VTab::integrity() for PRAGMA integrity_check support (SQLite 3.44+)

These features enable full FTS-like virtual table implementations in Rust."

git push origin custom-vtab-features
```

## Test with sqlite-vec-hnsw

```bash
cd ~/open_dev/sqlite-vec-hnsw
cargo clean
cargo update -p rusqlite
cargo build
cargo test
```

If all tests pass, you're ready to use the features!
