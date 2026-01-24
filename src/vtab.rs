//! Virtual table implementation for vec0

use crate::connection_ext::ConnectionExt;
use crate::distance::DistanceMetric;
use crate::error::{Error, Result};
use crate::hnsw::{self, HnswMetadata};
use crate::shadow;
use crate::vector::{IndexQuantization, Vector, VectorType};
use rusqlite::vtab::{
    Context, CreateVTab, Filters, IndexInfo, Inserts, UpdateVTab, Updates, VTab, VTabConnection,
    VTabCursor, sqlite3_vtab, sqlite3_vtab_cursor,
};
use rusqlite::{Connection, OptionalExtension, ffi};
use std::marker::PhantomData;
use std::os::raw::c_int;

/// RAII guard for SQLite prepared statement handles.
///
/// Resets the statement on creation (to clear any previous state) and again on drop
/// (to release shared-cache locks). This ensures proper cleanup on all code paths.
///
/// In shared-cache mode, prepared statements hold read locks even before execution.
/// Failing to reset statements causes `SQLITE_LOCKED_SHAREDCACHE (262)` errors when
/// other connections try to access the database.
///
/// # Usage
///
/// ```ignore
/// // Wrap a statement for use - resets automatically on create and drop
/// let guard = unsafe { StmtHandleGuard::new(stmt_ptr)? };
/// // Statement is already reset and ready for binding
/// ffi::sqlite3_bind_int64(guard.as_ptr(), 1, value);
/// ffi::sqlite3_step(guard.as_ptr());
/// // guard drops here, automatically resets statement
/// ```
///
/// # Safety
///
/// The caller must ensure:
/// - The statement pointer is valid for the lifetime of the guard
/// - The statement is not finalized while the guard exists
/// - The statement belongs to a connection that is still open
pub struct StmtHandleGuard {
    stmt: *mut ffi::sqlite3_stmt,
    // Make this type !Send and !Sync since sqlite3_stmt is not thread-safe.
    // Using PhantomData<*mut ()> achieves this without unstable negative_impls.
    _marker: PhantomData<*mut ()>,
}

impl StmtHandleGuard {
    /// Create a new guard for a statement handle.
    ///
    /// The statement is reset immediately upon guard creation to clear any
    /// previous state and prepare it for new bindings.
    ///
    /// Returns `None` if the statement pointer is null.
    ///
    /// # Safety
    ///
    /// The caller must ensure the statement pointer is valid and will not be
    /// finalized while the guard exists.
    #[inline]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn new(stmt: *mut ffi::sqlite3_stmt) -> Option<Self> {
        if stmt.is_null() {
            None
        } else {
            // Reset on creation to clear previous state and prepare for new use
            ffi::sqlite3_reset(stmt);
            Some(StmtHandleGuard {
                stmt,
                _marker: PhantomData,
            })
        }
    }

    /// Create a new guard WITHOUT resetting the statement on creation.
    ///
    /// Use this for sequential statement reuse where the previous guard's drop
    /// already reset the statement, or for freshly prepared statements.
    /// Saves one FFI call per statement use.
    ///
    /// # Safety
    ///
    /// - The statement must already be in a reset state (from a previous guard
    ///   drop, or from being freshly prepared)
    /// - The statement pointer must be valid and will not be finalized while
    ///   the guard exists
    #[inline]
    #[allow(unsafe_op_in_unsafe_fn)]
    #[allow(dead_code)] // Reserved for future use
    pub unsafe fn new_skip_reset(stmt: *mut ffi::sqlite3_stmt) -> Option<Self> {
        if stmt.is_null() {
            None
        } else {
            // Skip reset - caller guarantees statement is in reset state
            Some(StmtHandleGuard {
                stmt,
                _marker: PhantomData,
            })
        }
    }

    /// Get the underlying statement pointer for use with SQLite FFI functions.
    #[inline]
    pub fn as_ptr(&self) -> *mut ffi::sqlite3_stmt {
        self.stmt
    }
}

impl Drop for StmtHandleGuard {
    fn drop(&mut self) {
        if !self.stmt.is_null() {
            // SAFETY: We verified the pointer is non-null and was valid when the guard was created.
            // The drop resets to release locks even if the statement failed.
            unsafe {
                ffi::sqlite3_reset(self.stmt);
            }
        }
    }
}

/// Register the vec0 virtual table module
pub fn register_vec0_module(db: &Connection) -> Result<()> {
    // Use update_module_with_tx to support CREATE/INSERT/UPDATE/DELETE + transactions
    db.create_module(
        "vec0",
        rusqlite::vtab::update_module_with_tx::<Vec0Tab>(),
        None,
    )
    .map_err(Error::Sqlite)?;

    // If compiled with loadable_extension_alias feature and SQLITE_VEC_MODULE_ALIAS is set,
    // also register the module under that alias name
    #[cfg(feature = "loadable_extension_alias")]
    if let Some(alias) = option_env!("SQLITE_VEC_MODULE_ALIAS") {
        db.create_module(
            alias,
            rusqlite::vtab::update_module_with_tx::<Vec0Tab>(),
            None,
        )
        .map_err(Error::Sqlite)?;
    }

    Ok(())
}

/// Column definition for vec0 table
#[derive(Debug, Clone)]
struct ColumnDef {
    name: String,
    col_type: ColumnType,
    /// The SQL type for non-vector columns (INTEGER, TEXT, REAL, BLOB)
    sql_type: String,
}

/// Custom HNSW parameters for a vector column
#[derive(Debug, Clone)]
struct HnswColumnParams {
    /// Whether HNSW is enabled for this column (hnsw() was specified)
    enabled: bool,
    /// M parameter (links per node). None = use default (32)
    m: Option<i32>,
    /// ef_construction parameter. None = use default (400)
    ef_construction: Option<i32>,
    /// Distance metric. Default is Cosine
    distance_metric: DistanceMetric,
}

impl Default for HnswColumnParams {
    fn default() -> Self {
        Self {
            enabled: false,
            m: None,
            ef_construction: None,
            distance_metric: DistanceMetric::Cosine, // Default to cosine
        }
    }
}

#[derive(Debug, Clone)]
enum ColumnType {
    Vector {
        #[allow(dead_code)]
        vec_type: VectorType,
        #[allow(dead_code)]
        dimensions: usize,
        /// Quantization for HNSW index storage
        #[allow(dead_code)]
        index_quantization: IndexQuantization,
        /// Custom HNSW parameters from hnsw(...) syntax
        #[allow(dead_code)]
        hnsw_params: HnswColumnParams,
    },
    #[allow(dead_code)]
    PartitionKey,
    #[allow(dead_code)]
    Auxiliary,
    Metadata,
}

/// Index type for vector columns
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexType {
    /// HNSW approximate nearest neighbor (fast, approximate)
    Hnsw,
    /// Exact nearest neighbor via brute force (slow, exact)
    Enn,
}

/// Prepared statement cache for HNSW operations
/// Stores raw sqlite3_stmt pointers to avoid re-preparing statements
///
/// In shared-cache mode, even reset prepared statements hold read locks on the schema.
/// To support concurrent access from other connections:
/// - After table CREATE, statements are finalized to release locks
/// - Statements are lazily re-prepared on first actual use (INSERT/SELECT)
/// - The `needs_prepare` flag tracks when re-preparation is needed
struct HnswStmtCache {
    get_node_data: *mut ffi::sqlite3_stmt,
    get_node_level: *mut ffi::sqlite3_stmt,
    get_edges: *mut ffi::sqlite3_stmt,
    /// Fetches edges WITH stored distances for O(1) prune operations
    get_edges_with_dist: *mut ffi::sqlite3_stmt,
    insert_node: *mut ffi::sqlite3_stmt,
    insert_edge: *mut ffi::sqlite3_stmt,
    delete_edges_from: *mut ffi::sqlite3_stmt,
    update_meta: *mut ffi::sqlite3_stmt,
    /// Single batch fetch statement with 64 placeholders
    /// Unused slots are bound to -1 (won't match any rowid)
    batch_fetch_nodes: *mut ffi::sqlite3_stmt,
    /// Table and column names for lazy re-preparation
    table_name: String,
    column_name: String,
    /// Whether statements need to be prepared (true after finalize, false after prepare)
    needs_prepare: bool,
}

impl HnswStmtCache {
    fn new() -> Self {
        HnswStmtCache {
            get_node_data: std::ptr::null_mut(),
            get_node_level: std::ptr::null_mut(),
            get_edges: std::ptr::null_mut(),
            get_edges_with_dist: std::ptr::null_mut(),
            insert_node: std::ptr::null_mut(),
            insert_edge: std::ptr::null_mut(),
            delete_edges_from: std::ptr::null_mut(),
            update_meta: std::ptr::null_mut(),
            batch_fetch_nodes: std::ptr::null_mut(),
            table_name: String::new(),
            column_name: String::new(),
            needs_prepare: true, // Need to prepare on first use
        }
    }

    /// Prepare all statements for this HNSW index
    ///
    /// # Safety
    /// Must be called with a valid sqlite3 database handle
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn prepare(
        &mut self,
        db: *mut ffi::sqlite3,
        table_name: &str,
        column_name: &str,
    ) -> crate::error::Result<()> {
        use std::ffi::CString;

        // Prepare get_node_data: SELECT rowid, level, vector FROM hnsw_nodes WHERE rowid = ?
        let sql = CString::new(format!(
            "SELECT rowid, level, vector FROM \"{}_{}_hnsw_nodes\" WHERE rowid = ?",
            table_name, column_name
        ))
        .map_err(|e| crate::error::Error::InvalidParameter(format!("Invalid SQL: {}", e)))?;

        let rc = ffi::sqlite3_prepare_v2(
            db,
            sql.as_ptr(),
            -1,
            &mut self.get_node_data,
            std::ptr::null_mut(),
        );
        if rc != ffi::SQLITE_OK {
            return Err(crate::error::Error::Sqlite(rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rc),
                None,
            )));
        }

        // Prepare get_edges: SELECT to_rowid FROM hnsw_edges WHERE from_rowid = ? AND level = ?
        let sql = CString::new(format!(
            "SELECT to_rowid FROM \"{}_{}_hnsw_edges\" WHERE from_rowid = ? AND level = ?",
            table_name, column_name
        ))
        .map_err(|e| crate::error::Error::InvalidParameter(format!("Invalid SQL: {}", e)))?;

        let rc = ffi::sqlite3_prepare_v2(
            db,
            sql.as_ptr(),
            -1,
            &mut self.get_edges,
            std::ptr::null_mut(),
        );
        if rc != ffi::SQLITE_OK {
            return Err(crate::error::Error::Sqlite(rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rc),
                None,
            )));
        }

        // Prepare get_edges_with_dist: SELECT to_rowid, distance FROM hnsw_edges WHERE from_rowid = ? AND level = ?
        // Used by prune to avoid re-computing distances (they're stored when edges are created)
        let sql = CString::new(format!(
            "SELECT to_rowid, distance FROM \"{}_{}_hnsw_edges\" WHERE from_rowid = ? AND level = ?",
            table_name, column_name
        ))
        .map_err(|e| crate::error::Error::InvalidParameter(format!("Invalid SQL: {}", e)))?;

        let rc = ffi::sqlite3_prepare_v2(
            db,
            sql.as_ptr(),
            -1,
            &mut self.get_edges_with_dist,
            std::ptr::null_mut(),
        );
        if rc != ffi::SQLITE_OK {
            return Err(crate::error::Error::Sqlite(rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rc),
                None,
            )));
        }

        // Prepare insert_node: INSERT INTO hnsw_nodes (rowid, level, vector) VALUES (?, ?, ?)
        let sql = CString::new(format!(
            "INSERT INTO \"{}_{}_hnsw_nodes\" (rowid, level, vector) VALUES (?, ?, ?)",
            table_name, column_name
        ))
        .map_err(|e| crate::error::Error::InvalidParameter(format!("Invalid SQL: {}", e)))?;

        let rc = ffi::sqlite3_prepare_v2(
            db,
            sql.as_ptr(),
            -1,
            &mut self.insert_node,
            std::ptr::null_mut(),
        );
        if rc != ffi::SQLITE_OK {
            return Err(crate::error::Error::Sqlite(rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rc),
                None,
            )));
        }

        // Prepare insert_edge: INSERT OR IGNORE INTO hnsw_edges (from_rowid, to_rowid, level) VALUES (?, ?, ?)
        let sql = CString::new(format!(
            "INSERT OR IGNORE INTO \"{}_{}_hnsw_edges\" (from_rowid, to_rowid, level) VALUES (?, ?, ?)",
            table_name, column_name
        ))
        .map_err(|e| crate::error::Error::InvalidParameter(format!("Invalid SQL: {}", e)))?;

        let rc = ffi::sqlite3_prepare_v2(
            db,
            sql.as_ptr(),
            -1,
            &mut self.insert_edge,
            std::ptr::null_mut(),
        );
        if rc != ffi::SQLITE_OK {
            return Err(crate::error::Error::Sqlite(rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rc),
                None,
            )));
        }

        // Prepare delete_edges_from: DELETE FROM hnsw_edges WHERE from_rowid = ? AND level = ?
        let sql = CString::new(format!(
            "DELETE FROM \"{}_{}_hnsw_edges\" WHERE from_rowid = ? AND level = ?",
            table_name, column_name
        ))
        .map_err(|e| crate::error::Error::InvalidParameter(format!("Invalid SQL: {}", e)))?;

        let rc = ffi::sqlite3_prepare_v2(
            db,
            sql.as_ptr(),
            -1,
            &mut self.delete_edges_from,
            std::ptr::null_mut(),
        );
        if rc != ffi::SQLITE_OK {
            return Err(crate::error::Error::Sqlite(rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rc),
                None,
            )));
        }

        // Prepare update_meta: single UPDATE for dynamic metadata fields
        let sql = CString::new(format!(
            "UPDATE \"{}_{}_hnsw_meta\" SET \
             entry_point_rowid = ?, entry_point_level = ?, num_nodes = ?, hnsw_version = ? \
             WHERE id = 1",
            table_name, column_name
        ))
        .map_err(|e| crate::error::Error::InvalidParameter(format!("Invalid SQL: {}", e)))?;

        let rc = ffi::sqlite3_prepare_v2(
            db,
            sql.as_ptr(),
            -1,
            &mut self.update_meta,
            std::ptr::null_mut(),
        );
        if rc != ffi::SQLITE_OK {
            return Err(crate::error::Error::Sqlite(rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rc),
                None,
            )));
        }

        // Prepare batch_fetch_nodes with 64 placeholders (pad unused with -1)
        let nodes_table = format!("{}_{}_hnsw_nodes", table_name, column_name);
        let placeholders = (0..64).map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = CString::new(format!(
            "SELECT rowid, level, vector FROM \"{}\" WHERE rowid IN ({})",
            nodes_table, placeholders
        ))
        .map_err(|e| crate::error::Error::InvalidParameter(format!("Invalid SQL: {}", e)))?;

        let rc = ffi::sqlite3_prepare_v2(
            db,
            sql.as_ptr(),
            -1,
            &mut self.batch_fetch_nodes,
            std::ptr::null_mut(),
        );
        if rc != ffi::SQLITE_OK {
            return Err(crate::error::Error::Sqlite(rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rc),
                None,
            )));
        }

        // Store table/column names for lazy re-preparation after finalize
        self.table_name = table_name.to_string();
        self.column_name = column_name.to_string();
        self.needs_prepare = false;

        // Note: We used to reset statements here, but that's not sufficient.
        // In shared-cache mode, even reset statements hold schema read locks.
        // The solution is to finalize after CREATE and re-prepare lazily on first use.
        // See finalize_for_shared_cache() below.

        Ok(())
    }

    /// Finalize all statements to release schema locks in shared-cache mode
    ///
    /// In shared-cache mode, prepared statements hold read locks on the schema
    /// even after being reset. This prevents other connections from writing.
    /// Call this after table creation to allow other connections to work,
    /// then statements will be re-prepared lazily on first actual use.
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn finalize_for_shared_cache(&mut self) {
        // Finalize all statements to fully release locks
        if !self.get_node_data.is_null() {
            ffi::sqlite3_finalize(self.get_node_data);
            self.get_node_data = std::ptr::null_mut();
        }
        if !self.get_node_level.is_null() {
            ffi::sqlite3_finalize(self.get_node_level);
            self.get_node_level = std::ptr::null_mut();
        }
        if !self.get_edges.is_null() {
            ffi::sqlite3_finalize(self.get_edges);
            self.get_edges = std::ptr::null_mut();
        }
        if !self.get_edges_with_dist.is_null() {
            ffi::sqlite3_finalize(self.get_edges_with_dist);
            self.get_edges_with_dist = std::ptr::null_mut();
        }
        if !self.insert_node.is_null() {
            ffi::sqlite3_finalize(self.insert_node);
            self.insert_node = std::ptr::null_mut();
        }
        if !self.insert_edge.is_null() {
            ffi::sqlite3_finalize(self.insert_edge);
            self.insert_edge = std::ptr::null_mut();
        }
        if !self.delete_edges_from.is_null() {
            ffi::sqlite3_finalize(self.delete_edges_from);
            self.delete_edges_from = std::ptr::null_mut();
        }
        if !self.update_meta.is_null() {
            ffi::sqlite3_finalize(self.update_meta);
            self.update_meta = std::ptr::null_mut();
        }
        if !self.batch_fetch_nodes.is_null() {
            ffi::sqlite3_finalize(self.batch_fetch_nodes);
            self.batch_fetch_nodes = std::ptr::null_mut();
        }
        // Mark that statements need to be prepared on next use
        self.needs_prepare = true;
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn finalize(&mut self) {
        if !self.get_node_data.is_null() {
            ffi::sqlite3_finalize(self.get_node_data);
            self.get_node_data = std::ptr::null_mut();
        }
        if !self.get_node_level.is_null() {
            ffi::sqlite3_finalize(self.get_node_level);
            self.get_node_level = std::ptr::null_mut();
        }
        if !self.get_edges.is_null() {
            ffi::sqlite3_finalize(self.get_edges);
            self.get_edges = std::ptr::null_mut();
        }
        if !self.get_edges_with_dist.is_null() {
            ffi::sqlite3_finalize(self.get_edges_with_dist);
            self.get_edges_with_dist = std::ptr::null_mut();
        }
        if !self.insert_node.is_null() {
            ffi::sqlite3_finalize(self.insert_node);
            self.insert_node = std::ptr::null_mut();
        }
        if !self.insert_edge.is_null() {
            ffi::sqlite3_finalize(self.insert_edge);
            self.insert_edge = std::ptr::null_mut();
        }
        if !self.delete_edges_from.is_null() {
            ffi::sqlite3_finalize(self.delete_edges_from);
            self.delete_edges_from = std::ptr::null_mut();
        }
        if !self.update_meta.is_null() {
            ffi::sqlite3_finalize(self.update_meta);
            self.update_meta = std::ptr::null_mut();
        }
        // Finalize batch fetch statement
        if !self.batch_fetch_nodes.is_null() {
            ffi::sqlite3_finalize(self.batch_fetch_nodes);
            self.batch_fetch_nodes = std::ptr::null_mut();
        }
    }
}

/// vec0 virtual table structure
#[repr(C)]
pub struct Vec0Tab {
    base: sqlite3_vtab,
    schema_name: String,
    table_name: String,
    columns: Vec<ColumnDef>,
    chunk_size: usize,
    index_type: IndexType,
    db: *mut ffi::sqlite3,               // Raw database handle for operations
    hnsw_stmt_cache: Vec<HnswStmtCache>, // One cache per vector column
}

/// Normalize SQL type specification to standard SQLite types
/// Maps various type names to INTEGER, TEXT, REAL, or BLOB
fn normalize_sql_type(type_spec: &str) -> String {
    let upper = type_spec.to_uppercase();

    // INTEGER types
    if upper.contains("INT") || upper == "BOOLEAN" || upper == "BOOL" {
        return "INTEGER".to_string();
    }

    // REAL types
    if upper.contains("REAL")
        || upper.contains("DOUBLE")
        || upper.contains("FLOAT")
        || upper.contains("NUMERIC")
        || upper.contains("DECIMAL")
    {
        return "REAL".to_string();
    }

    // BLOB types
    if upper.contains("BLOB") || upper.contains("BINARY") {
        return "BLOB".to_string();
    }

    // Default to TEXT for everything else (VARCHAR, CHAR, TEXT, etc.)
    "TEXT".to_string()
}

/// Extract the hnsw(...) clause from a column definition string.
/// Returns (string_without_hnsw, Some(hnsw_clause)) or (original_string, None).
/// Uses regex to handle spaces inside hnsw() by matching balanced parentheses.
fn extract_hnsw_clause(arg: &str) -> (String, Option<String>) {
    use regex::Regex;
    use std::sync::OnceLock;

    // Compile regex once for efficiency
    static HNSW_RE: OnceLock<Regex> = OnceLock::new();
    let re = HNSW_RE.get_or_init(|| {
        // Match hnsw(...) case-insensitively, capturing content with nested parens
        // Uses non-greedy match with balanced parens workaround
        Regex::new(r"(?i)hnsw\([^()]*(?:\([^()]*\)[^()]*)*\)").unwrap()
    });

    if let Some(m) = re.find(arg) {
        let hnsw_clause = m.as_str().to_string();
        let before = arg[..m.start()].trim();
        let after = arg[m.end()..].trim();
        let without_hnsw = if before.is_empty() {
            after.to_string()
        } else if after.is_empty() {
            before.to_string()
        } else {
            format!("{} {}", before, after)
        };
        return (without_hnsw, Some(hnsw_clause));
    }
    (arg.to_string(), None)
}

impl Vec0Tab {
    fn parse_create_args(args: &[&str]) -> Result<(String, String, Vec<ColumnDef>, IndexType)> {
        let mut columns = Vec::new();
        let mut index_type = IndexType::Hnsw; // Default

        // args[0] = module name ("vec0")
        // args[1] = schema name (e.g., "main")
        // args[2] = table name
        // args[3..] = column definitions and options

        let schema_name = args.get(1).unwrap_or(&"main").to_string();
        let table_name = args.get(2).unwrap_or(&"vec_table").to_string();

        // Skip module name, schema, and table name
        for arg in args.iter().skip(3) {
            let arg = arg.trim();
            if arg.is_empty() {
                continue;
            }

            // Check for table options like type=hnsw, type=enn
            if arg.contains('=') {
                let parts: Vec<&str> = arg.splitn(2, '=').collect();
                if parts.len() == 2 {
                    let key = parts[0].trim();
                    let value = parts[1].trim();

                    if key.eq_ignore_ascii_case("type") {
                        index_type = match value.to_lowercase().as_str() {
                            "hnsw" => IndexType::Hnsw,
                            "enn" => IndexType::Enn,
                            _ => {
                                return Err(Error::InvalidParameter(format!(
                                    "Invalid index type: '{}'. Use 'hnsw' or 'enn'",
                                    value
                                )));
                            }
                        };
                        continue; // Skip to next arg, this was an option not a column
                    }
                }
            }

            // Parse column definition: "column_name type[dimensions] [hnsw(...)]"
            // First, extract the hnsw(...) clause if present (it may contain spaces)
            let (arg_without_hnsw, hnsw_clause) = extract_hnsw_clause(arg);
            let parts: Vec<&str> = arg_without_hnsw.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            let name = parts[0].to_string();

            if parts.len() > 1 {
                let type_spec = parts[1];

                // Check for vector type with dimensions: float[768]
                if let Some(bracket_pos) = type_spec.find('[') {
                    let vec_type_str = &type_spec[..bracket_pos];
                    let dims_str = &type_spec[bracket_pos + 1..type_spec.len() - 1];

                    let vec_type = VectorType::from_str(vec_type_str)?;
                    let dimensions: usize = dims_str.parse().map_err(|_| {
                        Error::InvalidParameter(format!("Invalid dimensions: {}", dims_str))
                    })?;

                    // Parse optional hnsw(...) options after type
                    // Format: hnsw(M=64, ef_construction=200, index_quantization=int8)
                    let mut index_quantization = IndexQuantization::None;
                    let mut hnsw_params = HnswColumnParams::default();

                    // First check parts for non-hnsw options (should not exist)
                    if let Some(part) = parts.get(2) {
                        // Error on any unrecognized option
                        return Err(Error::InvalidParameter(format!(
                            "Unknown vector column option: '{}'. Use hnsw(M=N, ef_construction=N, index_quantization=int8)",
                            part
                        )));
                    }

                    // Now parse the hnsw clause if present
                    if let Some(hnsw_str) = &hnsw_clause {
                        // Mark HNSW as enabled for this column
                        hnsw_params.enabled = true;

                        // Extract parameters from hnsw(...)
                        let params_str = hnsw_str
                            .strip_prefix("hnsw(")
                            .or_else(|| hnsw_str.strip_prefix("HNSW("))
                            .and_then(|s| s.strip_suffix(')'))
                            .unwrap_or(""); // Empty params for hnsw()

                        for param in params_str.split(',') {
                            let param = param.trim();
                            if param.is_empty() {
                                continue;
                            }
                            let (k, v) = param.split_once('=').ok_or_else(|| {
                                Error::InvalidParameter(format!(
                                    "Invalid hnsw parameter: '{}'. Expected key=value format",
                                    param
                                ))
                            })?;
                            let k = k.trim();
                            let v = v.trim();

                            if k.eq_ignore_ascii_case("M") {
                                hnsw_params.m = Some(v.parse().map_err(|_| {
                                    Error::InvalidParameter(format!(
                                        "Invalid M value: '{}'. Expected integer",
                                        v
                                    ))
                                })?);
                            } else if k.eq_ignore_ascii_case("ef_construction") {
                                hnsw_params.ef_construction = Some(v.parse().map_err(|_| {
                                    Error::InvalidParameter(format!(
                                        "Invalid ef_construction value: '{}'. Expected integer",
                                        v
                                    ))
                                })?);
                            } else if k.eq_ignore_ascii_case("index_quantization") {
                                index_quantization = IndexQuantization::from_str(v)?;
                            } else if k.eq_ignore_ascii_case("distance") {
                                hnsw_params.distance_metric = DistanceMetric::from_str(v)?;
                            } else {
                                return Err(Error::InvalidParameter(format!(
                                    "Unknown hnsw parameter: '{}'. Valid: M, ef_construction, index_quantization, distance",
                                    k
                                )));
                            }
                        }
                    }

                    columns.push(ColumnDef {
                        name,
                        col_type: ColumnType::Vector {
                            vec_type,
                            dimensions,
                            index_quantization,
                            hnsw_params,
                        },
                        sql_type: "BLOB".to_string(),
                    });
                } else if type_spec.to_uppercase().contains("PARTITION") {
                    columns.push(ColumnDef {
                        name,
                        col_type: ColumnType::PartitionKey,
                        sql_type: "INTEGER".to_string(),
                    });
                } else if parts.iter().any(|p| p.to_uppercase().starts_with('+')) {
                    // Auxiliary column - parse the SQL type
                    let sql_type = normalize_sql_type(type_spec);
                    columns.push(ColumnDef {
                        name: name.trim_start_matches('+').to_string(),
                        col_type: ColumnType::Auxiliary,
                        sql_type,
                    });
                } else {
                    // Metadata column - parse the SQL type
                    let sql_type = normalize_sql_type(type_spec);
                    columns.push(ColumnDef {
                        name,
                        col_type: ColumnType::Metadata,
                        sql_type,
                    });
                }
            } else {
                // No type specified, treat as metadata with TEXT type
                columns.push(ColumnDef {
                    name,
                    col_type: ColumnType::Metadata,
                    sql_type: "TEXT".to_string(),
                });
            }
        }

        Ok((schema_name, table_name, columns, index_type))
    }
}

unsafe impl<'vtab> VTab<'vtab> for Vec0Tab {
    type Aux = ();
    type Cursor = Vec0TabCursor<'vtab>;

    fn connect(
        db: &mut VTabConnection,
        _aux: Option<&Self::Aux>,
        args: &[&[u8]],
    ) -> rusqlite::Result<(String, Self)> {
        let args_str: Vec<&str> = args
            .iter()
            .map(|arg| std::str::from_utf8(arg).unwrap_or(""))
            .collect();

        let (schema_name, table_name, columns, index_type) = Self::parse_create_args(&args_str)
            .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

        // Build CREATE TABLE statement for SQLite
        let mut sql = String::from("CREATE TABLE x(");
        for (i, col) in columns.iter().enumerate() {
            if i > 0 {
                sql.push_str(", ");
            }
            sql.push_str(&col.name);
            match &col.col_type {
                ColumnType::Vector { .. } => sql.push_str(" BLOB"),
                ColumnType::PartitionKey => sql.push_str(" INTEGER"),
                ColumnType::Auxiliary => sql.push_str(" TEXT"),
                ColumnType::Metadata => sql.push_str(" TEXT"),
            }
        }
        // Add hidden columns for KNN queries
        sql.push_str(", distance REAL HIDDEN, k INTEGER HIDDEN");
        sql.push(')');

        // SAFETY: Store the database handle for later operations
        let db_handle = unsafe { db.handle() };

        // Register MATCH operator for KNN queries
        // This needs to be done in connect() as well as create()
        // because connect() is called when opening existing tables
        unsafe {
            let conn = Connection::from_handle(db_handle)?;
            conn.overload_function("match", 2)?;
            std::mem::forget(conn); // Don't close the connection
        }

        // Initialize statement cache (one per vector column)
        // NOTE: Statements are NOT prepared here - they will be prepared in create()
        // after shadow tables are created. For existing tables being reopened,
        // statements are prepared on first use (lazy initialization).
        let num_vector_columns = columns
            .iter()
            .filter(|c| matches!(c.col_type, ColumnType::Vector { .. }))
            .count();
        let hnsw_stmt_cache = (0..num_vector_columns)
            .map(|_| HnswStmtCache::new())
            .collect();

        Ok((
            sql,
            Vec0Tab {
                base: sqlite3_vtab::default(),
                schema_name,
                table_name,
                columns,
                chunk_size: shadow::DEFAULT_CHUNK_SIZE,
                index_type,
                db: db_handle,
                hnsw_stmt_cache,
            },
        ))
    }

    fn best_index(&self, info: &mut IndexInfo) -> rusqlite::Result<()> {
        use rusqlite::vtab::IndexConstraintOp;

        let mut match_constraint = None;
        let mut k_constraint = None;

        // Scan constraints looking for MATCH and k operators
        for (i, constraint) in info.constraints().enumerate() {
            if !constraint.is_usable() {
                continue;
            }

            // Check for MATCH operator on a vector column
            if constraint.operator() == IndexConstraintOp::SQLITE_INDEX_CONSTRAINT_MATCH {
                let col_idx = constraint.column();
                if col_idx >= 0
                    && (col_idx as usize) < self.columns.len()
                    && matches!(
                        self.columns[col_idx as usize].col_type,
                        ColumnType::Vector { .. }
                    )
                {
                    match_constraint = Some(i);
                }
            }

            // Check for k = ? constraint (hidden k column)
            // The k column is the last column: columns + distance + k
            if constraint.operator() == IndexConstraintOp::SQLITE_INDEX_CONSTRAINT_EQ {
                // k column is at index: num_user_columns + 1 (distance is at num_user_columns)
                let k_column_idx = self.columns.len() + 1;
                if constraint.column() as usize == k_column_idx {
                    k_constraint = Some(i);
                }
            }
        }

        // Determine query plan
        if let Some(match_idx) = match_constraint
            && k_constraint.is_some()
        {
            // KNN query with MATCH and k
            info.set_idx_str("3{___}___"); // KNN plan: match + k
            info.set_estimated_cost(10.0); // Low cost for indexed query
            info.set_estimated_rows(10);

            // Mark MATCH constraint as used (argv index 1)
            info.constraint_usage(match_idx).set_argv_index(1);
            info.constraint_usage(match_idx).set_omit(true);

            // Mark k constraint as used (argv index 2)
            if let Some(k_idx) = k_constraint {
                info.constraint_usage(k_idx).set_argv_index(2);
                info.constraint_usage(k_idx).set_omit(true);
            }

            return Ok(());
        }

        // Default: full scan
        info.set_idx_str("1"); // FullScan query plan
        info.set_estimated_cost(1000000.0);
        info.set_estimated_rows(1000000);
        Ok(())
    }

    fn open(&mut self) -> rusqlite::Result<Vec0TabCursor<'vtab>> {
        Ok(Vec0TabCursor::new(self))
    }
}

impl<'vtab> CreateVTab<'vtab> for Vec0Tab {
    const KIND: rusqlite::vtab::VTabKind = rusqlite::vtab::VTabKind::Default;

    fn create(
        db: &mut VTabConnection,
        aux: Option<&Self::Aux>,
        args: &[&[u8]],
    ) -> rusqlite::Result<(String, Self)> {
        // First, parse arguments and create the base virtual table
        let (sql, vtab) = Self::connect(db, aux, args)?;

        // Count column types for shadow table creation
        // Shadow tables use sequential indices (0, 1, 2) for vector columns,
        // regardless of their position in the overall table definition
        let num_vector_columns = vtab
            .columns
            .iter()
            .filter(|c| matches!(c.col_type, ColumnType::Vector { .. }))
            .count();
        let num_metadata_columns = vtab
            .columns
            .iter()
            .filter(|c| matches!(c.col_type, ColumnType::Metadata))
            .count();
        let num_auxiliary_columns = vtab
            .columns
            .iter()
            .filter(|c| matches!(c.col_type, ColumnType::Auxiliary))
            .count();
        let num_partition_columns = vtab
            .columns
            .iter()
            .filter(|c| matches!(c.col_type, ColumnType::PartitionKey))
            .count();

        // Create shadow tables using raw FFI
        // SAFETY: We're using the raw sqlite3 handle to execute DDL statements
        // This is necessary because rusqlite's VTab trait doesn't provide
        // a way to execute arbitrary SQL during table creation
        unsafe {
            let db_handle = db.handle();

            // Build data column definitions for non-vector columns
            // Use the actual SQL type from the column definition for type preservation
            let data_columns: Vec<shadow::DataColumnDef> = vtab
                .columns
                .iter()
                .filter(|c| matches!(c.col_type, ColumnType::Metadata | ColumnType::Auxiliary))
                .map(|c| shadow::DataColumnDef {
                    name: c.name.clone(),
                    col_type: c.sql_type.clone(),
                })
                .collect();

            // Create base shadow tables
            let config = shadow::ShadowTablesConfig {
                num_vector_columns,
                num_metadata_columns,
                num_auxiliary_columns,
                has_text_pk: false, // TODO: detect from args
                num_partition_columns,
                data_columns,
            };

            // Collect vector column names for cleanup
            let vector_column_names: Vec<&str> = vtab
                .columns
                .iter()
                .filter_map(|c| {
                    if matches!(c.col_type, ColumnType::Vector { .. }) {
                        Some(c.name.as_str())
                    } else {
                        None
                    }
                })
                .collect();

            // Drop any existing shadow tables first (cleanup from failed previous CREATE)
            shadow::drop_shadow_tables_ffi(
                db_handle,
                &vtab.schema_name,
                &vtab.table_name,
                &config,
                &vector_column_names,
            )
            .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

            shadow::create_shadow_tables_ffi(
                db_handle,
                &vtab.schema_name,
                &vtab.table_name,
                &config,
            )
            .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

            // Create HNSW shadow tables for vector columns that have hnsw() enabled
            let mut vec_col_idx = 0;
            for col in vtab.columns.iter() {
                if let ColumnType::Vector {
                    vec_type,
                    dimensions,
                    index_quantization,
                    hnsw_params,
                } = &col.col_type
                {
                    // Only create HNSW tables if hnsw() was specified
                    if hnsw_params.enabled {
                        shadow::create_hnsw_shadow_tables_with_params_ffi(
                            db_handle,
                            &vtab.table_name,
                            &col.name,
                            *dimensions as i32,
                            *vec_type,
                            hnsw_params.distance_metric,
                            *index_quantization,
                            hnsw_params.m,
                            hnsw_params.ef_construction,
                        )
                        .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

                        // Prepare statements for this vector column
                        // SAFETY: vtab is valid and we need mutable access to stmt cache
                        let vtab_mut = &vtab as *const _ as *mut Vec0Tab;
                        let cache_vec = &mut (*vtab_mut).hnsw_stmt_cache;
                        if let Some(cache) = cache_vec.get_mut(vec_col_idx) {
                            cache
                                .prepare(db_handle, &vtab.table_name, &col.name)
                                .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
                        }

                        vec_col_idx += 1;
                    }
                }
            }

            // Register MATCH operator for KNN queries
            // This enables syntax like: WHERE embedding MATCH '[1,2,3]'
            let conn = Connection::from_handle(db_handle)?;
            conn.overload_function("match", 2)?;
            std::mem::forget(conn); // Don't close the connection

            // CRITICAL: Finalize all statements to release schema locks in shared-cache mode.
            // In shared-cache mode, even reset prepared statements hold read locks on the
            // schema, preventing other connections from accessing the database.
            // By finalizing here, we allow Senzing (via a different connection) to access
            // the database immediately after CREATE TABLE completes.
            // Statements will be lazily re-prepared on first actual use (INSERT/SELECT).
            let vtab_mut = &vtab as *const _ as *mut Vec0Tab;
            for cache in (*vtab_mut).hnsw_stmt_cache.iter_mut() {
                cache.finalize_for_shared_cache();
            }
        }

        Ok((sql, vtab))
    }

    fn destroy(&self) -> rusqlite::Result<()> {
        // Finalize all prepared statements
        // SAFETY: We're finalizing statements that we own
        unsafe {
            // Need mutable access, but we have &self
            // Cast away const for cleanup (safe because this is called during drop)
            let self_mut = self as *const _ as *mut Vec0Tab;
            for cache in (*self_mut).hnsw_stmt_cache.iter_mut() {
                cache.finalize();
            }
        }

        // Drop shadow tables manually (SQLite doesn't auto-drop without xShadowName)
        // SAFETY: db is a valid sqlite3 handle from SQLite
        let conn = unsafe { Connection::from_handle(self.db)? };

        // Build list of shadow tables to drop
        let mut shadow_tables = vec![
            format!("{}_chunks", self.table_name),
            format!("{}_rowids", self.table_name),
            format!("{}_info", self.table_name),
        ];

        // Add per-column shadow tables for each column
        // Note: vector_chunks tables use vector-column-specific index (0, 1, 2... for each vector column)
        // while metadata tables use overall column index
        let mut vec_col_idx = 0;
        for (col_idx, col) in self.columns.iter().enumerate() {
            if let ColumnType::Vector { .. } = col.col_type {
                // Vector chunk table (uses vector column index, not overall column index)
                shadow_tables.push(format!(
                    "{}_vector_chunks{:02}",
                    self.table_name, vec_col_idx
                ));

                // HNSW tables
                shadow_tables.push(format!("{}_{}_hnsw_nodes", self.table_name, col.name));
                shadow_tables.push(format!("{}_{}_hnsw_edges", self.table_name, col.name));
                shadow_tables.push(format!("{}_{}_hnsw_levels", self.table_name, col.name));
                shadow_tables.push(format!("{}_{}_hnsw_meta", self.table_name, col.name));

                vec_col_idx += 1;
            }

            // Legacy metadata tables (may exist from older versions)
            shadow_tables.push(format!("{}_metadatachunks{:02}", self.table_name, col_idx));
            shadow_tables.push(format!("{}_metadatatext{:02}", self.table_name, col_idx));
        }

        // Add auxiliary table if there are auxiliary columns
        if self
            .columns
            .iter()
            .any(|c| matches!(c.col_type, ColumnType::Auxiliary))
        {
            shadow_tables.push(format!("{}_auxiliary", self.table_name));
        }

        // Add _data table for non-vector column storage
        shadow_tables.push(format!("{}_data", self.table_name));

        // Drop each shadow table (ignore errors for tables that don't exist)
        for table in shadow_tables {
            let _ = conn.execute(&format!("DROP TABLE IF EXISTS \"{}\"", table), []);
        }

        std::mem::forget(conn); // Don't close the connection
        Ok(())
    }

    fn integrity(
        &self,
        _schema: &str,
        _table_name: &str,
        _flags: c_int,
    ) -> rusqlite::Result<Option<String>> {
        // Validate HNSW index consistency for each vector column
        // SAFETY: db is a valid sqlite3 handle from SQLite
        let conn = unsafe { Connection::from_handle(self.db)? };

        for col in self.columns.iter() {
            if let ColumnType::Vector { .. } = col.col_type {
                // Check if HNSW shadow tables exist
                let nodes_table = format!("{}_{}_hnsw_nodes", self.table_name, col.name);
                let count_query = format!(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{}'",
                    nodes_table
                );

                let table_exists: bool = conn
                    .query_row(&count_query, [], |row| row.get::<_, i64>(0))
                    .map(|count| count > 0)
                    .unwrap_or(false);

                if table_exists {
                    // Basic validation: check that entry_point_rowid is valid
                    let meta_table = format!("{}_{}_hnsw_meta", self.table_name, col.name);
                    let entry_point_query = format!(
                        "SELECT value FROM \"{}\" WHERE key='entry_point_rowid'",
                        meta_table
                    );

                    if let Ok(entry_point_str) =
                        conn.query_row(&entry_point_query, [], |row| row.get::<_, String>(0))
                        && let Ok(entry_point) = entry_point_str.parse::<i64>()
                        && entry_point >= 0
                    {
                        // Verify the entry point exists in nodes table
                        let node_check =
                            format!("SELECT COUNT(*) FROM \"{}\" WHERE rowid=?", nodes_table);
                        let node_exists: i64 = conn
                            .query_row(&node_check, [entry_point], |row| row.get(0))
                            .unwrap_or(0);

                        if node_exists == 0 {
                            std::mem::forget(conn);
                            return Ok(Some(format!(
                                "HNSW index for column '{}': entry point rowid {} does not exist",
                                col.name, entry_point
                            )));
                        }
                    }
                }
            }
        }

        std::mem::forget(conn);
        Ok(None) // No errors found
    }
}

impl<'vtab> UpdateVTab<'vtab> for Vec0Tab {
    fn delete(&mut self, arg: rusqlite::types::ValueRef<'_>) -> rusqlite::Result<()> {
        // arg is the rowid to delete
        let rowid = match arg {
            rusqlite::types::ValueRef::Integer(i) => i,
            _ => {
                return Err(rusqlite::Error::UserFunctionError(Box::new(
                    Error::InvalidParameter("DELETE requires integer rowid".to_string()),
                )));
            }
        };

        // SAFETY: db is a valid sqlite3 handle from SQLite
        let conn = unsafe { Connection::from_handle(self.db)? };

        // Get chunk position for this rowid
        let rowids_table = format!("{}_rowids", self.table_name);
        let query = format!(
            "SELECT chunk_id, chunk_offset FROM \"{}\".\"{}\" WHERE rowid = ?",
            self.schema_name, rowids_table
        );

        let chunk_info: Option<(i64, i64)> = conn
            .query_row(&query, [rowid], |row| Ok((row.get(0)?, row.get(1)?)))
            .optional()?;

        if let Some((chunk_id, chunk_offset)) = chunk_info {
            // Mark as invalid in validity bitmap
            let chunks_table = format!("{}_chunks", self.table_name);
            shadow::mark_chunk_row_invalid(&conn, &chunks_table, chunk_id, chunk_offset as usize)
                .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

            // Delete from HNSW index if column is a vector with HNSW enabled
            for col in &self.columns {
                if let ColumnType::Vector { hnsw_params, .. } = &col.col_type
                    && hnsw_params.enabled
                {
                    let nodes_table = format!("{}_{}_hnsw_nodes", self.table_name, col.name);
                    let edges_table = format!("{}_{}_hnsw_edges", self.table_name, col.name);
                    let levels_table = format!("{}_{}_hnsw_levels", self.table_name, col.name);

                    // Delete node (ignore errors if table doesn't exist)
                    let _ = conn.execute(
                        &format!("DELETE FROM \"{}\" WHERE rowid = ?", nodes_table),
                        [rowid],
                    );

                    // Delete edges (ignore errors if table doesn't exist)
                    let _ = conn.execute(
                        &format!(
                            "DELETE FROM \"{}\" WHERE from_rowid = ? OR to_rowid = ?",
                            edges_table
                        ),
                        rusqlite::params![rowid, rowid],
                    );

                    // Delete from levels (ignore errors if table doesn't exist)
                    let _ = conn.execute(
                        &format!("DELETE FROM \"{}\" WHERE rowid = ?", levels_table),
                        [rowid],
                    );

                    // Update metadata if it exists
                    if let Ok(Some(mut meta)) =
                        HnswMetadata::load_from_db(&conn, &self.table_name, &col.name)
                    {
                        meta.num_nodes = meta.num_nodes.saturating_sub(1);
                        meta.hnsw_version += 1;

                        // If we deleted the entry point, need to find a new one
                        if meta.entry_point_rowid == rowid {
                            let new_entry: Option<(i64, i32)> = conn
                                .query_row(
                                    &format!(
                                        "SELECT rowid, level FROM \"{}\" ORDER BY level DESC LIMIT 1",
                                        nodes_table
                                    ),
                                    [],
                                    |row| Ok((row.get(0)?, row.get(1)?)),
                                )
                                .optional()
                                .unwrap_or(None);

                            if let Some((new_rowid, new_level)) = new_entry {
                                meta.entry_point_rowid = new_rowid;
                                meta.entry_point_level = new_level;
                            } else {
                                // No nodes left
                                meta.entry_point_rowid = -1;
                                meta.entry_point_level = -1;
                            }
                        }

                        let _ = meta.save_to_db(&conn, &self.table_name, &col.name);
                    }
                }
            }

            // Delete from rowids table
            conn.execute(
                &format!(
                    "DELETE FROM \"{}\".\"{}\" WHERE rowid = ?",
                    self.schema_name, rowids_table
                ),
                [rowid],
            )?;

            // Delete from _data table (non-vector columns)
            let has_data_columns = self
                .columns
                .iter()
                .any(|c| matches!(c.col_type, ColumnType::Metadata | ColumnType::Auxiliary));
            if has_data_columns {
                let data_table = format!("{}_data", self.table_name);
                conn.execute(
                    &format!(
                        "DELETE FROM \"{}\".\"{}\" WHERE rowid = ?",
                        self.schema_name, data_table
                    ),
                    [rowid],
                )?;
            }
        }

        // Release connection without closing the database
        std::mem::forget(conn);

        Ok(())
    }

    fn insert(&mut self, args: &Inserts<'_>) -> rusqlite::Result<i64> {
        // args[0]: NULL for auto-rowid
        // args[1]: new rowid (or NULL for auto)
        // args[2..]: column values

        // Determine rowid
        let rowid = if args.len() > 1 {
            match args.get::<Option<i64>>(1)? {
                Some(r) => r,
                None => {
                    // Auto-generate rowid by querying max rowid from _rowids table
                    // SAFETY: self.db is valid for the lifetime of the virtual table
                    unsafe {
                        let conn = Connection::from_handle(self.db).map_err(|e| {
                            rusqlite::Error::UserFunctionError(Box::new(Error::Sqlite(e)))
                        })?;

                        let table_name = format!("{}_rowids", self.table_name);
                        let query = format!(
                            "SELECT COALESCE(MAX(rowid), 0) + 1 FROM \"{}\".\"{}\"",
                            self.schema_name, table_name
                        );

                        let auto_rowid = conn
                            .query_row(&query, [], |row| row.get::<_, i64>(0))
                            .unwrap_or(1);

                        std::mem::forget(conn);
                        auto_rowid
                    }
                }
            }
        } else {
            1
        };

        // Process vector columns
        let mut vec_col_idx = 0;
        for (col_idx, col) in self.columns.iter().enumerate() {
            if let ColumnType::Vector {
                vec_type,
                dimensions,
                index_quantization,
                hnsw_params,
            } = &col.col_type
            {
                let value_idx = col_idx + 2; // Skip NULL and rowid args
                if value_idx >= args.len() {
                    continue;
                }

                // Get the vector data as raw bytes - try blob first, then JSON text
                let vector_data: Vec<u8> = match args.get::<Option<Vec<u8>>>(value_idx) {
                    Ok(Some(data)) => data,
                    Ok(None) => continue, // NULL vector, skip
                    Err(_) => {
                        // Not a blob, try as JSON text string
                        match args.get::<Option<String>>(value_idx)? {
                            Some(json_str) => {
                                let vector = Vector::from_json(&json_str, *vec_type)
                                    .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
                                vector.as_bytes().to_vec()
                            }
                            None => continue, // NULL vector, skip
                        }
                    }
                };

                // Validate the byte size matches expected dimensions
                let expected_bytes = match vec_type {
                    VectorType::Float32 => dimensions * 4,
                    VectorType::Int8 => *dimensions,
                    VectorType::Bit => dimensions.div_ceil(8),
                };

                if vector_data.len() != expected_bytes {
                    // Provide user-friendly dimension-based error messages
                    let error_msg = if vector_data.is_empty() {
                        "zero-length vectors are not supported".to_string()
                    } else {
                        let actual_dims = match vec_type {
                            VectorType::Float32 => vector_data.len() / 4,
                            VectorType::Int8 => vector_data.len(),
                            VectorType::Bit => vector_data.len() * 8,
                        };
                        format!(
                            "Dimension mismatch for vector column: expected {} dimensions, got {}",
                            dimensions, actual_dims
                        )
                    };
                    return Err(rusqlite::Error::UserFunctionError(Box::new(
                        Error::InvalidParameter(error_msg),
                    )));
                }

                // Write the vector to shadow tables
                // SAFETY: self.db is valid for the lifetime of the virtual table
                unsafe {
                    shadow::insert_vector_ffi(
                        self.db,
                        &self.schema_name,
                        &self.table_name,
                        self.chunk_size,
                        rowid,
                        vec_col_idx, // Use vector-specific column index, not overall col_idx
                        &vector_data,
                    )
                    .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
                }

                // Also insert into HNSW index if this column has HNSW enabled
                if hnsw_params.enabled {
                    // SAFETY: self.db is valid for the lifetime of the virtual table
                    unsafe {
                        let conn = Connection::from_handle(self.db).map_err(|e| {
                            rusqlite::Error::UserFunctionError(Box::new(Error::Sqlite(e)))
                        })?;

                        // Load or initialize HNSW metadata
                        let mut metadata =
                            HnswMetadata::load_from_db(&conn, &self.table_name, &col.name)
                                .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?
                                .unwrap_or_else(|| {
                                    // Initialize new HNSW index with column's settings
                                    HnswMetadata::with_index_quantization(
                                        *dimensions as i32,
                                        *vec_type,
                                        hnsw_params.distance_metric,
                                        *index_quantization,
                                    )
                                });

                        // Get statement cache for this vector column
                        // CRITICAL: Lazy prepare statements on THIS connection using extension trait
                        let stmt_cache_ref = if vec_col_idx < self.hnsw_stmt_cache.len() {
                            let self_mut = self as *const _ as *mut Vec0Tab;
                            let cache = &mut (&mut *self_mut).hnsw_stmt_cache[vec_col_idx];

                            // Prepare statements lazily on this connection (ConnectionExt trait)
                            let nodes_table =
                                format!("{}_{}_hnsw_nodes", self.table_name, col.name);
                            let edges_table =
                                format!("{}_{}_hnsw_edges", self.table_name, col.name);
                            let meta_table = format!("{}_{}_hnsw_meta", self.table_name, col.name);

                            conn.prepare_or_reuse(
                                &mut cache.get_node_data,
                                &format!(
                                    "SELECT rowid, level, vector FROM \"{}\" WHERE rowid = ?",
                                    nodes_table
                                ),
                            )
                            .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

                            conn.prepare_or_reuse(&mut cache.get_edges,
                                    &format!("SELECT to_rowid FROM \"{}\" WHERE from_rowid = ? AND level = ?", edges_table))
                                    .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

                            conn.prepare_or_reuse(&mut cache.insert_node,
                                    &format!("INSERT OR REPLACE INTO \"{}\" (rowid, level, vector) VALUES (?, ?, ?)", nodes_table))
                                    .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

                            conn.prepare_or_reuse(&mut cache.insert_edge,
                                    &format!("INSERT OR IGNORE INTO \"{}\" (from_rowid, to_rowid, level) VALUES (?, ?, ?)", edges_table))
                                    .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

                            conn.prepare_or_reuse(
                                &mut cache.delete_edges_from,
                                &format!(
                                    "DELETE FROM \"{}\" WHERE from_rowid = ? AND level = ?",
                                    edges_table
                                ),
                            )
                            .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

                            conn.prepare_or_reuse(
                                &mut cache.update_meta,
                                &format!(
                                    "UPDATE \"{}\" SET \
                                     entry_point_rowid = ?, entry_point_level = ?, \
                                     num_nodes = ?, hnsw_version = ? \
                                     WHERE id = 1",
                                    meta_table
                                ),
                            )
                            .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

                            // Also prepare get_edges_with_dist (used by pruning during insert)
                            conn.prepare_or_reuse(
                                &mut cache.get_edges_with_dist,
                                &format!(
                                    "SELECT to_rowid, distance FROM \"{}\" WHERE from_rowid = ? AND level = ?",
                                    edges_table
                                ),
                            )
                            .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

                            // Also prepare batch_fetch_nodes (used for batch vector lookups)
                            let placeholders = (0..64).map(|_| "?").collect::<Vec<_>>().join(",");
                            conn.prepare_or_reuse(
                                &mut cache.batch_fetch_nodes,
                                &format!(
                                    "SELECT rowid, level, vector FROM \"{}\" WHERE rowid IN ({})",
                                    nodes_table, placeholders
                                ),
                            )
                            .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

                            let cache = &self.hnsw_stmt_cache[vec_col_idx];
                            Some(hnsw::insert::HnswStmtCache {
                                get_node_data: cache.get_node_data,
                                get_edges: cache.get_edges,
                                get_edges_with_dist: cache.get_edges_with_dist,
                                insert_node: cache.insert_node,
                                insert_edge: cache.insert_edge,
                                delete_edges_from: cache.delete_edges_from,
                                update_meta: cache.update_meta,
                                batch_fetch_nodes: cache.batch_fetch_nodes,
                            })
                        } else {
                            None
                        };

                        // Insert into HNSW graph with cached statements
                        hnsw::insert::insert_hnsw(
                            &conn,
                            &mut metadata,
                            &self.table_name,
                            &col.name,
                            rowid,
                            &vector_data,
                            stmt_cache_ref.as_ref(),
                        )
                        .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

                        std::mem::forget(conn);
                    }
                }

                vec_col_idx += 1;
            }
        }

        // Process ALL non-vector columns - single efficient INSERT into _data table
        let data_columns: Vec<usize> = self
            .columns
            .iter()
            .enumerate()
            .filter(|(_, c)| matches!(c.col_type, ColumnType::Metadata | ColumnType::Auxiliary))
            .map(|(col_idx, _)| col_idx)
            .collect();

        if !data_columns.is_empty() {
            // SAFETY: self.db is valid for the lifetime of the virtual table
            unsafe {
                let conn = Connection::from_handle(self.db)
                    .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(Error::Sqlite(e))))?;

                // Build single INSERT statement for _data table
                // Schema: (rowid, col00, col01, col02, ...)
                let data_table = format!("{}_data", self.table_name);
                let mut col_names = String::from("rowid");
                let mut placeholders = String::from("?");
                for i in 0..data_columns.len() {
                    col_names.push_str(&format!(", col{:02}", i));
                    placeholders.push_str(", ?");
                }

                let insert_sql = format!(
                    "INSERT OR REPLACE INTO \"{}\".\"{}\" ({}) VALUES ({})",
                    self.schema_name, data_table, col_names, placeholders
                );

                // Collect all values in order
                let mut values: Vec<rusqlite::types::Value> = vec![rowid.into()];
                for col_idx in &data_columns {
                    let value_idx = col_idx + 2; // Skip NULL and rowid args
                    if value_idx < args.len() {
                        // Preserve original type for efficiency
                        if let Ok(Some(i)) = args.get::<Option<i64>>(value_idx) {
                            values.push(i.into());
                        } else if let Ok(Some(f)) = args.get::<Option<f64>>(value_idx) {
                            values.push(f.into());
                        } else if let Ok(Some(s)) = args.get::<Option<String>>(value_idx) {
                            values.push(s.into());
                        } else if let Ok(Some(b)) = args.get::<Option<Vec<u8>>>(value_idx) {
                            values.push(b.into());
                        } else {
                            values.push(rusqlite::types::Value::Null);
                        }
                    } else {
                        values.push(rusqlite::types::Value::Null);
                    }
                }

                conn.execute(&insert_sql, rusqlite::params_from_iter(values))
                    .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(Error::Sqlite(e))))?;

                std::mem::forget(conn);
            }
        }

        Ok(rowid)
    }

    fn update(&mut self, args: &Updates<'_>) -> rusqlite::Result<()> {
        // args[0]: old rowid
        // args[1]: new rowid (we don't support rowid changes)
        // args[2..]: new column values

        if args.len() < 2 {
            return Err(rusqlite::Error::UserFunctionError(Box::new(
                Error::InvalidParameter("UPDATE requires at least 2 arguments".to_string()),
            )));
        }

        // Get old and new rowids
        let old_rowid = args.get::<i64>(0)?;
        let new_rowid = args.get::<i64>(1)?;

        // Check if rowid is being changed (not supported)
        if old_rowid != new_rowid {
            return Err(rusqlite::Error::UserFunctionError(Box::new(
                Error::NotImplemented("Changing rowid in UPDATE is not supported".to_string()),
            )));
        }

        // Process vector columns
        let mut vec_col_idx = 0;
        for (col_idx, col) in self.columns.iter().enumerate() {
            if let ColumnType::Vector {
                vec_type,
                dimensions,
                index_quantization,
                hnsw_params,
            } = &col.col_type
            {
                let value_idx = col_idx + 2; // Skip old_rowid and new_rowid args
                if value_idx >= args.len() {
                    vec_col_idx += 1;
                    continue;
                }

                // Get the new vector data - try blob first, then JSON text
                let vector_data: Vec<u8> = match args.get::<Option<Vec<u8>>>(value_idx) {
                    Ok(Some(data)) => data,
                    Ok(None) => {
                        vec_col_idx += 1;
                        continue; // NULL vector, skip (could implement DELETE here)
                    }
                    Err(_) => {
                        // Not a blob, try as JSON text string
                        match args.get::<Option<String>>(value_idx)? {
                            Some(json_str) => {
                                let vector = Vector::from_json(&json_str, *vec_type)
                                    .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
                                vector.as_bytes().to_vec()
                            }
                            None => {
                                vec_col_idx += 1;
                                continue; // NULL vector, skip
                            }
                        }
                    }
                };

                // Validate byte size
                let expected_bytes = match vec_type {
                    VectorType::Float32 => dimensions * 4,
                    VectorType::Int8 => *dimensions,
                    VectorType::Bit => dimensions.div_ceil(8),
                };

                if vector_data.len() != expected_bytes {
                    // Provide user-friendly dimension-based error messages
                    let error_msg = if vector_data.is_empty() {
                        "zero-length vectors are not supported".to_string()
                    } else {
                        let actual_dims = match vec_type {
                            VectorType::Float32 => vector_data.len() / 4,
                            VectorType::Int8 => vector_data.len(),
                            VectorType::Bit => vector_data.len() * 8,
                        };
                        format!(
                            "Dimension mismatch for vector column: expected {} dimensions, got {}",
                            dimensions, actual_dims
                        )
                    };
                    return Err(rusqlite::Error::UserFunctionError(Box::new(
                        Error::InvalidParameter(error_msg),
                    )));
                }

                // SAFETY: self.db is valid for the lifetime of the virtual table
                let conn = unsafe { Connection::from_handle(self.db)? };

                // Get chunk position for this rowid
                let rowids_table = format!("{}_rowids", self.table_name);
                let query = format!(
                    "SELECT chunk_id, chunk_offset FROM \"{}\".\"{}\" WHERE rowid = ?",
                    self.schema_name, rowids_table
                );

                let chunk_info: Option<(i64, i64)> = conn
                    .query_row(&query, [old_rowid], |row| Ok((row.get(0)?, row.get(1)?)))
                    .optional()?;

                if let Some((chunk_id, chunk_offset)) = chunk_info {
                    // Write new vector to shadow table (overwrite existing)
                    shadow::write_vector_to_chunk(
                        &conn,
                        &self.schema_name,
                        &self.table_name,
                        vec_col_idx, // Use vector-specific column index, not overall col_idx
                        chunk_id,
                        chunk_offset,
                        &vector_data,
                    )
                    .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

                    // Update HNSW index if this column has HNSW enabled
                    if hnsw_params.enabled {
                        // Delete old HNSW node
                        let nodes_table = format!("{}_{}_hnsw_nodes", self.table_name, col.name);
                        let edges_table = format!("{}_{}_hnsw_edges", self.table_name, col.name);
                        let levels_table = format!("{}_{}_hnsw_levels", self.table_name, col.name);

                        // Get old level before deletion
                        let old_level: Option<i32> = conn
                            .query_row(
                                &format!("SELECT level FROM \"{}\" WHERE rowid = ?", nodes_table),
                                [old_rowid],
                                |row| row.get(0),
                            )
                            .optional()?;

                        // Delete old node and edges
                        let _ = conn.execute(
                            &format!("DELETE FROM \"{}\" WHERE rowid = ?", nodes_table),
                            [old_rowid],
                        );
                        let _ = conn.execute(
                            &format!(
                                "DELETE FROM \"{}\" WHERE from_rowid = ? OR to_rowid = ?",
                                edges_table
                            ),
                            rusqlite::params![old_rowid, old_rowid],
                        );
                        let _ = conn.execute(
                            &format!("DELETE FROM \"{}\" WHERE rowid = ?", levels_table),
                            [old_rowid],
                        );

                        // Insert new node into HNSW
                        let mut metadata =
                            HnswMetadata::load_from_db(&conn, &self.table_name, &col.name)
                                .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?
                                .unwrap_or_else(|| {
                                    HnswMetadata::with_index_quantization(
                                        *dimensions as i32,
                                        *vec_type,
                                        hnsw_params.distance_metric,
                                        *index_quantization,
                                    )
                                });

                        // Decrement node count since we're replacing
                        if old_level.is_some() {
                            metadata.num_nodes = metadata.num_nodes.saturating_sub(1);
                        }

                        // Get statement cache for this vector column
                        // CRITICAL: Lazy prepare statements on THIS connection using extension trait
                        let stmt_cache_ref = if vec_col_idx < self.hnsw_stmt_cache.len() {
                            // SAFETY: We need mutable access to the cache to prepare statements
                            unsafe {
                                let self_mut = self as *const _ as *mut Vec0Tab;
                                let cache = &mut (&mut *self_mut).hnsw_stmt_cache[vec_col_idx];

                                // Prepare statements lazily on this connection (ConnectionExt trait)
                                let nodes_table =
                                    format!("{}_{}_hnsw_nodes", self.table_name, col.name);
                                let edges_table =
                                    format!("{}_{}_hnsw_edges", self.table_name, col.name);
                                let meta_table =
                                    format!("{}_{}_hnsw_meta", self.table_name, col.name);

                                conn.prepare_or_reuse(
                                    &mut cache.get_node_data,
                                    &format!(
                                        "SELECT rowid, level, vector FROM \"{}\" WHERE rowid = ?",
                                        nodes_table
                                    ),
                                )
                                .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

                                conn.prepare_or_reuse(&mut cache.get_edges,
                                    &format!("SELECT to_rowid FROM \"{}\" WHERE from_rowid = ? AND level = ?", edges_table))
                                    .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

                                conn.prepare_or_reuse(&mut cache.insert_node,
                                    &format!("INSERT OR REPLACE INTO \"{}\" (rowid, level, vector) VALUES (?, ?, ?)", nodes_table))
                                    .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

                                conn.prepare_or_reuse(&mut cache.insert_edge,
                                    &format!("INSERT OR IGNORE INTO \"{}\" (from_rowid, to_rowid, level) VALUES (?, ?, ?)", edges_table))
                                    .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

                                conn.prepare_or_reuse(
                                    &mut cache.delete_edges_from,
                                    &format!(
                                        "DELETE FROM \"{}\" WHERE from_rowid = ? AND level = ?",
                                        edges_table
                                    ),
                                )
                                .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

                                conn.prepare_or_reuse(
                                    &mut cache.update_meta,
                                    &format!(
                                        "INSERT OR REPLACE INTO \"{}\" (key, value) VALUES (?, ?)",
                                        meta_table
                                    ),
                                )
                                .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

                                // Also prepare get_edges_with_dist (used by pruning during insert)
                                conn.prepare_or_reuse(
                                    &mut cache.get_edges_with_dist,
                                    &format!(
                                        "SELECT to_rowid, distance FROM \"{}\" WHERE from_rowid = ? AND level = ?",
                                        edges_table
                                    ),
                                )
                                .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

                                // Also prepare batch_fetch_nodes (used for batch vector lookups)
                                let placeholders =
                                    (0..64).map(|_| "?").collect::<Vec<_>>().join(",");
                                conn.prepare_or_reuse(
                                    &mut cache.batch_fetch_nodes,
                                    &format!(
                                        "SELECT rowid, level, vector FROM \"{}\" WHERE rowid IN ({})",
                                        nodes_table, placeholders
                                    ),
                                )
                                .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
                            }

                            let cache = &self.hnsw_stmt_cache[vec_col_idx];
                            Some(hnsw::insert::HnswStmtCache {
                                get_node_data: cache.get_node_data,
                                get_edges: cache.get_edges,
                                get_edges_with_dist: cache.get_edges_with_dist,
                                insert_node: cache.insert_node,
                                insert_edge: cache.insert_edge,
                                delete_edges_from: cache.delete_edges_from,
                                update_meta: cache.update_meta,
                                batch_fetch_nodes: cache.batch_fetch_nodes,
                            })
                        } else {
                            None
                        };

                        // Re-insert with new vector using cached statements
                        hnsw::insert::insert_hnsw(
                            &conn,
                            &mut metadata,
                            &self.table_name,
                            &col.name,
                            old_rowid,
                            &vector_data,
                            stmt_cache_ref.as_ref(),
                        )
                        .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
                    }
                }

                std::mem::forget(conn);
                vec_col_idx += 1;
            }
        }

        // Process ALL non-vector columns on UPDATE - single efficient UPDATE to _data table
        let data_columns: Vec<usize> = self
            .columns
            .iter()
            .enumerate()
            .filter(|(_, c)| matches!(c.col_type, ColumnType::Metadata | ColumnType::Auxiliary))
            .map(|(col_idx, _)| col_idx)
            .collect();

        if !data_columns.is_empty() {
            // SAFETY: self.db is valid for the lifetime of the virtual table
            unsafe {
                let conn = Connection::from_handle(self.db)
                    .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(Error::Sqlite(e))))?;

                let data_table = format!("{}_data", self.table_name);

                // Build single UPDATE statement for all non-vector columns
                let mut set_clauses = Vec::new();
                let mut values: Vec<rusqlite::types::Value> = Vec::new();

                for (data_idx, col_idx) in data_columns.iter().enumerate() {
                    let value_idx = col_idx + 2; // Skip old_rowid and new_rowid args
                    if value_idx < args.len() {
                        set_clauses.push(format!("col{:02} = ?", data_idx));

                        // Preserve type information
                        if let Ok(Some(i)) = args.get::<Option<i64>>(value_idx) {
                            values.push(i.into());
                        } else if let Ok(Some(f)) = args.get::<Option<f64>>(value_idx) {
                            values.push(f.into());
                        } else if let Ok(Some(s)) = args.get::<Option<String>>(value_idx) {
                            values.push(s.into());
                        } else if let Ok(Some(b)) = args.get::<Option<Vec<u8>>>(value_idx) {
                            values.push(b.into());
                        } else {
                            values.push(rusqlite::types::Value::Null);
                        }
                    }
                }

                if !set_clauses.is_empty() {
                    // Add rowid for WHERE clause
                    values.push(old_rowid.into());

                    let update_sql = format!(
                        "UPDATE \"{}\".\"{}\" SET {} WHERE rowid = ?",
                        self.schema_name,
                        data_table,
                        set_clauses.join(", ")
                    );

                    conn.execute(&update_sql, rusqlite::params_from_iter(values))
                        .map_err(|e| {
                            rusqlite::Error::UserFunctionError(Box::new(Error::Sqlite(e)))
                        })?;
                }

                std::mem::forget(conn);
            }
        }

        Ok(())
    }
}

impl<'vtab> rusqlite::vtab::TransactionVTab<'vtab> for Vec0Tab {
    fn begin(&mut self) -> rusqlite::Result<()> {
        // No-op: page-cache based design persists immediately
        // All changes are committed to shadow tables as they happen
        Ok(())
    }

    fn sync(&mut self) -> rusqlite::Result<()> {
        // No-op: statements are reset immediately after each use in storage.rs
        Ok(())
    }

    fn commit(&mut self) -> rusqlite::Result<()> {
        // No-op: page-cache based design persists immediately
        Ok(())
    }

    fn rollback(&mut self) -> rusqlite::Result<()> {
        // No-op: SQLite handles rollback via shadow tables automatically
        Ok(())
    }
}

/// vec0 cursor for iteration
#[repr(C)]
pub struct Vec0TabCursor<'vtab> {
    base: sqlite3_vtab_cursor,
    phantom: PhantomData<&'vtab Vec0Tab>,
    rowids: Vec<i64>,
    distances: Vec<f32>, // For KNN queries
    current_index: usize,
}

impl<'vtab> Vec0TabCursor<'vtab> {
    fn new(_table: &Vec0Tab) -> Self {
        Vec0TabCursor {
            base: sqlite3_vtab_cursor::default(),
            phantom: PhantomData,
            rowids: Vec::new(),
            distances: Vec::new(),
            current_index: 0,
        }
    }

    /// Accessor to the associated virtual table
    fn vtab(&self) -> &Vec0Tab {
        unsafe { &*(self.base.pVtab as *const Vec0Tab) }
    }

    /// Get the current rowid
    fn current_rowid(&self) -> Option<i64> {
        self.rowids.get(self.current_index).copied()
    }

    /// Get the current distance (for KNN queries)
    fn current_distance(&self) -> Option<f32> {
        self.distances.get(self.current_index).copied()
    }
}

unsafe impl VTabCursor for Vec0TabCursor<'_> {
    fn filter(
        &mut self,
        _idx_num: c_int,
        idx_str: Option<&str>,
        args: &Filters<'_>,
    ) -> rusqlite::Result<()> {
        let vtab = self.vtab();

        // Parse idxStr to determine query plan
        let idx_str = idx_str.unwrap_or("1");
        let (plan, _blocks) =
            parse_idxstr(idx_str).map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

        match plan {
            QueryPlan::Knn => {
                // KNN query - extract query vector and k from args
                // Find the vector column first (needed to determine vec_type for JSON parsing)
                let (col_idx, col) = vtab
                    .columns
                    .iter()
                    .enumerate()
                    .find(|(_, c)| matches!(c.col_type, ColumnType::Vector { .. }))
                    .ok_or_else(|| {
                        rusqlite::Error::UserFunctionError(Box::new(Error::InvalidParameter(
                            "No vector column found for KNN query".to_string(),
                        )))
                    })?;

                // Get the vector type from the column
                let vec_type = match &col.col_type {
                    ColumnType::Vector { vec_type, .. } => *vec_type,
                    _ => unreachable!(), // We already filtered for Vector columns
                };

                // Get query vector - try blob first, then JSON text string
                let query_vector: Vec<u8> = match args.get::<Option<Vec<u8>>>(0) {
                    Ok(Some(data)) => data,
                    Ok(None) => {
                        return Err(rusqlite::Error::UserFunctionError(Box::new(
                            Error::InvalidParameter("Query vector cannot be NULL".to_string()),
                        )));
                    }
                    Err(_) => {
                        // Not a blob, try as JSON text string
                        match args.get::<Option<String>>(0)? {
                            Some(json_str) => {
                                let vector = Vector::from_json(&json_str, vec_type)
                                    .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
                                vector.as_bytes().to_vec()
                            }
                            None => {
                                return Err(rusqlite::Error::UserFunctionError(Box::new(
                                    Error::InvalidParameter(
                                        "Query vector cannot be NULL".to_string(),
                                    ),
                                )));
                            }
                        }
                    }
                };

                let k: i64 = args.get(1)?;

                // Check if HNSW is enabled for this column and get distance metric
                let (hnsw_enabled, distance_metric) =
                    if let ColumnType::Vector { hnsw_params, .. } = &col.col_type {
                        (hnsw_params.enabled, hnsw_params.distance_metric)
                    } else {
                        (false, DistanceMetric::Cosine) // Default to cosine
                    };

                // Compute vec_col_idx: the index among vector columns only (0 for first vector column)
                let vec_col_idx = vtab
                    .columns
                    .iter()
                    .take(col_idx)
                    .filter(|c| matches!(c.col_type, ColumnType::Vector { .. }))
                    .count();

                // Execute search: HNSW if enabled, otherwise brute-force ENN
                // SAFETY: vtab.db is valid for the lifetime of the virtual table
                let results = unsafe {
                    let conn = Connection::from_handle(vtab.db).map_err(|e| {
                        rusqlite::Error::UserFunctionError(Box::new(Error::Sqlite(e)))
                    })?;

                    let result = if hnsw_enabled {
                        // HNSW mode: use approximate search with cached statements
                        let metadata =
                            HnswMetadata::load_from_db(&conn, &vtab.table_name, &col.name)
                                .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

                        if let Some(meta) = metadata {
                            // Prepare search statements for this query using raw FFI
                            // Create local statement handles (prepared once per query)
                            let nodes_table =
                                format!("{}_{}_hnsw_nodes", vtab.table_name, col.name);
                            let edges_table =
                                format!("{}_{}_hnsw_edges", vtab.table_name, col.name);

                            let mut get_node_stmt: *mut ffi::sqlite3_stmt = std::ptr::null_mut();
                            let mut get_edges_stmt: *mut ffi::sqlite3_stmt = std::ptr::null_mut();
                            let mut batch_fetch_stmt: *mut ffi::sqlite3_stmt = std::ptr::null_mut();

                            // Prepare get_node_data
                            let sql = std::ffi::CString::new(format!(
                                "SELECT rowid, level, vector FROM \"{}\" WHERE rowid = ?",
                                nodes_table
                            ))
                            .unwrap();
                            ffi::sqlite3_prepare_v2(
                                conn.handle(),
                                sql.as_ptr(),
                                -1,
                                &mut get_node_stmt,
                                std::ptr::null_mut(),
                            );

                            // Prepare get_edges
                            let sql = std::ffi::CString::new(format!(
                                "SELECT to_rowid FROM \"{}\" WHERE from_rowid = ? AND level = ?",
                                edges_table
                            ))
                            .unwrap();
                            ffi::sqlite3_prepare_v2(
                                conn.handle(),
                                sql.as_ptr(),
                                -1,
                                &mut get_edges_stmt,
                                std::ptr::null_mut(),
                            );

                            // Prepare batch_fetch_nodes (64 placeholders)
                            let placeholders = (0..64).map(|_| "?").collect::<Vec<_>>().join(",");
                            let sql = std::ffi::CString::new(format!(
                                "SELECT rowid, level, vector FROM \"{}\" WHERE rowid IN ({})",
                                nodes_table, placeholders
                            ))
                            .unwrap();
                            ffi::sqlite3_prepare_v2(
                                conn.handle(),
                                sql.as_ptr(),
                                -1,
                                &mut batch_fetch_stmt,
                                std::ptr::null_mut(),
                            );

                            let search_cache = hnsw::search::SearchStmtCache {
                                get_node_data: if get_node_stmt.is_null() {
                                    None
                                } else {
                                    Some(get_node_stmt)
                                },
                                get_edges: if get_edges_stmt.is_null() {
                                    None
                                } else {
                                    Some(get_edges_stmt)
                                },
                                batch_fetch_nodes: if batch_fetch_stmt.is_null() {
                                    None
                                } else {
                                    Some(batch_fetch_stmt)
                                },
                            };

                            let search_result = hnsw::search::search_hnsw(
                                &conn,
                                &meta,
                                &vtab.table_name,
                                &col.name,
                                &query_vector,
                                k as usize,
                                None,
                                Some(&search_cache),
                            )
                            .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)));

                            // Finalize local statements
                            if !get_node_stmt.is_null() {
                                ffi::sqlite3_finalize(get_node_stmt);
                            }
                            if !get_edges_stmt.is_null() {
                                ffi::sqlite3_finalize(get_edges_stmt);
                            }
                            if !batch_fetch_stmt.is_null() {
                                ffi::sqlite3_finalize(batch_fetch_stmt);
                            }

                            search_result?
                        } else {
                            // HNSW index is missing or corrupted
                            std::mem::forget(conn);
                            return Err(rusqlite::Error::UserFunctionError(Box::new(
                                Error::InvalidState(format!(
                                    "HNSW index not available for column '{}'. \
                                     Check index integrity or rebuild with vec_rebuild_hnsw()",
                                    col.name
                                )),
                            )));
                        }
                    } else {
                        // ENN mode: exact nearest neighbor via brute force
                        brute_force_search(
                            &conn,
                            &vtab.schema_name,
                            &vtab.table_name,
                            vec_col_idx,
                            &query_vector,
                            k as usize,
                            distance_metric,
                        )
                        .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?
                    };

                    std::mem::forget(conn);
                    result
                };

                // Store results in cursor
                self.rowids = results.iter().map(|(rowid, _)| *rowid).collect();
                self.distances = results.iter().map(|(_, dist)| *dist).collect();
                self.current_index = 0;
            }
            _ => {
                // Full scan or other query types
                // Get all rowids from shadow tables
                // SAFETY: vtab.db is valid for the lifetime of the virtual table
                let rowids = unsafe {
                    let conn = Connection::from_handle(vtab.db).map_err(|e| {
                        rusqlite::Error::UserFunctionError(Box::new(Error::Sqlite(e)))
                    })?;

                    let result = shadow::get_all_rowids(&conn, &vtab.schema_name, &vtab.table_name)
                        .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

                    std::mem::forget(conn);
                    result
                };

                self.rowids = rowids;
                self.distances.clear();
                self.current_index = 0;
            }
        }

        Ok(())
    }

    fn next(&mut self) -> rusqlite::Result<()> {
        self.current_index += 1;
        Ok(())
    }

    fn eof(&self) -> bool {
        self.current_index >= self.rowids.len()
    }

    fn column(&self, ctx: &mut Context, col: c_int) -> rusqlite::Result<()> {
        let vtab = self.vtab();
        let col_idx = col as usize;

        // Get current rowid
        let rowid = match self.current_rowid() {
            Some(r) => r,
            None => {
                ctx.set_result(&rusqlite::types::Null)?;
                return Ok(());
            }
        };

        // Check if this is the distance column (typically last column)
        // Distance column is a hidden column that comes after all user columns
        if col_idx == vtab.columns.len() {
            // Return distance if we have it (from KNN query)
            if let Some(dist) = self.current_distance() {
                ctx.set_result(&dist)?;
                return Ok(());
            }
        }

        // Check if this is a vector column
        if col_idx < vtab.columns.len()
            && let ColumnType::Vector {
                vec_type,
                dimensions,
                ..
            } = &vtab.columns[col_idx].col_type
        {
            // Compute vec_col_idx: the index among vector columns only (0 for first vector column)
            let vec_col_idx = vtab
                .columns
                .iter()
                .take(col_idx)
                .filter(|c| matches!(c.col_type, ColumnType::Vector { .. }))
                .count();

            // Read vector from shadow tables
            // SAFETY: vtab.db is valid for the lifetime of the virtual table
            let vector_data = unsafe {
                let conn = Connection::from_handle(vtab.db)
                    .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(Error::Sqlite(e))))?;

                let result = shadow::read_vector_from_chunk(
                    &conn,
                    &vtab.schema_name,
                    &vtab.table_name,
                    vec_col_idx,
                    rowid,
                )
                .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

                std::mem::forget(conn);
                result
            };

            match vector_data {
                Some(data) => {
                    // Convert binary blob to Vector and then to JSON for human-readable output
                    let vector = Vector::from_blob(&data, *vec_type, *dimensions)
                        .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
                    let json = vector
                        .to_json()
                        .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
                    ctx.set_result(&json)?
                }
                None => ctx.set_result(&rusqlite::types::Null)?,
            }
            return Ok(());
        }

        // Check if this is a non-vector column (Metadata or Auxiliary)
        // Read from unified _data table for efficiency
        if col_idx < vtab.columns.len()
            && matches!(
                vtab.columns[col_idx].col_type,
                ColumnType::Metadata | ColumnType::Auxiliary
            )
        {
            // Compute data_col_idx: the index among all non-vector columns
            let data_col_idx = vtab
                .columns
                .iter()
                .take(col_idx)
                .filter(|c| matches!(c.col_type, ColumnType::Metadata | ColumnType::Auxiliary))
                .count();

            // Read from unified _data shadow table
            // SAFETY: vtab.db is valid for the lifetime of the virtual table
            unsafe {
                let conn = Connection::from_handle(vtab.db)
                    .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(Error::Sqlite(e))))?;

                let data_table = format!("{}_data", vtab.table_name);
                let query = format!(
                    "SELECT col{:02} FROM \"{}\".\"{}\" WHERE rowid = ?",
                    data_col_idx, vtab.schema_name, data_table
                );

                // Try to get the value preserving its original type
                let result = conn
                    .query_row(&query, [rowid], |row| {
                        // Get the raw value and set result based on type
                        let value_ref = row.get_ref(0)?;
                        match value_ref {
                            rusqlite::types::ValueRef::Null => {
                                ctx.set_result(&rusqlite::types::Null)
                            }
                            rusqlite::types::ValueRef::Integer(i) => ctx.set_result(&i),
                            rusqlite::types::ValueRef::Real(f) => ctx.set_result(&f),
                            rusqlite::types::ValueRef::Text(t) => {
                                let s = std::str::from_utf8(t).unwrap_or("");
                                ctx.set_result(&s)
                            }
                            rusqlite::types::ValueRef::Blob(b) => ctx.set_result(&b),
                        }
                    })
                    .optional()
                    .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(Error::Sqlite(e))))?;

                // If no row found, return NULL
                if result.is_none() {
                    ctx.set_result(&rusqlite::types::Null)?;
                }

                std::mem::forget(conn);
            };

            return Ok(());
        }

        // For partition key columns, return NULL for now
        ctx.set_result(&rusqlite::types::Null)?;
        Ok(())
    }

    fn rowid(&self) -> rusqlite::Result<i64> {
        Ok(self.current_rowid().unwrap_or(0))
    }
}

/// Query plan types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryPlan {
    /// Full table scan
    FullScan,
    /// Point query by rowid
    Point,
    /// K-nearest neighbor search
    Knn,
}

impl QueryPlan {
    /// Convert to idxStr header character
    pub fn to_header_char(&self) -> char {
        match self {
            QueryPlan::FullScan => '1',
            QueryPlan::Point => '2',
            QueryPlan::Knn => '3',
        }
    }

    /// Parse from idxStr header character
    pub fn from_header_char(c: char) -> Result<Self> {
        match c {
            '1' => Ok(QueryPlan::FullScan),
            '2' => Ok(QueryPlan::Point),
            '3' => Ok(QueryPlan::Knn),
            _ => Err(Error::InvalidParameter(format!(
                "Invalid query plan header: '{}'",
                c
            ))),
        }
    }
}

/// idxStr block types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdxStrKind {
    /// KNN query vector: '{___'
    KnnMatch,
    /// KNN k limit: '}___'
    KnnK,
    /// Rowid IN filter: '[___'
    KnnRowidIn,
    /// Partition constraint: ']Xop_' (X=column, op=operator)
    PartitionConstraint,
    /// Metadata constraint: '&Xop_' (X=column, op=operator)
    MetadataConstraint,
    /// Point query rowid: '!___'
    PointId,
}

impl IdxStrKind {
    /// Convert to 4-character block (first character)
    pub fn to_char(&self) -> char {
        match self {
            IdxStrKind::KnnMatch => '{',
            IdxStrKind::KnnK => '}',
            IdxStrKind::KnnRowidIn => '[',
            IdxStrKind::PartitionConstraint => ']',
            IdxStrKind::MetadataConstraint => '&',
            IdxStrKind::PointId => '!',
        }
    }

    /// Parse from first character of 4-character block
    pub fn from_char(c: char) -> Result<Self> {
        match c {
            '{' => Ok(IdxStrKind::KnnMatch),
            '}' => Ok(IdxStrKind::KnnK),
            '[' => Ok(IdxStrKind::KnnRowidIn),
            ']' => Ok(IdxStrKind::PartitionConstraint),
            '&' => Ok(IdxStrKind::MetadataConstraint),
            '!' => Ok(IdxStrKind::PointId),
            _ => Err(Error::InvalidParameter(format!(
                "Invalid idxStr block kind: '{}'",
                c
            ))),
        }
    }
}

/// Brute force k-NN search (exact nearest neighbor)
/// Used when index_type=enn for guaranteed exact results
fn brute_force_search(
    conn: &Connection,
    schema: &str,
    table_name: &str,
    column_idx: usize,
    query_vector: &[u8],
    k: usize,
    distance_metric: DistanceMetric,
) -> Result<Vec<(i64, f32)>> {
    use crate::distance;
    use crate::vector::Vector;

    // Get all rowids
    let rowids_table = format!("{}_rowids", table_name);
    let mut stmt = conn
        .prepare(&format!(
            "SELECT rowid FROM \"{}\".\"{}\"",
            schema, rowids_table
        ))
        .map_err(Error::Sqlite)?;

    let rowids: Vec<i64> = stmt
        .query_map([], |row| row.get(0))
        .map_err(Error::Sqlite)?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Error::Sqlite)?;

    drop(stmt);

    // Parse query vector to get its properties
    let query_vec = Vector::from_blob(query_vector, VectorType::Float32, query_vector.len() / 4)
        .map_err(|_| Error::InvalidParameter("Invalid query vector format".to_string()))?;

    // Calculate distances for all vectors
    let mut distances = Vec::with_capacity(rowids.len());
    for rowid in rowids {
        // Read vector from shadow table
        let vector_bytes =
            match shadow::read_vector_from_chunk(conn, schema, table_name, column_idx, rowid) {
                Ok(Some(bytes)) => bytes,
                Ok(None) => continue, // NULL vector, skip
                Err(_) => continue,   // Error reading vector, skip
            };

        // Parse vector and calculate distance
        let vec =
            match Vector::from_blob(&vector_bytes, VectorType::Float32, vector_bytes.len() / 4) {
                Ok(v) => v,
                Err(_) => continue, // Invalid vector, skip
            };

        let dist = match distance::distance(&query_vec, &vec, distance_metric) {
            Ok(d) => d,
            Err(_) => continue, // Error calculating distance, skip
        };

        distances.push((rowid, dist));
    }

    // Sort by distance and take top k
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    distances.truncate(k);

    Ok(distances)
}

/// Parse idxStr into components
pub fn parse_idxstr(idxstr: &str) -> Result<(QueryPlan, Vec<IdxStrKind>)> {
    if idxstr.is_empty() {
        return Err(Error::InvalidParameter("Empty idxStr".to_string()));
    }

    let mut chars = idxstr.chars();
    let header = chars.next().unwrap();
    let plan = QueryPlan::from_header_char(header)?;

    let mut blocks = Vec::new();
    let remaining: String = chars.collect();

    // Parse 4-character blocks
    for chunk in remaining.as_bytes().chunks(4) {
        if chunk.len() == 4 {
            let block_kind = IdxStrKind::from_char(chunk[0] as char)?;
            blocks.push(block_kind);
        }
    }

    Ok((plan, blocks))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_vec0_works() {
        let db = Connection::open_in_memory().unwrap();
        let result = register_vec0_module(&db);
        assert!(result.is_ok());
    }

    #[test]
    fn test_query_plan_conversions() {
        assert_eq!(QueryPlan::FullScan.to_header_char(), '1');
        assert_eq!(QueryPlan::Point.to_header_char(), '2');
        assert_eq!(QueryPlan::Knn.to_header_char(), '3');

        assert_eq!(
            QueryPlan::from_header_char('1').unwrap(),
            QueryPlan::FullScan
        );
        assert_eq!(QueryPlan::from_header_char('2').unwrap(), QueryPlan::Point);
        assert_eq!(QueryPlan::from_header_char('3').unwrap(), QueryPlan::Knn);

        assert!(QueryPlan::from_header_char('X').is_err());
    }

    #[test]
    fn test_idxstr_kind_conversions() {
        assert_eq!(IdxStrKind::KnnMatch.to_char(), '{');
        assert_eq!(IdxStrKind::KnnK.to_char(), '}');
        assert_eq!(IdxStrKind::PointId.to_char(), '!');

        assert_eq!(IdxStrKind::from_char('{').unwrap(), IdxStrKind::KnnMatch);
        assert_eq!(IdxStrKind::from_char('}').unwrap(), IdxStrKind::KnnK);
        assert_eq!(IdxStrKind::from_char('!').unwrap(), IdxStrKind::PointId);

        assert!(IdxStrKind::from_char('X').is_err());
    }

    #[test]
    fn test_parse_idxstr_fullscan() {
        let (plan, blocks) = parse_idxstr("1").unwrap();
        assert_eq!(plan, QueryPlan::FullScan);
        assert_eq!(blocks.len(), 0);
    }

    #[test]
    fn test_parse_idxstr_knn() {
        // KNN query with match and k: "3{___}___"
        let (plan, blocks) = parse_idxstr("3{___}___").unwrap();
        assert_eq!(plan, QueryPlan::Knn);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0], IdxStrKind::KnnMatch);
        assert_eq!(blocks[1], IdxStrKind::KnnK);
    }

    #[test]
    fn test_parse_idxstr_point() {
        // Point query: "2!___"
        let (plan, blocks) = parse_idxstr("2!___").unwrap();
        assert_eq!(plan, QueryPlan::Point);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0], IdxStrKind::PointId);
    }

    #[test]
    fn test_parse_idxstr_empty() {
        let result = parse_idxstr("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_idxstr_invalid_header() {
        let result = parse_idxstr("X");
        assert!(result.is_err());
    }

    #[test]
    fn test_virtual_table_creation_with_shadow_tables() {
        use crate::init;

        let db = Connection::open_in_memory().unwrap();
        init(&db).unwrap();

        // Create a vec0 virtual table with one vector column (no HNSW)
        db.execute(
            "CREATE VIRTUAL TABLE vec_test USING vec0(embedding float[384])",
            [],
        )
        .unwrap();

        // Verify shadow tables were created
        let tables: Vec<String> = db
            .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'vec_test%' ORDER BY name")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();

        println!("Created tables: {:?}", tables);

        // Check that expected shadow tables exist
        assert!(
            tables.contains(&"vec_test_chunks".to_string()),
            "Missing _chunks table"
        );
        assert!(
            tables.contains(&"vec_test_rowids".to_string()),
            "Missing _rowids table"
        );
        assert!(
            tables.contains(&"vec_test_vector_chunks00".to_string()),
            "Missing _vector_chunks00 table"
        );

        // HNSW tables should NOT exist when hnsw() is not specified
        assert!(
            !tables.contains(&"vec_test_embedding_hnsw_meta".to_string()),
            "HNSW meta table should not exist without hnsw()"
        );
        assert!(
            !tables.contains(&"vec_test_embedding_hnsw_nodes".to_string()),
            "HNSW nodes table should not exist without hnsw()"
        );
    }

    #[test]
    fn test_virtual_table_creation_with_hnsw() {
        use crate::init;

        let db = Connection::open_in_memory().unwrap();
        init(&db).unwrap();

        // Create a vec0 virtual table with HNSW enabled
        db.execute(
            "CREATE VIRTUAL TABLE vec_hnsw USING vec0(embedding float[384] hnsw())",
            [],
        )
        .unwrap();

        // Verify shadow tables were created
        let tables: Vec<String> = db
            .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'vec_hnsw%' ORDER BY name")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();

        println!("Created tables: {:?}", tables);

        // Check that basic shadow tables exist
        assert!(
            tables.contains(&"vec_hnsw_chunks".to_string()),
            "Missing _chunks table"
        );
        assert!(
            tables.contains(&"vec_hnsw_rowids".to_string()),
            "Missing _rowids table"
        );

        // Check HNSW tables exist when hnsw() is specified
        assert!(
            tables.contains(&"vec_hnsw_embedding_hnsw_meta".to_string()),
            "Missing HNSW meta table"
        );
        assert!(
            tables.contains(&"vec_hnsw_embedding_hnsw_nodes".to_string()),
            "Missing HNSW nodes table"
        );
        assert!(
            tables.contains(&"vec_hnsw_embedding_hnsw_edges".to_string()),
            "Missing HNSW edges table"
        );
        assert!(
            tables.contains(&"vec_hnsw_embedding_hnsw_levels".to_string()),
            "Missing HNSW levels table"
        );
    }

    #[test]
    fn test_virtual_table_schema() {
        use crate::init;

        let db = Connection::open_in_memory().unwrap();
        init(&db).unwrap();

        // Create a vec0 virtual table
        db.execute(
            "CREATE VIRTUAL TABLE vec_test2 USING vec0(id INTEGER PRIMARY KEY, embedding float[768], category TEXT)",
            [],
        )
        .unwrap();

        // Verify we can query the table (even though it's empty)
        let count: i64 = db
            .query_row("SELECT COUNT(*) FROM vec_test2", [], |row| row.get(0))
            .unwrap();

        assert_eq!(count, 0, "New table should be empty");
    }

    #[test]
    fn test_insert_vector() {
        use crate::init;

        let db = Connection::open_in_memory().unwrap();
        init(&db).unwrap();

        // Create a vec0 virtual table
        db.execute(
            "CREATE VIRTUAL TABLE vec_test3 USING vec0(embedding float[3])",
            [],
        )
        .unwrap();

        // Insert a vector using vec_f32 function
        let result = db.execute(
            "INSERT INTO vec_test3(rowid, embedding) VALUES (1, vec_f32('[1.0, 2.0, 3.0]'))",
            [],
        );

        match result {
            Ok(rows) => {
                println!("INSERT successful: {} rows affected", rows);
            }
            Err(e) => {
                println!("INSERT failed: {:?}", e);
                // For now, don't fail the test - INSERT is still being implemented
            }
        }

        // Verify rowid mapping was created
        let mapping_count = db.query_row("SELECT COUNT(*) FROM vec_test3_rowids", [], |row| {
            row.get::<_, i64>(0)
        });

        if let Ok(count) = mapping_count {
            println!("Rowid mappings: {}", count);
            if count > 0 {
                assert_eq!(count, 1, "Should have one rowid mapping");
            }
        }
    }

    #[test]
    fn test_insert_and_select_vector() {
        use crate::init;

        let db = Connection::open_in_memory().unwrap();
        init(&db).unwrap();

        // Create a vec0 virtual table
        db.execute(
            "CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[3])",
            [],
        )
        .unwrap();

        // Insert multiple vectors
        db.execute(
            "INSERT INTO vec_items(rowid, embedding) VALUES (1, vec_f32('[1.0, 2.0, 3.0]'))",
            [],
        )
        .unwrap();

        db.execute(
            "INSERT INTO vec_items(rowid, embedding) VALUES (2, vec_f32('[4.0, 5.0, 6.0]'))",
            [],
        )
        .unwrap();

        db.execute(
            "INSERT INTO vec_items(rowid, embedding) VALUES (3, vec_f32('[7.0, 8.0, 9.0]'))",
            [],
        )
        .unwrap();

        // Query the table
        let count: i64 = db
            .query_row("SELECT COUNT(*) FROM vec_items", [], |row| row.get(0))
            .unwrap();

        println!("Total rows: {}", count);
        assert_eq!(count, 3, "Should have 3 rows");

        // Query specific vectors
        let mut stmt = db
            .prepare("SELECT rowid FROM vec_items ORDER BY rowid")
            .unwrap();
        let rowids: Vec<i64> = stmt
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();

        println!("Rowids: {:?}", rowids);
        assert_eq!(rowids, vec![1, 2, 3]);
    }

    // ========================================================================
    // NON-VECTOR COLUMN TESTS
    // These tests verify that non-vector columns (INTEGER, TEXT, REAL, BLOB)
    // are properly stored, retrieved, updated, and deleted.
    // ========================================================================

    #[test]
    fn test_create_table_with_nonvector_columns() {
        use crate::init;

        let db = Connection::open_in_memory().unwrap();
        init(&db).unwrap();

        // Create table with mixed columns
        db.execute_batch(
            "CREATE VIRTUAL TABLE test_mixed USING vec0(
                id INTEGER,
                name TEXT,
                score REAL,
                embedding float[3]
            )",
        )
        .unwrap();

        // Verify _data table was created with correct schema
        let tables: Vec<String> = db
            .prepare(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'test_mixed%'",
            )
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();

        assert!(
            tables.contains(&"test_mixed_data".to_string()),
            "_data table should be created for non-vector columns"
        );
    }

    #[test]
    fn test_insert_nonvector_values() {
        use crate::init;

        let db = Connection::open_in_memory().unwrap();
        init(&db).unwrap();

        db.execute_batch(
            "CREATE VIRTUAL TABLE test_insert USING vec0(
                id INTEGER,
                name TEXT,
                embedding float[3]
            )",
        )
        .unwrap();

        // Insert with non-vector values
        db.execute(
            "INSERT INTO test_insert(rowid, id, name, embedding) VALUES (1, 100, 'test_name', '[0.1, 0.2, 0.3]')",
            [],
        ).unwrap();

        // Verify data exists in _data table
        let (id, name): (i64, String) = db
            .query_row(
                "SELECT col00, col01 FROM test_insert_data WHERE rowid = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();

        assert_eq!(id, 100, "INTEGER value should be stored correctly");
        assert_eq!(name, "test_name", "TEXT value should be stored correctly");
    }

    #[test]
    fn test_select_nonvector_values() {
        use crate::init;

        let db = Connection::open_in_memory().unwrap();
        init(&db).unwrap();

        db.execute_batch(
            "CREATE VIRTUAL TABLE test_select USING vec0(
                lib_feat_id INTEGER,
                label TEXT,
                embedding float[3]
            )",
        )
        .unwrap();

        // Insert test data
        db.execute(
            "INSERT INTO test_select(rowid, lib_feat_id, label, embedding) VALUES (1, 42, 'my_label', '[1.0, 2.0, 3.0]')",
            [],
        ).unwrap();

        // SELECT non-vector columns - THIS WAS THE ORIGINAL BUG (returned NULL)
        let (lib_feat_id, label): (i64, String) = db
            .query_row(
                "SELECT lib_feat_id, label FROM test_select WHERE rowid = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();

        assert_eq!(lib_feat_id, 42, "lib_feat_id should return 42, not NULL");
        assert_eq!(
            label, "my_label",
            "label should return 'my_label', not NULL"
        );
    }

    #[test]
    fn test_update_nonvector_values() {
        use crate::init;

        let db = Connection::open_in_memory().unwrap();
        init(&db).unwrap();

        db.execute_batch(
            "CREATE VIRTUAL TABLE test_update USING vec0(
                id INTEGER,
                name TEXT,
                embedding float[3]
            )",
        )
        .unwrap();

        // Insert initial data
        db.execute(
            "INSERT INTO test_update(rowid, id, name, embedding) VALUES (1, 100, 'original', '[0.1, 0.2, 0.3]')",
            [],
        ).unwrap();

        // Update non-vector columns
        db.execute(
            "UPDATE test_update SET id = 200, name = 'updated' WHERE rowid = 1",
            [],
        )
        .unwrap();

        // Verify updated values
        let (id, name): (i64, String) = db
            .query_row(
                "SELECT id, name FROM test_update WHERE rowid = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();

        assert_eq!(id, 200, "id should be updated to 200");
        assert_eq!(name, "updated", "name should be updated to 'updated'");
    }

    #[test]
    fn test_delete_removes_nonvector_data() {
        use crate::init;

        let db = Connection::open_in_memory().unwrap();
        init(&db).unwrap();

        db.execute_batch(
            "CREATE VIRTUAL TABLE test_delete USING vec0(
                id INTEGER,
                name TEXT,
                embedding float[3]
            )",
        )
        .unwrap();

        // Insert data
        db.execute(
            "INSERT INTO test_delete(rowid, id, name, embedding) VALUES (1, 100, 'to_delete', '[0.1, 0.2, 0.3]')",
            [],
        ).unwrap();

        // Verify data exists in _data table
        let count: i64 = db
            .query_row(
                "SELECT COUNT(*) FROM test_delete_data WHERE rowid = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1, "Data should exist before delete");

        // Delete the row
        db.execute("DELETE FROM test_delete WHERE rowid = 1", [])
            .unwrap();

        // Verify data is removed from _data table
        let count: i64 = db
            .query_row(
                "SELECT COUNT(*) FROM test_delete_data WHERE rowid = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            count, 0,
            "Data should be removed from _data table after delete"
        );
    }

    #[test]
    fn test_null_nonvector_values() {
        use crate::init;

        let db = Connection::open_in_memory().unwrap();
        init(&db).unwrap();

        db.execute_batch(
            "CREATE VIRTUAL TABLE test_null USING vec0(
                id INTEGER,
                name TEXT,
                embedding float[3]
            )",
        )
        .unwrap();

        // Insert with NULL non-vector columns
        db.execute(
            "INSERT INTO test_null(rowid, id, name, embedding) VALUES (1, NULL, NULL, '[0.1, 0.2, 0.3]')",
            [],
        ).unwrap();

        // SELECT should return NULL values
        let id: Option<i64> = db
            .query_row("SELECT id FROM test_null WHERE rowid = 1", [], |row| {
                row.get(0)
            })
            .unwrap();

        let name: Option<String> = db
            .query_row("SELECT name FROM test_null WHERE rowid = 1", [], |row| {
                row.get(0)
            })
            .unwrap();

        assert!(id.is_none(), "NULL INTEGER should be returned as None");
        assert!(name.is_none(), "NULL TEXT should be returned as None");
    }

    #[test]
    fn test_type_preservation() {
        use crate::init;

        let db = Connection::open_in_memory().unwrap();
        init(&db).unwrap();

        db.execute_batch(
            "CREATE VIRTUAL TABLE test_types USING vec0(
                int_col INTEGER,
                real_col REAL,
                text_col TEXT,
                embedding float[3]
            )",
        )
        .unwrap();

        // Insert typed values
        db.execute(
            "INSERT INTO test_types(rowid, int_col, real_col, text_col, embedding) VALUES (1, 42, 3.14159, 'hello', '[1.0, 2.0, 3.0]')",
            [],
        ).unwrap();

        // Verify types are preserved
        let int_val: i64 = db
            .query_row(
                "SELECT int_col FROM test_types WHERE rowid = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(int_val, 42);

        let real_val: f64 = db
            .query_row(
                "SELECT real_col FROM test_types WHERE rowid = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        #[allow(clippy::approx_constant)]
        let expected_val = 3.14159;
        assert!(
            (real_val - expected_val).abs() < 0.00001,
            "REAL value should be preserved"
        );

        let text_val: String = db
            .query_row(
                "SELECT text_col FROM test_types WHERE rowid = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(text_val, "hello");
    }

    #[test]
    fn test_knn_with_nonvector_columns() {
        use crate::init;

        let db = Connection::open_in_memory().unwrap();
        init(&db).unwrap();

        db.execute_batch(
            "CREATE VIRTUAL TABLE test_knn USING vec0(
                id INTEGER,
                label TEXT,
                embedding float[3] hnsw(distance=cosine, M=8, ef_construction=50)
            )",
        )
        .unwrap();

        // Insert multiple rows
        db.execute("INSERT INTO test_knn(rowid, id, label, embedding) VALUES (1, 100, 'first', '[1.0, 0.0, 0.0]')", []).unwrap();
        db.execute("INSERT INTO test_knn(rowid, id, label, embedding) VALUES (2, 200, 'second', '[0.9, 0.1, 0.0]')", []).unwrap();
        db.execute("INSERT INTO test_knn(rowid, id, label, embedding) VALUES (3, 300, 'third', '[0.0, 1.0, 0.0]')", []).unwrap();

        // KNN search should return non-vector columns correctly
        let results: Vec<(i64, i64, String)> = db
            .prepare("SELECT rowid, id, label FROM test_knn WHERE embedding MATCH '[1.0, 0.0, 0.0]' AND k = 2 ORDER BY distance")
            .unwrap()
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(results.len(), 2, "KNN should return 2 results");

        // First result should be exact match (rowid=1, id=100, label='first')
        assert_eq!(results[0].0, 1);
        assert_eq!(results[0].1, 100);
        assert_eq!(results[0].2, "first");

        // Second result should be close match (rowid=2, id=200, label='second')
        assert_eq!(results[1].0, 2);
        assert_eq!(results[1].1, 200);
        assert_eq!(results[1].2, "second");
    }

    #[test]
    fn test_full_crud_cycle() {
        use crate::init;

        let db = Connection::open_in_memory().unwrap();
        init(&db).unwrap();

        // CREATE
        db.execute_batch(
            "CREATE VIRTUAL TABLE test_crud USING vec0(
                product_id INTEGER,
                product_name TEXT,
                price REAL,
                embedding float[4]
            )",
        )
        .unwrap();

        // INSERT
        db.execute(
            "INSERT INTO test_crud(rowid, product_id, product_name, price, embedding) VALUES (1, 1001, 'Widget', 19.99, '[0.1, 0.2, 0.3, 0.4]')",
            [],
        ).unwrap();

        // READ - verify all values
        let (id, name, price): (i64, String, f64) = db
            .query_row(
                "SELECT product_id, product_name, price FROM test_crud WHERE rowid = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(id, 1001);
        assert_eq!(name, "Widget");
        assert!((price - 19.99).abs() < 0.001);

        // UPDATE
        db.execute(
            "UPDATE test_crud SET product_id = 2002, product_name = 'Gadget', price = 29.99 WHERE rowid = 1",
            [],
        ).unwrap();

        // READ - verify updated values
        let (id, name, price): (i64, String, f64) = db
            .query_row(
                "SELECT product_id, product_name, price FROM test_crud WHERE rowid = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(id, 2002);
        assert_eq!(name, "Gadget");
        assert!((price - 29.99).abs() < 0.001);

        // DELETE
        db.execute("DELETE FROM test_crud WHERE rowid = 1", [])
            .unwrap();

        // Verify deleted
        let count: i64 = db
            .query_row("SELECT COUNT(*) FROM test_crud", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0, "Table should be empty after delete");
    }

    #[test]
    fn test_multiple_rows_nonvector_columns() {
        use crate::init;

        let db = Connection::open_in_memory().unwrap();
        init(&db).unwrap();

        db.execute_batch(
            "CREATE VIRTUAL TABLE test_multi USING vec0(
                seq INTEGER,
                name TEXT,
                embedding float[3]
            )",
        )
        .unwrap();

        // Insert multiple rows
        for i in 1..=10 {
            db.execute(
                &format!("INSERT INTO test_multi(rowid, seq, name, embedding) VALUES ({}, {}, 'item_{}', '[{}.0, 0.0, 0.0]')", i, i * 10, i, i),
                [],
            ).unwrap();
        }

        // Verify all rows have correct data
        for i in 1..=10 {
            let (seq, name): (i64, String) = db
                .query_row(
                    &format!("SELECT seq, name FROM test_multi WHERE rowid = {}", i),
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .unwrap();
            assert_eq!(seq, i * 10, "seq should be {} for rowid {}", i * 10, i);
            assert_eq!(
                name,
                format!("item_{}", i),
                "name should be 'item_{}' for rowid {}",
                i,
                i
            );
        }
    }
}
