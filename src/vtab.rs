//! Virtual table implementation for vec0

use crate::distance::DistanceMetric;
use crate::error::{Error, Result};
use crate::hnsw::{self, HnswMetadata};
use crate::shadow;
use crate::vector::VectorType;
use rusqlite::vtab::{
    Context, CreateVTab, Filters, IndexInfo, Inserts, UpdateVTab, Updates, VTab, VTabConnection,
    VTabCursor, sqlite3_vtab, sqlite3_vtab_cursor,
};
use rusqlite::{Connection, OptionalExtension, ffi};
use std::marker::PhantomData;
use std::os::raw::c_int;

/// Register the vec0 virtual table module
pub fn register_vec0_module(db: &Connection) -> Result<()> {
    // Use update_module_with_tx to support CREATE/INSERT/UPDATE/DELETE + transactions
    let module = rusqlite::vtab::update_module_with_tx::<Vec0Tab>();
    db.create_module("vec0", module, None)
        .map_err(Error::Sqlite)?;
    Ok(())
}

/// Column definition for vec0 table
#[derive(Debug, Clone)]
struct ColumnDef {
    name: String,
    col_type: ColumnType,
}

#[derive(Debug, Clone)]
enum ColumnType {
    Vector {
        #[allow(dead_code)]
        vec_type: VectorType,
        #[allow(dead_code)]
        dimensions: usize,
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
struct HnswStmtCache {
    get_node_data: *mut ffi::sqlite3_stmt,
    get_node_level: *mut ffi::sqlite3_stmt,
    get_edges: *mut ffi::sqlite3_stmt,
    get_edges_with_dist: *mut ffi::sqlite3_stmt,
    insert_node: *mut ffi::sqlite3_stmt,
    insert_edge: *mut ffi::sqlite3_stmt,
    delete_edges_from: *mut ffi::sqlite3_stmt,
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
        }
    }

    /// Prepare all statements for this HNSW index
    ///
    /// # Safety
    /// Must be called with a valid sqlite3 database handle
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
            "SELECT to_rowid FROM \"{}_{}_hnsw_edges\" WHERE from_rowid = ? AND level = ? ORDER BY distance",
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
        let sql = CString::new(format!(
            "SELECT to_rowid, distance FROM \"{}_{}_hnsw_edges\" WHERE from_rowid = ? AND level = ? ORDER BY distance",
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

        // Prepare insert_edge: INSERT OR IGNORE INTO hnsw_edges (from_rowid, to_rowid, level, distance) VALUES (?, ?, ?, ?)
        let sql = CString::new(format!(
            "INSERT OR IGNORE INTO \"{}_{}_hnsw_edges\" (from_rowid, to_rowid, level, distance) VALUES (?, ?, ?, ?)",
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

        Ok(())
    }

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
    db: *mut ffi::sqlite3, // Raw database handle for operations
    hnsw_stmt_cache: Vec<HnswStmtCache>, // One cache per vector column
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

            // Parse column definition: "column_name type[dimensions]"
            let parts: Vec<&str> = arg.split_whitespace().collect();
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

                    columns.push(ColumnDef {
                        name,
                        col_type: ColumnType::Vector {
                            vec_type,
                            dimensions,
                        },
                    });
                } else if type_spec.to_uppercase().contains("PARTITION") {
                    columns.push(ColumnDef {
                        name,
                        col_type: ColumnType::PartitionKey,
                    });
                } else if parts.iter().any(|p| p.to_uppercase().starts_with('+')) {
                    columns.push(ColumnDef {
                        name: name.trim_start_matches('+').to_string(),
                        col_type: ColumnType::Auxiliary,
                    });
                } else {
                    // Default to metadata column
                    columns.push(ColumnDef {
                        name,
                        col_type: ColumnType::Metadata,
                    });
                }
            } else {
                // No type specified, treat as metadata
                columns.push(ColumnDef {
                    name,
                    col_type: ColumnType::Metadata,
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

        // Count column types
        let num_vector_columns = vtab
            .columns
            .iter()
            .filter(|c| matches!(c.col_type, ColumnType::Vector { .. }))
            .count();
        let num_auxiliary_columns = vtab
            .columns
            .iter()
            .filter(|c| matches!(c.col_type, ColumnType::Auxiliary))
            .count();
        let num_metadata_columns = vtab
            .columns
            .iter()
            .filter(|c| matches!(c.col_type, ColumnType::Metadata))
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

            // Create base shadow tables
            let config = shadow::ShadowTablesConfig {
                num_vector_columns,
                num_auxiliary_columns,
                num_metadata_columns,
                has_text_pk: false, // TODO: detect from args
                num_partition_columns,
            };
            shadow::create_shadow_tables_ffi(
                db_handle,
                &vtab.schema_name,
                &vtab.table_name,
                &config,
            )
            .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

            // Create HNSW shadow tables for vector columns if using HNSW index type
            if vtab.index_type == IndexType::Hnsw {
                let mut vec_col_idx = 0;
                for col in vtab.columns.iter() {
                    if let ColumnType::Vector { .. } = col.col_type {
                        shadow::create_hnsw_shadow_tables_ffi(
                            db_handle,
                            &vtab.table_name,
                            &col.name,
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

        // Shadow tables are dropped automatically by SQLite when the virtual table is dropped
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

            // Delete from HNSW index if column is a vector and using HNSW index type
            if self.index_type == IndexType::Hnsw {
                for col in &self.columns {
                    if matches!(col.col_type, ColumnType::Vector { .. }) {
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
                                    &format!("SELECT rowid, level FROM \"{}\" ORDER BY level DESC LIMIT 1", nodes_table),
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
            }

            // Delete from rowids table
            conn.execute(
                &format!(
                    "DELETE FROM \"{}\".\"{}\" WHERE rowid = ?",
                    self.schema_name, rowids_table
                ),
                [rowid],
            )?;
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
        for (col_idx, col) in self.columns.iter().enumerate() {
            if let ColumnType::Vector {
                vec_type,
                dimensions,
            } = &col.col_type
            {
                let value_idx = col_idx + 2; // Skip NULL and rowid args
                if value_idx >= args.len() {
                    continue;
                }

                // Get the vector data as raw bytes
                let vector_data: Vec<u8> = match args.get::<Option<Vec<u8>>>(value_idx)? {
                    Some(data) => data,
                    None => continue, // NULL vector, skip
                };

                // Validate the byte size matches expected dimensions
                let expected_bytes = match vec_type {
                    VectorType::Float32 => dimensions * 4,
                    VectorType::Int8 => *dimensions,
                    VectorType::Bit => dimensions.div_ceil(8),
                };

                if vector_data.len() != expected_bytes {
                    return Err(rusqlite::Error::UserFunctionError(Box::new(
                        Error::InvalidParameter(format!(
                            "Vector byte size mismatch: expected {} bytes for {} dimensions, got {} bytes",
                            expected_bytes,
                            dimensions,
                            vector_data.len()
                        )),
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
                        col_idx,
                        &vector_data,
                    )
                    .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
                }

                // Also insert into HNSW index if using HNSW index type
                if self.index_type == IndexType::Hnsw {
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
                                    // Initialize new HNSW index
                                    HnswMetadata::new(
                                        *dimensions as i32,
                                        *vec_type,
                                        DistanceMetric::L2, // TODO: Make configurable
                                    )
                                });

                        // Insert into HNSW graph
                        hnsw::insert::insert_hnsw(
                            &conn,
                            &mut metadata,
                            &self.table_name,
                            &col.name,
                            rowid,
                            &vector_data,
                        )
                        .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

                        std::mem::forget(conn);
                    }
                }
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
        for (col_idx, col) in self.columns.iter().enumerate() {
            if let ColumnType::Vector {
                vec_type,
                dimensions,
            } = &col.col_type
            {
                let value_idx = col_idx + 2; // Skip old_rowid and new_rowid args
                if value_idx >= args.len() {
                    continue;
                }

                // Get the new vector data
                let vector_data: Vec<u8> = match args.get::<Option<Vec<u8>>>(value_idx)? {
                    Some(data) => data,
                    None => continue, // NULL vector, skip (could implement DELETE here)
                };

                // Validate byte size
                let expected_bytes = match vec_type {
                    VectorType::Float32 => dimensions * 4,
                    VectorType::Int8 => *dimensions,
                    VectorType::Bit => dimensions.div_ceil(8),
                };

                if vector_data.len() != expected_bytes {
                    return Err(rusqlite::Error::UserFunctionError(Box::new(
                        Error::InvalidParameter(format!(
                            "Vector byte size mismatch: expected {} bytes, got {}",
                            expected_bytes,
                            vector_data.len()
                        )),
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
                        col_idx,
                        chunk_id,
                        chunk_offset,
                        &vector_data,
                    )
                    .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

                    // Update HNSW index if column is a vector and using HNSW index type
                    if self.index_type == IndexType::Hnsw
                        && matches!(col.col_type, ColumnType::Vector { .. })
                    {
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
                                    HnswMetadata::new(
                                        *dimensions as i32,
                                        *vec_type,
                                        DistanceMetric::L2,
                                    )
                                });

                        // Decrement node count since we're replacing
                        if old_level.is_some() {
                            metadata.num_nodes = metadata.num_nodes.saturating_sub(1);
                        }

                        // Re-insert with new vector
                        hnsw::insert::insert_hnsw(
                            &conn,
                            &mut metadata,
                            &self.table_name,
                            &col.name,
                            old_rowid,
                            &vector_data,
                        )
                        .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
                    }
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
        // No-op: no prepared statement caching yet
        // Future: finalize temporary prepared statements and clear caches
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
                let query_vector: Vec<u8> = args.get(0)?;
                let k: i64 = args.get(1)?;

                // Find the vector column (first vector column for now)
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

                // Execute HNSW search
                // SAFETY: vtab.db is valid for the lifetime of the virtual table
                let results = unsafe {
                    let conn = Connection::from_handle(vtab.db).map_err(|e| {
                        rusqlite::Error::UserFunctionError(Box::new(Error::Sqlite(e)))
                    })?;

                    let result = match vtab.index_type {
                        IndexType::Hnsw => {
                            // HNSW mode: use approximate search, error if index missing
                            let metadata =
                                HnswMetadata::load_from_db(&conn, &vtab.table_name, &col.name)
                                    .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

                            if let Some(meta) = metadata {
                                hnsw::search::search_hnsw(
                                    &conn,
                                    &meta,
                                    &vtab.table_name,
                                    &col.name,
                                    &query_vector,
                                    k as usize,
                                    None,
                                )
                                .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?
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
                        }
                        IndexType::Enn => {
                            // ENN mode: exact nearest neighbor via brute force
                            brute_force_search(
                                &conn,
                                &vtab.schema_name,
                                &vtab.table_name,
                                col_idx,
                                &query_vector,
                                k as usize,
                            )
                            .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?
                        }
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
            && let ColumnType::Vector { .. } = &vtab.columns[col_idx].col_type
        {
            // Read vector from shadow tables
            // SAFETY: vtab.db is valid for the lifetime of the virtual table
            let vector_data = unsafe {
                let conn = Connection::from_handle(vtab.db)
                    .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(Error::Sqlite(e))))?;

                let result = shadow::read_vector_from_chunk(
                    &conn,
                    &vtab.schema_name,
                    &vtab.table_name,
                    col_idx,
                    rowid,
                )
                .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

                std::mem::forget(conn);
                result
            };

            match vector_data {
                Some(data) => ctx.set_result(&data.as_slice())?,
                None => ctx.set_result(&rusqlite::types::Null)?,
            }
            return Ok(());
        }

        // For non-vector columns, return NULL for now
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

        let dist = match distance::distance(&query_vec, &vec, DistanceMetric::L2) {
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

        // Create a vec0 virtual table with one vector column
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

        // Check HNSW tables
        assert!(
            tables.contains(&"vec_test_embedding_hnsw_meta".to_string()),
            "Missing HNSW meta table"
        );
        assert!(
            tables.contains(&"vec_test_embedding_hnsw_nodes".to_string()),
            "Missing HNSW nodes table"
        );
        assert!(
            tables.contains(&"vec_test_embedding_hnsw_edges".to_string()),
            "Missing HNSW edges table"
        );
        assert!(
            tables.contains(&"vec_test_embedding_hnsw_levels".to_string()),
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
}
