//! Shadow table management for vec0 virtual tables
//!
//! The vec0 virtual table uses shadow tables for persistent storage:
//! - {table}_data: All columns (vectors + non-vectors) in unified storage
//! - {table}_info: Version metadata
//! - {table}_{column}_hnsw_meta: HNSW index metadata
//! - {table}_{column}_hnsw_nodes: HNSW graph nodes (rowid, level, vector)
//! - {table}_{column}_hnsw_edges: HNSW graph edges

use crate::error::{Error, Result};
use rusqlite::{Connection, OptionalExtension, ffi};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

/// Definition of a vector column for the _data table
#[derive(Debug, Clone)]
pub struct VectorColumnDef {
    pub name: String,
    pub dimensions: usize,
    pub element_size: usize, // 4 for float32, 1 for int8, etc.
}

/// Definition of a non-vector column for the _data table
#[derive(Debug, Clone)]
pub struct DataColumnDef {
    pub name: String,
    pub col_type: String, // "INTEGER", "TEXT", "REAL", "BLOB"
}

/// Configuration for shadow table creation
pub struct ShadowTablesConfig {
    /// Vector column definitions
    pub vector_columns: Vec<VectorColumnDef>,
    /// Non-vector column definitions for the unified _data table
    pub data_columns: Vec<DataColumnDef>,
}

/// Drop all shadow tables for a vec0 virtual table (cleanup before creation)
///
/// # Safety
/// This function must be called with a valid sqlite3 database handle
pub unsafe fn drop_shadow_tables_ffi(
    db: *mut ffi::sqlite3,
    schema: &str,
    table_name: &str,
    config: &ShadowTablesConfig,
    vector_column_names: &[&str],
) -> Result<()> {
    // Build list of all possible shadow tables (old and new schema)
    let mut tables_to_drop = vec![
        // New unified schema
        format!("\"{}\".\"{}_data\"", schema, table_name),
        format!("\"{}\".\"{}_info\"", schema, table_name),
        // Old chunked schema (for migration)
        format!("\"{}\".\"{}_chunks\"", schema, table_name),
        format!("\"{}\".\"{}_rowids\"", schema, table_name),
        format!("\"{}\".\"{}_auxiliary\"", schema, table_name),
    ];

    // Old vector chunk tables (for migration)
    for i in 0..config.vector_columns.len().max(4) {
        tables_to_drop.push(format!(
            "\"{}\".\"{}_vector_chunks{:02}\"",
            schema, table_name, i
        ));
    }

    // Old metadata tables (for migration)
    for i in 0..16 {
        tables_to_drop.push(format!(
            "\"{}\".\"{}_metadatachunks{:02}\"",
            schema, table_name, i
        ));
        tables_to_drop.push(format!(
            "\"{}\".\"{}_metadatatext{:02}\"",
            schema, table_name, i
        ));
    }

    // HNSW tables for each vector column
    for col_name in vector_column_names {
        tables_to_drop.push(format!("\"{}_{}_hnsw_meta\"", table_name, col_name));
        tables_to_drop.push(format!("\"{}_{}_hnsw_nodes\"", table_name, col_name));
        tables_to_drop.push(format!("\"{}_{}_hnsw_edges\"", table_name, col_name));
    }

    // Drop each table (ignore errors - table may not exist)
    for table in tables_to_drop {
        let drop_sql = format!("DROP TABLE IF EXISTS {}", table);
        // SAFETY: db is a valid sqlite3 handle passed to this unsafe function
        let _ = unsafe { execute_sql_ffi(db, &drop_sql) };
    }

    Ok(())
}

/// Create all shadow tables for a vec0 virtual table
///
/// # Arguments
/// * `db` - Database connection
/// * `schema` - Schema name (typically "main")
/// * `table_name` - Name of the virtual table
/// * `config` - Shadow table configuration
pub fn create_shadow_tables(
    db: &Connection,
    schema: &str,
    table_name: &str,
    config: &ShadowTablesConfig,
) -> Result<()> {
    // Create unified _data table for ALL columns (vectors + non-vectors)
    // Schema: (rowid INTEGER PRIMARY KEY, vec00 BLOB, vec01 BLOB, ..., col00 TYPE, col01 TYPE, ...)
    let mut data_sql = format!(
        "CREATE TABLE \"{}\".\"{}_data\" (rowid INTEGER PRIMARY KEY",
        schema, table_name
    );

    // Add vector columns (stored as BLOB)
    for i in 0..config.vector_columns.len() {
        data_sql.push_str(&format!(", vec{:02} BLOB", i));
    }

    // Add non-vector columns with their types
    for (i, col) in config.data_columns.iter().enumerate() {
        data_sql.push_str(&format!(", col{:02} {}", i, col.col_type));
    }
    data_sql.push_str(");");

    db.execute(&data_sql, []).map_err(Error::Sqlite)?;

    // Create _info shadow table (stores version metadata)
    let info_sql = format!(
        "CREATE TABLE \"{}\".\"{}_info\" (\
         key TEXT PRIMARY KEY, \
         value\
         );",
        schema, table_name
    );
    db.execute(&info_sql, []).map_err(Error::Sqlite)?;

    // Populate _info table with version information
    db.execute(
        &format!(
            "INSERT INTO \"{}\".\"{}_info\" (key, value) VALUES ('CREATE_VERSION', '0.2.0')",
            schema, table_name
        ),
        [],
    )
    .map_err(Error::Sqlite)?;
    db.execute(
        &format!(
            "INSERT INTO \"{}\".\"{}_info\" (key, value) VALUES ('CREATE_VERSION_MAJOR', 0)",
            schema, table_name
        ),
        [],
    )
    .map_err(Error::Sqlite)?;
    db.execute(
        &format!(
            "INSERT INTO \"{}\".\"{}_info\" (key, value) VALUES ('CREATE_VERSION_MINOR', 2)",
            schema, table_name
        ),
        [],
    )
    .map_err(Error::Sqlite)?;
    db.execute(
        &format!(
            "INSERT INTO \"{}\".\"{}_info\" (key, value) VALUES ('CREATE_VERSION_PATCH', 0)",
            schema, table_name
        ),
        [],
    )
    .map_err(Error::Sqlite)?;
    db.execute(
        &format!(
            "INSERT INTO \"{}\".\"{}_info\" (key, value) VALUES ('STORAGE_SCHEMA', 'unified')",
            schema, table_name
        ),
        [],
    )
    .map_err(Error::Sqlite)?;

    Ok(())
}

/// Create HNSW shadow tables for a vector column
///
/// # Arguments
/// * `db` - Database connection
/// * `table_name` - Name of the virtual table
/// * `column_name` - Name of the vector column
pub fn create_hnsw_shadow_tables(
    db: &Connection,
    table_name: &str,
    column_name: &str,
) -> Result<()> {
    // Create metadata table (single row schema - much simpler than key-value)
    let meta_sql = format!(
        "CREATE TABLE IF NOT EXISTS \"{}_{}_hnsw_meta\" (\
         id INTEGER PRIMARY KEY CHECK (id = 1), \
         m INTEGER NOT NULL DEFAULT 32, \
         max_m0 INTEGER NOT NULL DEFAULT 64, \
         ef_construction INTEGER NOT NULL DEFAULT 400, \
         ef_search INTEGER NOT NULL DEFAULT 200, \
         max_level INTEGER NOT NULL DEFAULT 16, \
         level_factor REAL NOT NULL DEFAULT 0.28768207245178085, \
         entry_point_rowid INTEGER NOT NULL DEFAULT -1, \
         entry_point_level INTEGER NOT NULL DEFAULT -1, \
         num_nodes INTEGER NOT NULL DEFAULT 0, \
         dimensions INTEGER NOT NULL DEFAULT 0, \
         element_type TEXT NOT NULL DEFAULT 'float32', \
         distance_metric TEXT NOT NULL DEFAULT 'l2', \
         rng_seed INTEGER NOT NULL DEFAULT 12345, \
         hnsw_version INTEGER NOT NULL DEFAULT 1, \
         index_quantization TEXT NOT NULL DEFAULT 'none', \
         normalize_vectors INTEGER NOT NULL DEFAULT 1\
         )",
        table_name, column_name
    );
    db.execute(&meta_sql, []).map_err(Error::Sqlite)?;

    // Insert default metadata row (uses all defaults)
    let insert_meta_sql = format!(
        "INSERT OR IGNORE INTO \"{}_{}_hnsw_meta\" (id) VALUES (1)",
        table_name, column_name
    );
    db.execute(&insert_meta_sql, []).map_err(Error::Sqlite)?;

    // Create nodes table
    let nodes_sql = format!(
        "CREATE TABLE IF NOT EXISTS \"{}_{}_hnsw_nodes\" (\
         rowid INTEGER PRIMARY KEY, \
         level INTEGER NOT NULL, \
         vector BLOB, \
         created_at INTEGER DEFAULT (unixepoch())\
         )",
        table_name, column_name
    );
    db.execute(&nodes_sql, []).map_err(Error::Sqlite)?;

    // Create edges table - WITHOUT ROWID clusters edges by (from_rowid, level)
    // for efficient neighbor lookups. No separate index needed.
    // Distance column enables O(1) prune without vector fetching.
    let edges_sql = format!(
        "CREATE TABLE IF NOT EXISTS \"{}_{}_hnsw_edges\" (\
         from_rowid INTEGER NOT NULL, \
         to_rowid INTEGER NOT NULL, \
         level INTEGER NOT NULL, \
         distance REAL NOT NULL DEFAULT 0.0, \
         PRIMARY KEY (from_rowid, level, to_rowid)\
         ) WITHOUT ROWID",
        table_name, column_name
    );
    db.execute(&edges_sql, []).map_err(Error::Sqlite)?;

    Ok(())
}

/// Execute SQL using raw FFI
unsafe fn execute_sql_ffi(db: *mut ffi::sqlite3, sql: &str) -> Result<()> {
    let c_sql =
        CString::new(sql).map_err(|e| Error::InvalidParameter(format!("Invalid SQL: {}", e)))?;
    let mut err_msg: *mut c_char = ptr::null_mut();

    // SAFETY: We're calling sqlite3_exec with a valid database handle and SQL string
    let rc = unsafe { ffi::sqlite3_exec(db, c_sql.as_ptr(), None, ptr::null_mut(), &mut err_msg) };

    if rc != ffi::SQLITE_OK {
        let error = if !err_msg.is_null() {
            // SAFETY: err_msg is valid and non-null
            let err_str = unsafe { CStr::from_ptr(err_msg).to_string_lossy().to_string() };
            // SAFETY: Free the error message allocated by SQLite
            unsafe {
                ffi::sqlite3_free(err_msg as *mut _);
            }
            Error::Sqlite(rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rc),
                Some(err_str),
            ))
        } else {
            Error::Sqlite(rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rc),
                None,
            ))
        };
        return Err(error);
    }

    Ok(())
}

/// Create all shadow tables using raw FFI (for use in virtual table creation)
///
/// # Safety
/// This function must be called with a valid sqlite3 database handle
pub unsafe fn create_shadow_tables_ffi(
    db: *mut ffi::sqlite3,
    schema: &str,
    table_name: &str,
    config: &ShadowTablesConfig,
) -> Result<()> {
    // Create unified _data table for ALL columns (vectors + non-vectors)
    let mut data_sql = format!(
        "CREATE TABLE \"{}\".\"{}_data\" (rowid INTEGER PRIMARY KEY",
        schema, table_name
    );

    // Add vector columns (stored as BLOB)
    for i in 0..config.vector_columns.len() {
        data_sql.push_str(&format!(", vec{:02} BLOB", i));
    }

    // Add non-vector columns with their types
    for (i, col) in config.data_columns.iter().enumerate() {
        data_sql.push_str(&format!(", col{:02} {}", i, col.col_type));
    }
    data_sql.push_str(");");

    // SAFETY: execute_sql_ffi is called with a valid database handle
    unsafe { execute_sql_ffi(db, &data_sql)? };

    // Create _info shadow table (stores version metadata)
    let info_sql = format!(
        "CREATE TABLE \"{}\".\"{}_info\" (\
         key TEXT PRIMARY KEY, \
         value\
         );",
        schema, table_name
    );
    unsafe { execute_sql_ffi(db, &info_sql)? };

    // Populate _info table with version information
    let version_sqls = vec![
        format!(
            "INSERT INTO \"{}\".\"{}_info\" (key, value) VALUES ('CREATE_VERSION', '0.2.0')",
            schema, table_name
        ),
        format!(
            "INSERT INTO \"{}\".\"{}_info\" (key, value) VALUES ('CREATE_VERSION_MAJOR', 0)",
            schema, table_name
        ),
        format!(
            "INSERT INTO \"{}\".\"{}_info\" (key, value) VALUES ('CREATE_VERSION_MINOR', 2)",
            schema, table_name
        ),
        format!(
            "INSERT INTO \"{}\".\"{}_info\" (key, value) VALUES ('CREATE_VERSION_PATCH', 0)",
            schema, table_name
        ),
        format!(
            "INSERT INTO \"{}\".\"{}_info\" (key, value) VALUES ('STORAGE_SCHEMA', 'unified')",
            schema, table_name
        ),
    ];
    for sql in version_sqls {
        unsafe { execute_sql_ffi(db, &sql)? };
    }

    Ok(())
}

/// Create HNSW shadow tables using raw FFI
///
/// # Safety
/// This function must be called with a valid sqlite3 database handle
pub unsafe fn create_hnsw_shadow_tables_ffi(
    db: *mut ffi::sqlite3,
    table_name: &str,
    column_name: &str,
) -> Result<()> {
    // Delegate to the new function with default metadata values
    unsafe {
        create_hnsw_shadow_tables_with_metadata_ffi(
            db,
            table_name,
            column_name,
            0, // dimensions (0 = unknown, will be set on first insert)
            crate::vector::VectorType::Float32,
            crate::distance::DistanceMetric::L2,
            crate::vector::IndexQuantization::None,
        )
    }
}

/// Create HNSW shadow tables with specific metadata using raw FFI
///
/// # Arguments
/// * `db` - Database handle
/// * `table_name` - Name of the virtual table
/// * `column_name` - Name of the vector column
/// * `dimensions` - Vector dimensions
/// * `element_type` - Vector element type (float32, int8, bit)
/// * `distance_metric` - Distance metric (l2, cosine, l1)
/// * `index_quantization` - Index quantization mode (none, int8)
///
/// # Safety
/// This function must be called with a valid sqlite3 database handle
pub unsafe fn create_hnsw_shadow_tables_with_metadata_ffi(
    db: *mut ffi::sqlite3,
    table_name: &str,
    column_name: &str,
    dimensions: i32,
    element_type: crate::vector::VectorType,
    distance_metric: crate::distance::DistanceMetric,
    index_quantization: crate::vector::IndexQuantization,
) -> Result<()> {
    // Create metadata table (single row schema - much simpler than key-value)
    let meta_sql = format!(
        "CREATE TABLE IF NOT EXISTS \"{}_{}_hnsw_meta\" (\
         id INTEGER PRIMARY KEY CHECK (id = 1), \
         m INTEGER NOT NULL DEFAULT 32, \
         max_m0 INTEGER NOT NULL DEFAULT 64, \
         ef_construction INTEGER NOT NULL DEFAULT 400, \
         ef_search INTEGER NOT NULL DEFAULT 200, \
         max_level INTEGER NOT NULL DEFAULT 16, \
         level_factor REAL NOT NULL DEFAULT 0.28768207245178085, \
         entry_point_rowid INTEGER NOT NULL DEFAULT -1, \
         entry_point_level INTEGER NOT NULL DEFAULT -1, \
         num_nodes INTEGER NOT NULL DEFAULT 0, \
         dimensions INTEGER NOT NULL DEFAULT 0, \
         element_type TEXT NOT NULL DEFAULT 'float32', \
         distance_metric TEXT NOT NULL DEFAULT 'l2', \
         rng_seed INTEGER NOT NULL DEFAULT 12345, \
         hnsw_version INTEGER NOT NULL DEFAULT 1, \
         index_quantization TEXT NOT NULL DEFAULT 'none', \
         normalize_vectors INTEGER NOT NULL DEFAULT 1\
         )",
        table_name, column_name
    );
    // SAFETY: execute_sql_ffi is called with a valid database handle
    unsafe { execute_sql_ffi(db, &meta_sql)? };

    // Generate random seed for HNSW level generation
    use std::collections::hash_map::RandomState;
    use std::hash::BuildHasher;
    let random_state = RandomState::new();
    let rng_seed = random_state.hash_one(std::time::SystemTime::now()) as i64;

    // Insert metadata row with actual column values
    // normalize_vectors is 1 (true) for cosine distance to enable L2 internal optimization
    let normalize_vectors = if distance_metric == crate::distance::DistanceMetric::Cosine {
        1
    } else {
        0
    };
    let insert_meta_sql = format!(
        "INSERT OR IGNORE INTO \"{}_{}_hnsw_meta\" \
         (id, dimensions, element_type, distance_metric, index_quantization, rng_seed, normalize_vectors) \
         VALUES (1, {}, '{}', '{}', '{}', {}, {})",
        table_name,
        column_name,
        dimensions,
        element_type.as_str(),
        distance_metric.as_str(),
        index_quantization.as_str(),
        rng_seed,
        normalize_vectors
    );
    // SAFETY: execute_sql_ffi is called with a valid database handle
    unsafe { execute_sql_ffi(db, &insert_meta_sql)? };

    // Create nodes table
    let nodes_sql = format!(
        "CREATE TABLE IF NOT EXISTS \"{}_{}_hnsw_nodes\" (\
         rowid INTEGER PRIMARY KEY, \
         level INTEGER NOT NULL, \
         vector BLOB, \
         created_at INTEGER DEFAULT (unixepoch())\
         )",
        table_name, column_name
    );
    // SAFETY: execute_sql_ffi is called with a valid database handle
    unsafe { execute_sql_ffi(db, &nodes_sql)? };

    // Create edges table - WITHOUT ROWID clusters edges by (from_rowid, level)
    // for efficient neighbor lookups. No separate index needed.
    // Distance column enables O(1) prune without vector fetching.
    let edges_sql = format!(
        "CREATE TABLE IF NOT EXISTS \"{}_{}_hnsw_edges\" (\
         from_rowid INTEGER NOT NULL, \
         to_rowid INTEGER NOT NULL, \
         level INTEGER NOT NULL, \
         distance REAL NOT NULL DEFAULT 0.0, \
         PRIMARY KEY (from_rowid, level, to_rowid)\
         ) WITHOUT ROWID",
        table_name, column_name
    );
    // SAFETY: execute_sql_ffi is called with a valid database handle
    unsafe { execute_sql_ffi(db, &edges_sql)? };

    Ok(())
}

/// Create HNSW shadow tables with custom parameters using raw FFI
///
/// Like `create_hnsw_shadow_tables_with_metadata_ffi` but allows custom M and ef_construction.
///
/// # Safety
/// This function must be called with a valid sqlite3 database handle
#[allow(clippy::too_many_arguments)]
pub unsafe fn create_hnsw_shadow_tables_with_params_ffi(
    db: *mut ffi::sqlite3,
    table_name: &str,
    column_name: &str,
    dimensions: i32,
    element_type: crate::vector::VectorType,
    distance_metric: crate::distance::DistanceMetric,
    index_quantization: crate::vector::IndexQuantization,
    custom_m: Option<i32>,
    custom_ef_construction: Option<i32>,
) -> Result<()> {
    let m = custom_m.unwrap_or(32);
    let max_m0 = m * 2; // Standard HNSW: max_m0 = 2*M
    let ef_construction = custom_ef_construction.unwrap_or(400);

    // Create metadata table
    let meta_sql = format!(
        "CREATE TABLE IF NOT EXISTS \"{}_{}_hnsw_meta\" (\
         id INTEGER PRIMARY KEY CHECK (id = 1), \
         m INTEGER NOT NULL DEFAULT 32, \
         max_m0 INTEGER NOT NULL DEFAULT 64, \
         ef_construction INTEGER NOT NULL DEFAULT 400, \
         ef_search INTEGER NOT NULL DEFAULT 200, \
         max_level INTEGER NOT NULL DEFAULT 16, \
         level_factor REAL NOT NULL DEFAULT 0.28768207245178085, \
         entry_point_rowid INTEGER NOT NULL DEFAULT -1, \
         entry_point_level INTEGER NOT NULL DEFAULT -1, \
         num_nodes INTEGER NOT NULL DEFAULT 0, \
         dimensions INTEGER NOT NULL DEFAULT 0, \
         element_type TEXT NOT NULL DEFAULT 'float32', \
         distance_metric TEXT NOT NULL DEFAULT 'l2', \
         rng_seed INTEGER NOT NULL DEFAULT 12345, \
         hnsw_version INTEGER NOT NULL DEFAULT 1, \
         index_quantization TEXT NOT NULL DEFAULT 'none', \
         normalize_vectors INTEGER NOT NULL DEFAULT 1\
         )",
        table_name, column_name
    );
    // SAFETY: execute_sql_ffi is called with a valid database handle
    unsafe { execute_sql_ffi(db, &meta_sql)? };

    // Generate random seed for HNSW level generation
    use std::collections::hash_map::RandomState;
    use std::hash::BuildHasher;
    let random_state = RandomState::new();
    let rng_seed = random_state.hash_one(std::time::SystemTime::now()) as i64;

    // Insert metadata row with custom M and ef_construction values
    // Enable normalization for Cosine distance to use L2 internally for better performance
    let normalize_vectors = if distance_metric == crate::distance::DistanceMetric::Cosine {
        1
    } else {
        0
    };
    let insert_meta_sql = format!(
        "INSERT OR IGNORE INTO \"{}_{}_hnsw_meta\" \
         (id, m, max_m0, ef_construction, dimensions, element_type, distance_metric, index_quantization, rng_seed, normalize_vectors) \
         VALUES (1, {}, {}, {}, {}, '{}', '{}', '{}', {}, {})",
        table_name,
        column_name,
        m,
        max_m0,
        ef_construction,
        dimensions,
        element_type.as_str(),
        distance_metric.as_str(),
        index_quantization.as_str(),
        rng_seed,
        normalize_vectors
    );
    // SAFETY: execute_sql_ffi is called with a valid database handle
    unsafe { execute_sql_ffi(db, &insert_meta_sql)? };

    // Create nodes table
    let nodes_sql = format!(
        "CREATE TABLE IF NOT EXISTS \"{}_{}_hnsw_nodes\" (\
         rowid INTEGER PRIMARY KEY, \
         level INTEGER NOT NULL, \
         vector BLOB, \
         created_at INTEGER DEFAULT (unixepoch())\
         )",
        table_name, column_name
    );
    // SAFETY: execute_sql_ffi is called with a valid database handle
    unsafe { execute_sql_ffi(db, &nodes_sql)? };

    // Create edges table with distance column for O(1) prune
    let edges_sql = format!(
        "CREATE TABLE IF NOT EXISTS \"{}_{}_hnsw_edges\" (\
         from_rowid INTEGER NOT NULL, \
         to_rowid INTEGER NOT NULL, \
         level INTEGER NOT NULL, \
         distance REAL NOT NULL DEFAULT 0.0, \
         PRIMARY KEY (from_rowid, level, to_rowid)\
         ) WITHOUT ROWID",
        table_name, column_name
    );
    // SAFETY: execute_sql_ffi is called with a valid database handle
    unsafe { execute_sql_ffi(db, &edges_sql)? };

    Ok(())
}

/// Check if a table name is a shadow table for vec0
pub fn is_shadow_table(name: &str) -> bool {
    const SHADOWS: &[&str] = &[
        "_data",
        "_info",
        // Old schema (for compatibility checks)
        "_rowids",
        "_chunks",
        "_auxiliary",
        "_metadatachunks00",
        "_metadatachunks01",
        "_metadatachunks02",
        "_metadatachunks03",
        "_metadatatext00",
        "_metadatatext01",
        "_metadatatext02",
        "_metadatatext03",
        "_vector_chunks00",
        "_vector_chunks01",
        "_vector_chunks02",
        "_vector_chunks03",
        "_hnsw_meta",
        "_hnsw_nodes",
        "_hnsw_edges",
    ];

    SHADOWS.iter().any(|&suffix| name.ends_with(suffix))
}

/// Insert a row into the unified _data table
///
/// # Arguments
/// * `db` - Database connection
/// * `schema` - Schema name
/// * `table_name` - Table name
/// * `rowid` - Row ID
/// * `vectors` - Vector data as bytes (one per vector column)
/// * `columns` - Non-vector column values
pub fn insert_row(
    db: &Connection,
    schema: &str,
    table_name: &str,
    rowid: i64,
    vectors: &[&[u8]],
    columns: &[rusqlite::types::Value],
) -> Result<()> {
    let data_table = format!("{}_data", table_name);

    // Build column names and placeholders
    let mut col_names = String::from("rowid");
    let mut placeholders = String::from("?");

    for i in 0..vectors.len() {
        col_names.push_str(&format!(", vec{:02}", i));
        placeholders.push_str(", ?");
    }

    for i in 0..columns.len() {
        col_names.push_str(&format!(", col{:02}", i));
        placeholders.push_str(", ?");
    }

    let insert_sql = format!(
        "INSERT OR REPLACE INTO \"{}\".\"{}\" ({}) VALUES ({})",
        schema, data_table, col_names, placeholders
    );

    // Build parameter list
    let mut params: Vec<rusqlite::types::Value> = vec![rowid.into()];
    for v in vectors {
        params.push(rusqlite::types::Value::Blob(v.to_vec()));
    }
    for c in columns {
        params.push(c.clone());
    }

    db.execute(&insert_sql, rusqlite::params_from_iter(params))
        .map_err(Error::Sqlite)?;

    Ok(())
}

/// Insert a row into the unified _data table using raw FFI
///
/// # Safety
/// Must be called with a valid sqlite3 database handle
pub unsafe fn insert_row_ffi(
    db: *mut ffi::sqlite3,
    schema: &str,
    table_name: &str,
    rowid: i64,
    vectors: &[&[u8]],
    columns: &[rusqlite::types::Value],
) -> Result<()> {
    // Wrap the raw handle in a Connection for our helper functions
    // SAFETY: We're creating a non-owning Connection from the handle
    let conn = unsafe { Connection::from_handle(db).map_err(Error::Sqlite)? };

    insert_row(&conn, schema, table_name, rowid, vectors, columns)?;

    // Forget the Connection to avoid double-free
    std::mem::forget(conn);

    Ok(())
}

/// Read a vector from the unified _data table
///
/// # Arguments
/// * `db` - Database connection
/// * `schema` - Schema name
/// * `table_name` - Table name
/// * `vec_idx` - Vector column index (0, 1, 2, ...)
/// * `rowid` - Rowid of the vector to read
///
/// # Returns
/// Vector data as bytes, or None if not found
pub fn read_vector(
    db: &Connection,
    schema: &str,
    table_name: &str,
    vec_idx: usize,
    rowid: i64,
) -> Result<Option<Vec<u8>>> {
    let data_table = format!("{}_data", table_name);
    let query = format!(
        "SELECT vec{:02} FROM \"{}\".\"{}\" WHERE rowid = ?",
        vec_idx, schema, data_table
    );

    let result = db
        .query_row(&query, [rowid], |row| row.get::<_, Option<Vec<u8>>>(0))
        .optional()
        .map_err(Error::Sqlite)?;

    Ok(result.flatten())
}

/// Read a non-vector column from the unified _data table
///
/// # Arguments
/// * `db` - Database connection
/// * `schema` - Schema name
/// * `table_name` - Table name
/// * `col_idx` - Non-vector column index (0, 1, 2, ...)
/// * `rowid` - Rowid of the row to read
///
/// # Returns
/// Column value, or None if not found
pub fn read_column(
    db: &Connection,
    schema: &str,
    table_name: &str,
    col_idx: usize,
    rowid: i64,
) -> Result<Option<rusqlite::types::Value>> {
    let data_table = format!("{}_data", table_name);
    let query = format!(
        "SELECT col{:02} FROM \"{}\".\"{}\" WHERE rowid = ?",
        col_idx, schema, data_table
    );

    let result = db
        .query_row(&query, [rowid], |row| {
            row.get::<_, rusqlite::types::Value>(0)
        })
        .optional()
        .map_err(Error::Sqlite)?;

    Ok(result)
}

/// Update a row in the unified _data table
///
/// # Arguments
/// * `db` - Database connection
/// * `schema` - Schema name
/// * `table_name` - Table name
/// * `rowid` - Row ID
/// * `vectors` - Vector data as bytes (one per vector column, or None to skip)
/// * `columns` - Non-vector column values (or None to skip)
pub fn update_row(
    db: &Connection,
    schema: &str,
    table_name: &str,
    rowid: i64,
    vectors: &[Option<&[u8]>],
    columns: &[Option<rusqlite::types::Value>],
) -> Result<()> {
    let data_table = format!("{}_data", table_name);

    // Build SET clause and parameters
    let mut set_clauses = Vec::new();
    let mut params: Vec<rusqlite::types::Value> = Vec::new();

    for (i, v) in vectors.iter().enumerate() {
        if let Some(data) = v {
            set_clauses.push(format!("vec{:02} = ?", i));
            params.push(rusqlite::types::Value::Blob(data.to_vec()));
        }
    }

    for (i, c) in columns.iter().enumerate() {
        if let Some(val) = c {
            set_clauses.push(format!("col{:02} = ?", i));
            params.push(val.clone());
        }
    }

    if set_clauses.is_empty() {
        return Ok(()); // Nothing to update
    }

    // Add rowid for WHERE clause
    params.push(rowid.into());

    let update_sql = format!(
        "UPDATE \"{}\".\"{}\" SET {} WHERE rowid = ?",
        schema,
        data_table,
        set_clauses.join(", ")
    );

    db.execute(&update_sql, rusqlite::params_from_iter(params))
        .map_err(Error::Sqlite)?;

    Ok(())
}

/// Delete a row from the unified _data table
///
/// # Arguments
/// * `db` - Database connection
/// * `schema` - Schema name
/// * `table_name` - Table name
/// * `rowid` - Row ID to delete
pub fn delete_row(db: &Connection, schema: &str, table_name: &str, rowid: i64) -> Result<()> {
    let data_table = format!("{}_data", table_name);
    let delete_sql = format!(
        "DELETE FROM \"{}\".\"{}\" WHERE rowid = ?",
        schema, data_table
    );

    db.execute(&delete_sql, [rowid]).map_err(Error::Sqlite)?;

    Ok(())
}

/// Get all rowids from the _data table (for full scan)
pub fn get_all_rowids(db: &Connection, schema: &str, table_name: &str) -> Result<Vec<i64>> {
    let data_table = format!("{}_data", table_name);
    let query = format!(
        "SELECT rowid FROM \"{}\".\"{}\" ORDER BY rowid",
        schema, data_table
    );

    let mut stmt = db.prepare(&query).map_err(Error::Sqlite)?;
    let rowids = stmt
        .query_map([], |row| row.get(0))
        .map_err(Error::Sqlite)?
        .collect::<std::result::Result<Vec<i64>, _>>()
        .map_err(Error::Sqlite)?;

    Ok(rowids)
}

/// Check if a row exists in the _data table
pub fn row_exists(db: &Connection, schema: &str, table_name: &str, rowid: i64) -> Result<bool> {
    let data_table = format!("{}_data", table_name);
    let query = format!(
        "SELECT 1 FROM \"{}\".\"{}\" WHERE rowid = ?",
        schema, data_table
    );

    let exists = db
        .query_row(&query, [rowid], |_| Ok(()))
        .optional()
        .map_err(Error::Sqlite)?
        .is_some();

    Ok(exists)
}

/// Get the next available rowid
pub fn next_rowid(db: &Connection, schema: &str, table_name: &str) -> Result<i64> {
    let data_table = format!("{}_data", table_name);
    let query = format!(
        "SELECT COALESCE(MAX(rowid), 0) + 1 FROM \"{}\".\"{}\"",
        schema, data_table
    );

    let rowid = db
        .query_row(&query, [], |row| row.get::<_, i64>(0))
        .map_err(Error::Sqlite)?;

    Ok(rowid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    #[test]
    fn test_create_shadow_tables_basic() {
        let db = Connection::open_in_memory().unwrap();

        let config = ShadowTablesConfig {
            vector_columns: vec![VectorColumnDef {
                name: "embedding".to_string(),
                dimensions: 768,
                element_size: 4,
            }],
            data_columns: vec![],
        };
        let result = create_shadow_tables(&db, "main", "test_table", &config);
        assert!(
            result.is_ok(),
            "Failed to create shadow tables: {:?}",
            result
        );

        // Verify tables were created
        let tables: Vec<String> = db
            .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();

        assert!(tables.contains(&"test_table_data".to_string()));
        assert!(tables.contains(&"test_table_info".to_string()));
    }

    #[test]
    fn test_create_shadow_tables_with_columns() {
        let db = Connection::open_in_memory().unwrap();

        let config = ShadowTablesConfig {
            vector_columns: vec![
                VectorColumnDef {
                    name: "embedding".to_string(),
                    dimensions: 768,
                    element_size: 4,
                },
                VectorColumnDef {
                    name: "embedding2".to_string(),
                    dimensions: 384,
                    element_size: 4,
                },
            ],
            data_columns: vec![
                DataColumnDef {
                    name: "id".to_string(),
                    col_type: "TEXT".to_string(),
                },
                DataColumnDef {
                    name: "score".to_string(),
                    col_type: "REAL".to_string(),
                },
            ],
        };
        let result = create_shadow_tables(&db, "main", "test_table", &config);
        assert!(result.is_ok());

        // Verify _data table has correct columns
        let schema: String = db
            .query_row(
                "SELECT sql FROM sqlite_master WHERE name='test_table_data'",
                [],
                |row| row.get(0),
            )
            .unwrap();

        assert!(schema.contains("vec00 BLOB"));
        assert!(schema.contains("vec01 BLOB"));
        assert!(schema.contains("col00 TEXT"));
        assert!(schema.contains("col01 REAL"));
    }

    #[test]
    fn test_create_hnsw_shadow_tables() {
        let db = Connection::open_in_memory().unwrap();

        let result = create_hnsw_shadow_tables(&db, "test_table", "embedding");
        assert!(result.is_ok());

        // Verify HNSW tables were created
        let tables: Vec<String> = db
            .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'test_table_embedding_hnsw_%' ORDER BY name")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();

        assert!(tables.contains(&"test_table_embedding_hnsw_meta".to_string()));
        assert!(tables.contains(&"test_table_embedding_hnsw_nodes".to_string()));
        assert!(tables.contains(&"test_table_embedding_hnsw_edges".to_string()));
    }

    #[test]
    fn test_is_shadow_table() {
        assert!(is_shadow_table("test_table_data"));
        assert!(is_shadow_table("test_table_info"));
        assert!(is_shadow_table("test_table_embedding_hnsw_meta"));
        assert!(is_shadow_table("test_table_embedding_hnsw_nodes"));

        assert!(!is_shadow_table("test_table"));
        assert!(!is_shadow_table("other_table"));
    }

    #[test]
    fn test_insert_and_read_row() {
        let db = Connection::open_in_memory().unwrap();

        let config = ShadowTablesConfig {
            vector_columns: vec![VectorColumnDef {
                name: "embedding".to_string(),
                dimensions: 4,
                element_size: 4,
            }],
            data_columns: vec![DataColumnDef {
                name: "label".to_string(),
                col_type: "TEXT".to_string(),
            }],
        };
        create_shadow_tables(&db, "main", "test_table", &config).unwrap();

        // Insert a row
        let vector = [1.0f32, 2.0, 3.0, 4.0];
        let vector_bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();
        let columns = vec![rusqlite::types::Value::Text("test_label".to_string())];

        insert_row(&db, "main", "test_table", 1, &[&vector_bytes], &columns).unwrap();

        // Read vector back
        let result = read_vector(&db, "main", "test_table", 0, 1).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), vector_bytes);

        // Read column back
        let col_result = read_column(&db, "main", "test_table", 0, 1).unwrap();
        assert!(col_result.is_some());
        if let rusqlite::types::Value::Text(s) = col_result.unwrap() {
            assert_eq!(s, "test_label");
        } else {
            panic!("Expected Text value");
        }
    }

    #[test]
    fn test_update_row() {
        let db = Connection::open_in_memory().unwrap();

        let config = ShadowTablesConfig {
            vector_columns: vec![VectorColumnDef {
                name: "embedding".to_string(),
                dimensions: 4,
                element_size: 4,
            }],
            data_columns: vec![DataColumnDef {
                name: "label".to_string(),
                col_type: "TEXT".to_string(),
            }],
        };
        create_shadow_tables(&db, "main", "test_table", &config).unwrap();

        // Insert a row
        let vector = [1.0f32, 2.0, 3.0, 4.0];
        let vector_bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();
        let columns = vec![rusqlite::types::Value::Text("test_label".to_string())];

        insert_row(&db, "main", "test_table", 1, &[&vector_bytes], &columns).unwrap();

        // Update the row
        let new_vector = [5.0f32, 6.0, 7.0, 8.0];
        let new_vector_bytes: Vec<u8> = new_vector.iter().flat_map(|f| f.to_le_bytes()).collect();
        let new_columns = vec![Some(rusqlite::types::Value::Text("new_label".to_string()))];

        update_row(
            &db,
            "main",
            "test_table",
            1,
            &[Some(&new_vector_bytes[..])],
            &new_columns,
        )
        .unwrap();

        // Read back and verify
        let result = read_vector(&db, "main", "test_table", 0, 1).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), new_vector_bytes);

        let col_result = read_column(&db, "main", "test_table", 0, 1).unwrap();
        if let Some(rusqlite::types::Value::Text(s)) = col_result {
            assert_eq!(s, "new_label");
        } else {
            panic!("Expected Text value");
        }
    }

    #[test]
    fn test_delete_row() {
        let db = Connection::open_in_memory().unwrap();

        let config = ShadowTablesConfig {
            vector_columns: vec![VectorColumnDef {
                name: "embedding".to_string(),
                dimensions: 4,
                element_size: 4,
            }],
            data_columns: vec![],
        };
        create_shadow_tables(&db, "main", "test_table", &config).unwrap();

        // Insert a row
        let vector = [1.0f32, 2.0, 3.0, 4.0];
        let vector_bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();

        insert_row(&db, "main", "test_table", 1, &[&vector_bytes], &[]).unwrap();

        // Verify it exists
        assert!(row_exists(&db, "main", "test_table", 1).unwrap());

        // Delete it
        delete_row(&db, "main", "test_table", 1).unwrap();

        // Verify it's gone
        assert!(!row_exists(&db, "main", "test_table", 1).unwrap());
    }

    #[test]
    fn test_get_all_rowids() {
        let db = Connection::open_in_memory().unwrap();

        let config = ShadowTablesConfig {
            vector_columns: vec![VectorColumnDef {
                name: "embedding".to_string(),
                dimensions: 4,
                element_size: 4,
            }],
            data_columns: vec![],
        };
        create_shadow_tables(&db, "main", "test_table", &config).unwrap();

        // Insert multiple rows
        let vector = [1.0f32, 2.0, 3.0, 4.0];
        let vector_bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();

        insert_row(&db, "main", "test_table", 1, &[&vector_bytes], &[]).unwrap();
        insert_row(&db, "main", "test_table", 5, &[&vector_bytes], &[]).unwrap();
        insert_row(&db, "main", "test_table", 10, &[&vector_bytes], &[]).unwrap();

        // Get all rowids
        let rowids = get_all_rowids(&db, "main", "test_table").unwrap();
        assert_eq!(rowids, vec![1, 5, 10]);
    }

    #[test]
    fn test_next_rowid() {
        let db = Connection::open_in_memory().unwrap();

        let config = ShadowTablesConfig {
            vector_columns: vec![VectorColumnDef {
                name: "embedding".to_string(),
                dimensions: 4,
                element_size: 4,
            }],
            data_columns: vec![],
        };
        create_shadow_tables(&db, "main", "test_table", &config).unwrap();

        // Empty table should return 1
        assert_eq!(next_rowid(&db, "main", "test_table").unwrap(), 1);

        // Insert a row
        let vector = [1.0f32, 2.0, 3.0, 4.0];
        let vector_bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();

        insert_row(&db, "main", "test_table", 1, &[&vector_bytes], &[]).unwrap();

        // Should return 2
        assert_eq!(next_rowid(&db, "main", "test_table").unwrap(), 2);

        // Insert at rowid 10
        insert_row(&db, "main", "test_table", 10, &[&vector_bytes], &[]).unwrap();

        // Should return 11
        assert_eq!(next_rowid(&db, "main", "test_table").unwrap(), 11);
    }
}
