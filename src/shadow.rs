//! Shadow table management for vec0 virtual tables
//!
//! The vec0 virtual table uses shadow tables for persistent storage:
//! - {table}_chunks: Chunk metadata (chunk_id, size, validity bitmap, rowids)
//! - {table}_rowids: Rowid to chunk mapping
//! - {table}_vector_chunks{NN}: Vector data storage per column
//! - {table}_auxiliary: Non-indexed column data
//! - {table}_metadatachunks{NN}: Binary metadata per column
//! - {table}_metadatatext{NN}: Text metadata per column
//! - {table}_{column}_hnsw_meta: HNSW index metadata
//! - {table}_{column}_hnsw_nodes: HNSW graph nodes
//! - {table}_{column}_hnsw_edges: HNSW graph edges
//! - {table}_{column}_hnsw_levels: HNSW level index

use crate::error::{Error, Result};
use rusqlite::{Connection, OptionalExtension, Statement, ffi};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

/// Default chunk size (number of vectors per chunk)
pub const DEFAULT_CHUNK_SIZE: usize = 256;

/// Chunk information returned when allocating space for a new vector
#[derive(Debug, Clone)]
pub struct ChunkAllocation {
    pub chunk_id: i64,
    pub chunk_offset: i64,
    pub chunk_size: i64,
}

/// Validity bitmap operations for tracking which vectors are valid in a chunk
pub struct ValidityBitmap {
    data: Vec<u8>,
}

impl ValidityBitmap {
    /// Create a new validity bitmap for the given chunk size
    pub fn new(chunk_size: usize) -> Self {
        let byte_size = chunk_size.div_ceil(8);
        ValidityBitmap {
            data: vec![0u8; byte_size],
        }
    }

    /// Create from existing data
    pub fn from_bytes(data: Vec<u8>) -> Self {
        ValidityBitmap { data }
    }

    /// Check if a bit is set
    pub fn is_set(&self, offset: usize) -> bool {
        let byte_idx = offset / 8;
        let bit_idx = offset % 8;
        if byte_idx >= self.data.len() {
            return false;
        }
        (self.data[byte_idx] & (1 << bit_idx)) != 0
    }

    /// Set a bit
    pub fn set(&mut self, offset: usize) {
        let byte_idx = offset / 8;
        let bit_idx = offset % 8;
        if byte_idx < self.data.len() {
            self.data[byte_idx] |= 1 << bit_idx;
        }
    }

    /// Clear a bit
    pub fn clear(&mut self, offset: usize) {
        let byte_idx = offset / 8;
        let bit_idx = offset % 8;
        if byte_idx < self.data.len() {
            self.data[byte_idx] &= !(1 << bit_idx);
        }
    }

    /// Get the raw bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Find the first unset bit (returns None if all bits are set)
    pub fn find_first_unset(&self, max_offset: usize) -> Option<usize> {
        (0..max_offset).find(|&offset| !self.is_set(offset))
    }
}

/// Configuration for shadow table creation
pub struct ShadowTablesConfig {
    pub num_vector_columns: usize,
    pub num_auxiliary_columns: usize,
    pub num_metadata_columns: usize,
    pub has_text_pk: bool,
    pub num_partition_columns: usize,
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
    // Create _chunks shadow table
    let chunks_sql = if config.num_partition_columns > 0 {
        let mut sql = format!(
            "CREATE TABLE \"{}\".\"{}_chunks\" (\
             chunk_id INTEGER PRIMARY KEY AUTOINCREMENT, \
             size INTEGER NOT NULL, \
             sequence_id INTEGER",
            schema, table_name
        );
        for i in 0..config.num_partition_columns {
            sql.push_str(&format!(", partition{:02}", i));
        }
        sql.push_str(", validity BLOB NOT NULL, rowids BLOB NOT NULL);");
        sql
    } else {
        format!(
            "CREATE TABLE \"{}\".\"{}_chunks\" (\
             chunk_id INTEGER PRIMARY KEY AUTOINCREMENT, \
             size INTEGER NOT NULL, \
             validity BLOB NOT NULL, \
             rowids BLOB NOT NULL\
             );",
            schema, table_name
        )
    };

    db.execute(&chunks_sql, []).map_err(Error::Sqlite)?;

    // Create _rowids shadow table
    let rowids_sql = if config.has_text_pk {
        format!(
            "CREATE TABLE \"{}\".\"{}_rowids\" (\
             rowid INTEGER PRIMARY KEY AUTOINCREMENT, \
             id TEXT UNIQUE NOT NULL, \
             chunk_id INTEGER, \
             chunk_offset INTEGER\
             );",
            schema, table_name
        )
    } else {
        format!(
            "CREATE TABLE \"{}\".\"{}_rowids\" (\
             rowid INTEGER PRIMARY KEY AUTOINCREMENT, \
             id, \
             chunk_id INTEGER, \
             chunk_offset INTEGER\
             );",
            schema, table_name
        )
    };

    db.execute(&rowids_sql, []).map_err(Error::Sqlite)?;

    // Create vector_chunks shadow tables (one per vector column)
    for i in 0..config.num_vector_columns {
        let vector_sql = format!(
            "CREATE TABLE \"{}\".\"{}_vector_chunks{:02}\" (\
             rowid PRIMARY KEY, \
             vectors BLOB NOT NULL\
             );",
            schema, table_name, i
        );

        db.execute(&vector_sql, []).map_err(Error::Sqlite)?;
    }

    // Create metadata shadow tables
    for i in 0..config.num_metadata_columns {
        let metadata_sql = format!(
            "CREATE TABLE \"{}\".\"{}_metadatachunks{:02}\" (\
             rowid PRIMARY KEY, \
             data BLOB NOT NULL\
             );",
            schema, table_name, i
        );

        db.execute(&metadata_sql, []).map_err(Error::Sqlite)?;

        // Also create text metadata table (for TEXT type metadata)
        let metadata_text_sql = format!(
            "CREATE TABLE \"{}\".\"{}_metadatatext{:02}\" (\
             rowid PRIMARY KEY, \
             data TEXT\
             );",
            schema, table_name, i
        );

        db.execute(&metadata_text_sql, []).map_err(Error::Sqlite)?;
    }

    // Create auxiliary columns shadow table
    if config.num_auxiliary_columns > 0 {
        let mut auxiliary_sql = format!(
            "CREATE TABLE \"{}\".\"{}_auxiliary\" (rowid INTEGER PRIMARY KEY",
            schema, table_name
        );

        for i in 0..config.num_auxiliary_columns {
            auxiliary_sql.push_str(&format!(", value{:02}", i));
        }
        auxiliary_sql.push_str(");");

        db.execute(&auxiliary_sql, []).map_err(Error::Sqlite)?;
    }

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
    // Create metadata table
    let meta_sql = format!(
        "CREATE TABLE IF NOT EXISTS \"{}_{}_hnsw_meta\" (\
         key TEXT PRIMARY KEY, \
         value TEXT\
         )",
        table_name, column_name
    );
    db.execute(&meta_sql, []).map_err(Error::Sqlite)?;

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

    // Create edges table
    let edges_sql = format!(
        "CREATE TABLE IF NOT EXISTS \"{}_{}_hnsw_edges\" (\
         from_rowid INTEGER NOT NULL, \
         to_rowid INTEGER NOT NULL, \
         level INTEGER NOT NULL, \
         distance REAL, \
         PRIMARY KEY (from_rowid, to_rowid, level)\
         )",
        table_name, column_name
    );
    db.execute(&edges_sql, []).map_err(Error::Sqlite)?;

    // Create indexes for efficient traversal
    let index_from_sql = format!(
        "CREATE INDEX IF NOT EXISTS \"{}_{}_hnsw_edges_from_level\" \
         ON \"{}_{}_hnsw_edges\"(from_rowid, level)",
        table_name, column_name, table_name, column_name
    );
    db.execute(&index_from_sql, []).map_err(Error::Sqlite)?;

    let index_to_sql = format!(
        "CREATE INDEX IF NOT EXISTS \"{}_{}_hnsw_edges_to_level\" \
         ON \"{}_{}_hnsw_edges\"(to_rowid, level)",
        table_name, column_name, table_name, column_name
    );
    db.execute(&index_to_sql, []).map_err(Error::Sqlite)?;

    // Create levels table
    let levels_sql = format!(
        "CREATE TABLE IF NOT EXISTS \"{}_{}_hnsw_levels\" (\
         level INTEGER NOT NULL, \
         rowid INTEGER NOT NULL, \
         PRIMARY KEY (level, rowid)\
         )",
        table_name, column_name
    );
    db.execute(&levels_sql, []).map_err(Error::Sqlite)?;

    let index_level_sql = format!(
        "CREATE INDEX IF NOT EXISTS \"{}_{}_hnsw_levels_level\" \
         ON \"{}_{}_hnsw_levels\"(level)",
        table_name, column_name, table_name, column_name
    );
    db.execute(&index_level_sql, []).map_err(Error::Sqlite)?;

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
    // Create _chunks shadow table
    let chunks_sql = if config.num_partition_columns > 0 {
        let mut sql = format!(
            "CREATE TABLE \"{}\".\"{}_chunks\" (\
             chunk_id INTEGER PRIMARY KEY AUTOINCREMENT, \
             size INTEGER NOT NULL, \
             sequence_id INTEGER",
            schema, table_name
        );
        for i in 0..config.num_partition_columns {
            sql.push_str(&format!(", partition{:02}", i));
        }
        sql.push_str(", validity BLOB NOT NULL, rowids BLOB NOT NULL);");
        sql
    } else {
        format!(
            "CREATE TABLE \"{}\".\"{}_chunks\" (\
             chunk_id INTEGER PRIMARY KEY AUTOINCREMENT, \
             size INTEGER NOT NULL, \
             validity BLOB NOT NULL, \
             rowids BLOB NOT NULL\
             );",
            schema, table_name
        )
    };

    // SAFETY: execute_sql_ffi is called with a valid database handle
    unsafe { execute_sql_ffi(db, &chunks_sql)? };

    // Create _rowids shadow table
    let rowids_sql = if config.has_text_pk {
        format!(
            "CREATE TABLE \"{}\".\"{}_rowids\" (\
             rowid INTEGER PRIMARY KEY AUTOINCREMENT, \
             id TEXT UNIQUE NOT NULL, \
             chunk_id INTEGER, \
             chunk_offset INTEGER\
             );",
            schema, table_name
        )
    } else {
        format!(
            "CREATE TABLE \"{}\".\"{}_rowids\" (\
             rowid INTEGER PRIMARY KEY AUTOINCREMENT, \
             id, \
             chunk_id INTEGER, \
             chunk_offset INTEGER\
             );",
            schema, table_name
        )
    };

    // SAFETY: execute_sql_ffi is called with a valid database handle
    unsafe { execute_sql_ffi(db, &rowids_sql)? };

    // Create vector_chunks shadow tables
    for i in 0..config.num_vector_columns {
        let vector_sql = format!(
            "CREATE TABLE \"{}\".\"{}_vector_chunks{:02}\" (\
             rowid PRIMARY KEY, \
             vectors BLOB NOT NULL\
             );",
            schema, table_name, i
        );
        // SAFETY: execute_sql_ffi is called with a valid database handle
        unsafe { execute_sql_ffi(db, &vector_sql)? };
    }

    // Create metadata shadow tables
    for i in 0..config.num_metadata_columns {
        let metadata_sql = format!(
            "CREATE TABLE \"{}\".\"{}_metadatachunks{:02}\" (\
             rowid PRIMARY KEY, \
             data BLOB NOT NULL\
             );",
            schema, table_name, i
        );
        // SAFETY: execute_sql_ffi is called with a valid database handle
        unsafe { execute_sql_ffi(db, &metadata_sql)? };

        let metadata_text_sql = format!(
            "CREATE TABLE \"{}\".\"{}_metadatatext{:02}\" (\
             rowid PRIMARY KEY, \
             data TEXT\
             );",
            schema, table_name, i
        );
        // SAFETY: execute_sql_ffi is called with a valid database handle
        unsafe { execute_sql_ffi(db, &metadata_text_sql)? };
    }

    // Create auxiliary columns shadow table
    if config.num_auxiliary_columns > 0 {
        let mut auxiliary_sql = format!(
            "CREATE TABLE \"{}\".\"{}_auxiliary\" (rowid INTEGER PRIMARY KEY",
            schema, table_name
        );

        for i in 0..config.num_auxiliary_columns {
            auxiliary_sql.push_str(&format!(", value{:02}", i));
        }
        auxiliary_sql.push_str(");");

        // SAFETY: execute_sql_ffi is called with a valid database handle
        unsafe { execute_sql_ffi(db, &auxiliary_sql)? };
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
    // Create metadata table
    let meta_sql = format!(
        "CREATE TABLE IF NOT EXISTS \"{}_{}_hnsw_meta\" (\
         key TEXT PRIMARY KEY, \
         value TEXT\
         )",
        table_name, column_name
    );
    // SAFETY: execute_sql_ffi is called with a valid database handle
    unsafe { execute_sql_ffi(db, &meta_sql)? };

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

    // Create edges table
    let edges_sql = format!(
        "CREATE TABLE IF NOT EXISTS \"{}_{}_hnsw_edges\" (\
         from_rowid INTEGER NOT NULL, \
         to_rowid INTEGER NOT NULL, \
         level INTEGER NOT NULL, \
         distance REAL, \
         PRIMARY KEY (from_rowid, to_rowid, level)\
         )",
        table_name, column_name
    );
    // SAFETY: execute_sql_ffi is called with a valid database handle
    unsafe { execute_sql_ffi(db, &edges_sql)? };

    // Create indexes
    let index_from_sql = format!(
        "CREATE INDEX IF NOT EXISTS \"{}_{}_hnsw_edges_from_level\" \
         ON \"{}_{}_hnsw_edges\"(from_rowid, level)",
        table_name, column_name, table_name, column_name
    );
    // SAFETY: execute_sql_ffi is called with a valid database handle
    unsafe { execute_sql_ffi(db, &index_from_sql)? };

    let index_to_sql = format!(
        "CREATE INDEX IF NOT EXISTS \"{}_{}_hnsw_edges_to_level\" \
         ON \"{}_{}_hnsw_edges\"(to_rowid, level)",
        table_name, column_name, table_name, column_name
    );
    // SAFETY: execute_sql_ffi is called with a valid database handle
    unsafe { execute_sql_ffi(db, &index_to_sql)? };

    // Create levels table
    let levels_sql = format!(
        "CREATE TABLE IF NOT EXISTS \"{}_{}_hnsw_levels\" (\
         level INTEGER NOT NULL, \
         rowid INTEGER NOT NULL, \
         PRIMARY KEY (level, rowid)\
         )",
        table_name, column_name
    );
    // SAFETY: execute_sql_ffi is called with a valid database handle
    unsafe { execute_sql_ffi(db, &levels_sql)? };

    let index_level_sql = format!(
        "CREATE INDEX IF NOT EXISTS \"{}_{}_hnsw_levels_level\" \
         ON \"{}_{}_hnsw_levels\"(level)",
        table_name, column_name, table_name, column_name
    );
    // SAFETY: execute_sql_ffi is called with a valid database handle
    unsafe { execute_sql_ffi(db, &index_level_sql)? };

    Ok(())
}

/// Check if a table name is a shadow table for vec0
pub fn is_shadow_table(name: &str) -> bool {
    const SHADOWS: &[&str] = &[
        "_rowids",
        "_chunks",
        "_auxiliary",
        "_info",
        "_metadatachunks00",
        "_metadatachunks01",
        "_metadatachunks02",
        "_metadatachunks03",
        "_metadatachunks04",
        "_metadatachunks05",
        "_metadatachunks06",
        "_metadatachunks07",
        "_metadatachunks08",
        "_metadatachunks09",
        "_metadatachunks10",
        "_metadatachunks11",
        "_metadatachunks12",
        "_metadatachunks13",
        "_metadatachunks14",
        "_metadatachunks15",
        "_metadatatext00",
        "_metadatatext01",
        "_metadatatext02",
        "_metadatatext03",
        "_metadatatext04",
        "_metadatatext05",
        "_metadatatext06",
        "_metadatatext07",
        "_metadatatext08",
        "_metadatatext09",
        "_metadatatext10",
        "_metadatatext11",
        "_metadatatext12",
        "_metadatatext13",
        "_metadatatext14",
        "_metadatatext15",
        "_vector_chunks00",
        "_vector_chunks01",
        "_vector_chunks02",
        "_vector_chunks03",
        "_hnsw_meta",
        "_hnsw_nodes",
        "_hnsw_edges",
        "_hnsw_levels",
    ];

    SHADOWS.iter().any(|&suffix| name.ends_with(suffix))
}

/// Statement cache for shadow table operations
pub struct StatementCache<'conn> {
    // Read operations
    pub get_chunk_position: Option<Statement<'conn>>,
    pub get_vector_blob: Option<Statement<'conn>>,
    pub get_rowid_mapping: Option<Statement<'conn>>,

    // Write operations
    pub insert_chunk: Option<Statement<'conn>>,
    pub insert_rowid: Option<Statement<'conn>>,
    pub update_chunk_size: Option<Statement<'conn>>,
    pub update_rowid_position: Option<Statement<'conn>>,

    // Chunk management
    pub latest_chunk: Option<Statement<'conn>>,
    pub chunk_has_space: Option<Statement<'conn>>,
}

impl<'conn> StatementCache<'conn> {
    /// Create a new empty statement cache
    pub fn new() -> Self {
        StatementCache {
            get_chunk_position: None,
            get_vector_blob: None,
            get_rowid_mapping: None,
            insert_chunk: None,
            insert_rowid: None,
            update_chunk_size: None,
            update_rowid_position: None,
            latest_chunk: None,
            chunk_has_space: None,
        }
    }

    /// Clear all cached statements
    pub fn clear(&mut self) {
        self.get_chunk_position = None;
        self.get_vector_blob = None;
        self.get_rowid_mapping = None;
        self.insert_chunk = None;
        self.insert_rowid = None;
        self.update_chunk_size = None;
        self.update_rowid_position = None;
        self.latest_chunk = None;
        self.chunk_has_space = None;
    }
}

impl<'conn> Default for StatementCache<'conn> {
    fn default() -> Self {
        Self::new()
    }
}

/// Find or create a chunk with available space for inserting a vector
///
/// This implements the chunk allocation strategy from the C version
pub fn find_or_create_chunk(
    db: &Connection,
    schema: &str,
    table_name: &str,
    chunk_size: usize,
) -> Result<ChunkAllocation> {
    // Try to find an existing chunk with space
    let query = format!(
        "SELECT chunk_id, size FROM \"{}\".\"{}_chunks\" \
         WHERE size < ? ORDER BY chunk_id DESC LIMIT 1",
        schema, table_name
    );

    let mut stmt = db.prepare(&query).map_err(Error::Sqlite)?;
    let result = stmt
        .query_row([chunk_size as i64], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
        })
        .optional()
        .map_err(Error::Sqlite)?;

    if let Some((chunk_id, size)) = result {
        // Found a chunk with space
        return Ok(ChunkAllocation {
            chunk_id,
            chunk_offset: size,
            chunk_size: size,
        });
    }

    // No chunk with space, create a new one
    let validity = ValidityBitmap::new(chunk_size);
    let rowids_blob = vec![0u8; chunk_size * 8]; // 8 bytes per rowid

    let insert_sql = format!(
        "INSERT INTO \"{}\".\"{}_chunks\" (size, validity, rowids) VALUES (?, ?, ?)",
        schema, table_name
    );

    db.execute(
        &insert_sql,
        rusqlite::params![0i64, validity.as_bytes(), rowids_blob],
    )
    .map_err(Error::Sqlite)?;

    let chunk_id = db.last_insert_rowid();

    Ok(ChunkAllocation {
        chunk_id,
        chunk_offset: 0,
        chunk_size: 0,
    })
}

/// Write a vector to a vector_chunks table using BLOB operations
pub fn write_vector_to_chunk(
    db: &Connection,
    _schema: &str,
    table_name: &str,
    column_idx: usize,
    chunk_id: i64,
    chunk_offset: i64,
    vector_data: &[u8],
) -> Result<()> {
    let table = format!("{}_vector_chunks{:02}", table_name, column_idx);

    // Calculate byte offset within the chunk
    let byte_offset = (chunk_offset * vector_data.len() as i64) as usize;

    // Open or create the BLOB
    // First try to open, if it doesn't exist, insert a row
    let vectors_size = DEFAULT_CHUNK_SIZE * vector_data.len();
    let check_sql = format!("SELECT 1 FROM \"{}\" WHERE rowid = ?", table);

    let exists = db
        .query_row(&check_sql, [chunk_id], |_| Ok(()))
        .optional()
        .map_err(Error::Sqlite)?
        .is_some();

    if !exists {
        // Create the blob row
        let insert_sql = format!(
            "INSERT INTO \"{}\" (rowid, vectors) VALUES (?, zeroblob(?))",
            table
        );
        db.execute(
            &insert_sql,
            rusqlite::params![chunk_id, vectors_size as i64],
        )
        .map_err(Error::Sqlite)?;
    }

    // Now open the BLOB for writing
    let mut blob = db
        .blob_open(
            rusqlite::DatabaseName::Main,
            &table,
            "vectors",
            chunk_id,
            false, // read-write
        )
        .map_err(Error::Sqlite)?;

    // Write the vector data at the correct offset
    use std::io::{Seek, SeekFrom, Write};
    blob.seek(SeekFrom::Start(byte_offset as u64))
        .map_err(|e| Error::InvalidParameter(format!("Seek failed: {}", e)))?;
    blob.write_all(vector_data)
        .map_err(|e| Error::InvalidParameter(format!("Write failed: {}", e)))?;
    blob.close().map_err(Error::Sqlite)?;

    Ok(())
}

/// Update the validity bitmap and chunk size after inserting a vector
pub fn update_chunk_after_insert(
    db: &Connection,
    schema: &str,
    table_name: &str,
    chunk_id: i64,
    chunk_offset: i64,
    chunk_size: usize,
) -> Result<()> {
    // Read current validity bitmap
    let query = format!(
        "SELECT validity FROM \"{}\".\"{}_chunks\" WHERE chunk_id = ?",
        schema, table_name
    );
    let validity_data: Vec<u8> = db
        .query_row(&query, [chunk_id], |row| row.get(0))
        .map_err(Error::Sqlite)?;

    let mut validity = ValidityBitmap::from_bytes(validity_data);
    validity.set(chunk_offset as usize);

    // Update the chunk
    let new_size = (chunk_offset + 1).max(chunk_size as i64);
    let update_sql = format!(
        "UPDATE \"{}\".\"{}_chunks\" SET size = ?, validity = ? WHERE chunk_id = ?",
        schema, table_name
    );

    db.execute(
        &update_sql,
        rusqlite::params![new_size, validity.as_bytes(), chunk_id],
    )
    .map_err(Error::Sqlite)?;

    Ok(())
}

/// Insert or update rowid mapping
pub fn insert_rowid_mapping(
    db: &Connection,
    schema: &str,
    table_name: &str,
    rowid: i64,
    chunk_id: i64,
    chunk_offset: i64,
) -> Result<()> {
    let insert_sql = format!(
        "INSERT OR REPLACE INTO \"{}\".\"{}_rowids\" (rowid, id, chunk_id, chunk_offset) \
         VALUES (?, ?, ?, ?)",
        schema, table_name
    );

    db.execute(
        &insert_sql,
        rusqlite::params![rowid, rusqlite::types::Null, chunk_id, chunk_offset],
    )
    .map_err(Error::Sqlite)?;

    Ok(())
}

/// Perform a complete vector insert operation using raw FFI
///
/// # Safety
/// Must be called with a valid sqlite3 database handle
pub unsafe fn insert_vector_ffi(
    db: *mut ffi::sqlite3,
    schema: &str,
    table_name: &str,
    chunk_size: usize,
    rowid: i64,
    column_idx: usize,
    vector_data: &[u8],
) -> Result<()> {
    // Wrap the raw handle in a Connection for our helper functions
    // SAFETY: We're creating a non-owning Connection from the handle
    // This is safe because we don't drop the Connection (we forget it)
    let conn = unsafe { Connection::from_handle(db).map_err(Error::Sqlite)? };

    // Step 1: Find or create a chunk with space
    let allocation = find_or_create_chunk(&conn, schema, table_name, chunk_size)?;

    // Step 2: Write the vector data to the chunk
    write_vector_to_chunk(
        &conn,
        schema,
        table_name,
        column_idx,
        allocation.chunk_id,
        allocation.chunk_offset,
        vector_data,
    )?;

    // Step 3: Update the chunk metadata
    update_chunk_after_insert(
        &conn,
        schema,
        table_name,
        allocation.chunk_id,
        allocation.chunk_offset,
        allocation.chunk_size as usize,
    )?;

    // Step 4: Insert rowid mapping
    insert_rowid_mapping(
        &conn,
        schema,
        table_name,
        rowid,
        allocation.chunk_id,
        allocation.chunk_offset,
    )?;

    // Forget the Connection to avoid double-free
    // SAFETY: The Connection doesn't own the handle
    std::mem::forget(conn);

    Ok(())
}

/// Read a vector from shadow tables
///
/// # Arguments
/// * `db` - Database connection
/// * `schema` - Schema name
/// * `table_name` - Table name
/// * `column_idx` - Vector column index
/// * `rowid` - Rowid of the vector to read
///
/// # Returns
/// Vector data as bytes, or None if not found
pub fn read_vector_from_chunk(
    db: &Connection,
    schema: &str,
    table_name: &str,
    column_idx: usize,
    rowid: i64,
) -> Result<Option<Vec<u8>>> {
    // Step 1: Get chunk_id and chunk_offset from _rowids table
    let rowid_query = format!(
        "SELECT chunk_id, chunk_offset FROM \"{}\".\"{}_rowids\" WHERE rowid = ?",
        schema, table_name
    );

    let mapping = db
        .query_row(&rowid_query, [rowid], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
        })
        .optional()
        .map_err(Error::Sqlite)?;

    let (chunk_id, chunk_offset) = match mapping {
        Some(m) => m,
        None => return Ok(None), // Rowid not found
    };

    // Step 2: Check validity bitmap
    let validity_query = format!(
        "SELECT validity FROM \"{}\".\"{}_chunks\" WHERE chunk_id = ?",
        schema, table_name
    );

    let validity_data: Vec<u8> = db
        .query_row(&validity_query, [chunk_id], |row| row.get(0))
        .map_err(Error::Sqlite)?;

    let validity = ValidityBitmap::from_bytes(validity_data);
    if !validity.is_set(chunk_offset as usize) {
        return Ok(None); // Vector was deleted
    }

    // Step 3: Open BLOB and read vector data
    let table = format!("{}_vector_chunks{:02}", table_name, column_idx);

    let mut blob = db
        .blob_open(
            rusqlite::DatabaseName::Main,
            &table,
            "vectors",
            chunk_id,
            true, // read-only
        )
        .map_err(Error::Sqlite)?;

    // Determine vector size from BLOB
    use std::io::{Read, Seek, SeekFrom};
    let blob_size = blob
        .seek(SeekFrom::End(0))
        .map_err(|e| Error::InvalidParameter(format!("Seek failed: {}", e)))?;

    // Calculate vector size (blob_size / DEFAULT_CHUNK_SIZE)
    let vector_size = (blob_size as usize) / DEFAULT_CHUNK_SIZE;
    let byte_offset = (chunk_offset as usize) * vector_size;

    // Seek to the vector position and read
    blob.seek(SeekFrom::Start(byte_offset as u64))
        .map_err(|e| Error::InvalidParameter(format!("Seek failed: {}", e)))?;

    let mut vector_data = vec![0u8; vector_size];
    blob.read_exact(&mut vector_data)
        .map_err(|e| Error::InvalidParameter(format!("Read failed: {}", e)))?;

    blob.close().map_err(Error::Sqlite)?;

    Ok(Some(vector_data))
}

/// Mark a chunk row as invalid (for DELETE operations)
///
/// # Arguments
/// * `db` - Database connection
/// * `chunks_table` - Name of the chunks table
/// * `chunk_id` - Chunk ID
/// * `chunk_offset` - Offset within the chunk
pub fn mark_chunk_row_invalid(
    db: &Connection,
    chunks_table: &str,
    chunk_id: i64,
    chunk_offset: usize,
) -> Result<()> {
    // Read current validity bitmap
    let query = format!(
        "SELECT validity FROM \"{}\" WHERE chunk_id = ?",
        chunks_table
    );

    let validity_data: Vec<u8> = db
        .query_row(&query, [chunk_id], |row| row.get(0))
        .map_err(Error::Sqlite)?;

    // Clear the bit for this offset
    let mut validity = ValidityBitmap::from_bytes(validity_data);
    validity.clear(chunk_offset);

    // Write back to database
    let update_sql = format!(
        "UPDATE \"{}\" SET validity = ? WHERE chunk_id = ?",
        chunks_table
    );
    db.execute(
        &update_sql,
        rusqlite::params![validity.as_bytes(), chunk_id],
    )
    .map_err(Error::Sqlite)?;

    Ok(())
}

/// Get all rowids from the shadow table (for full scan)
pub fn get_all_rowids(db: &Connection, schema: &str, table_name: &str) -> Result<Vec<i64>> {
    let query = format!(
        "SELECT rowid FROM \"{}\".\"{}_rowids\" ORDER BY rowid",
        schema, table_name
    );

    let mut stmt = db.prepare(&query).map_err(Error::Sqlite)?;
    let rowids = stmt
        .query_map([], |row| row.get(0))
        .map_err(Error::Sqlite)?
        .collect::<std::result::Result<Vec<i64>, _>>()
        .map_err(Error::Sqlite)?;

    Ok(rowids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    #[test]
    fn test_create_shadow_tables_basic() {
        let db = Connection::open_in_memory().unwrap();

        let config = ShadowTablesConfig {
            num_vector_columns: 1,
            num_auxiliary_columns: 0,
            num_metadata_columns: 0,
            has_text_pk: false,
            num_partition_columns: 0,
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

        assert!(tables.contains(&"test_table_chunks".to_string()));
        assert!(tables.contains(&"test_table_rowids".to_string()));
        assert!(tables.contains(&"test_table_vector_chunks00".to_string()));
    }

    #[test]
    fn test_create_shadow_tables_with_text_pk() {
        let db = Connection::open_in_memory().unwrap();

        let config = ShadowTablesConfig {
            num_vector_columns: 1,
            num_auxiliary_columns: 0,
            num_metadata_columns: 0,
            has_text_pk: true,
            num_partition_columns: 0,
        };
        let result = create_shadow_tables(&db, "main", "test_table", &config);
        assert!(result.is_ok());

        // Verify rowids table has TEXT UNIQUE constraint on id
        let schema: String = db
            .query_row(
                "SELECT sql FROM sqlite_master WHERE name='test_table_rowids'",
                [],
                |row| row.get(0),
            )
            .unwrap();

        assert!(schema.contains("TEXT UNIQUE NOT NULL"));
    }

    #[test]
    fn test_create_shadow_tables_multiple_vectors() {
        let db = Connection::open_in_memory().unwrap();

        let config = ShadowTablesConfig {
            num_vector_columns: 3,
            num_auxiliary_columns: 0,
            num_metadata_columns: 0,
            has_text_pk: false,
            num_partition_columns: 0,
        };
        let result = create_shadow_tables(&db, "main", "test_table", &config);
        assert!(result.is_ok());

        // Verify all vector chunk tables were created
        let tables: Vec<String> = db
            .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'test_table_vector_%' ORDER BY name")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(tables.len(), 3);
        assert!(tables.contains(&"test_table_vector_chunks00".to_string()));
        assert!(tables.contains(&"test_table_vector_chunks01".to_string()));
        assert!(tables.contains(&"test_table_vector_chunks02".to_string()));
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
        assert!(tables.contains(&"test_table_embedding_hnsw_levels".to_string()));
    }

    #[test]
    fn test_is_shadow_table() {
        assert!(is_shadow_table("test_table_rowids"));
        assert!(is_shadow_table("test_table_chunks"));
        assert!(is_shadow_table("test_table_vector_chunks00"));
        assert!(is_shadow_table("test_table_metadatachunks00"));
        assert!(is_shadow_table("test_table_metadatatext00"));
        assert!(is_shadow_table("test_table_auxiliary"));
        assert!(is_shadow_table("test_table_embedding_hnsw_meta"));
        assert!(is_shadow_table("test_table_embedding_hnsw_nodes"));

        assert!(!is_shadow_table("test_table"));
        assert!(!is_shadow_table("other_table"));
        assert!(!is_shadow_table("test_table_data"));
    }

    #[test]
    fn test_statement_cache_new() {
        let cache = StatementCache::new();
        assert!(cache.get_chunk_position.is_none());
        assert!(cache.insert_chunk.is_none());
    }

    #[test]
    fn test_statement_cache_clear() {
        let mut cache = StatementCache::new();
        // Cache starts empty, so clear should work without errors
        cache.clear();
        assert!(cache.get_chunk_position.is_none());
    }

    #[test]
    fn test_validity_bitmap_basic() {
        let mut bitmap = ValidityBitmap::new(256);

        // Initially all bits should be unset
        assert!(!bitmap.is_set(0));
        assert!(!bitmap.is_set(100));
        assert!(!bitmap.is_set(255));

        // Set some bits
        bitmap.set(0);
        bitmap.set(100);
        bitmap.set(255);

        assert!(bitmap.is_set(0));
        assert!(bitmap.is_set(100));
        assert!(bitmap.is_set(255));
        assert!(!bitmap.is_set(50));

        // Clear a bit
        bitmap.clear(100);
        assert!(!bitmap.is_set(100));
    }

    #[test]
    fn test_validity_bitmap_find_first_unset() {
        let mut bitmap = ValidityBitmap::new(256);

        // First unset should be 0
        assert_eq!(bitmap.find_first_unset(256), Some(0));

        // Set first 5 bits
        for i in 0..5 {
            bitmap.set(i);
        }

        // First unset should now be 5
        assert_eq!(bitmap.find_first_unset(256), Some(5));

        // Set all bits
        for i in 0..256 {
            bitmap.set(i);
        }

        // No unset bits
        assert_eq!(bitmap.find_first_unset(256), None);
    }

    #[test]
    fn test_find_or_create_chunk() {
        let db = Connection::open_in_memory().unwrap();

        // Create shadow tables
        let config = ShadowTablesConfig {
            num_vector_columns: 1,
            num_auxiliary_columns: 0,
            num_metadata_columns: 0,
            has_text_pk: false,
            num_partition_columns: 0,
        };
        create_shadow_tables(&db, "main", "test_table", &config).unwrap();

        // First allocation should create a new chunk
        let allocation = find_or_create_chunk(&db, "main", "test_table", 256).unwrap();
        assert_eq!(allocation.chunk_offset, 0);
        assert_eq!(allocation.chunk_size, 0);

        // Verify chunk was created
        let count: i64 = db
            .query_row("SELECT COUNT(*) FROM test_table_chunks", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_insert_rowid_mapping() {
        let db = Connection::open_in_memory().unwrap();

        let config = ShadowTablesConfig {
            num_vector_columns: 1,
            num_auxiliary_columns: 0,
            num_metadata_columns: 0,
            has_text_pk: false,
            num_partition_columns: 0,
        };
        create_shadow_tables(&db, "main", "test_table", &config).unwrap();

        // Insert a rowid mapping
        insert_rowid_mapping(&db, "main", "test_table", 1, 100, 5).unwrap();

        // Verify it was inserted
        let (chunk_id, chunk_offset): (i64, i64) = db
            .query_row(
                "SELECT chunk_id, chunk_offset FROM test_table_rowids WHERE rowid = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();

        assert_eq!(chunk_id, 100);
        assert_eq!(chunk_offset, 5);
    }

    #[test]
    fn test_update_chunk_after_insert() {
        let db = Connection::open_in_memory().unwrap();

        let config = ShadowTablesConfig {
            num_vector_columns: 1,
            num_auxiliary_columns: 0,
            num_metadata_columns: 0,
            has_text_pk: false,
            num_partition_columns: 0,
        };
        create_shadow_tables(&db, "main", "test_table", &config).unwrap();

        // Create a chunk
        let allocation = find_or_create_chunk(&db, "main", "test_table", 256).unwrap();

        // Update it after inserting at offset 5
        update_chunk_after_insert(
            &db,
            "main",
            "test_table",
            allocation.chunk_id,
            5,
            allocation.chunk_size as usize,
        )
        .unwrap();

        // Verify the size was updated
        let size: i64 = db
            .query_row(
                "SELECT size FROM test_table_chunks WHERE chunk_id = ?",
                [allocation.chunk_id],
                |row| row.get(0),
            )
            .unwrap();

        assert_eq!(size, 6); // offset 5 means 6 vectors (0-5)

        // Verify validity bitmap was updated
        let validity: Vec<u8> = db
            .query_row(
                "SELECT validity FROM test_table_chunks WHERE chunk_id = ?",
                [allocation.chunk_id],
                |row| row.get(0),
            )
            .unwrap();

        let bitmap = ValidityBitmap::from_bytes(validity);
        assert!(bitmap.is_set(5));
        assert!(!bitmap.is_set(6));
    }
}
