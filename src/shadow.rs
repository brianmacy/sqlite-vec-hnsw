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
use rusqlite::{Connection, Statement, ffi};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

/// Default chunk size (number of vectors per chunk)
pub const DEFAULT_CHUNK_SIZE: usize = 256;

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
}
