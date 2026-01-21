//! HNSW (Hierarchical Navigable Small World) index implementation

pub mod cache;
pub mod insert;
pub mod rebuild;
pub mod search;
pub mod storage;

use crate::distance::DistanceMetric;
use crate::error::{Error, Result};
use crate::vector::VectorType;
use rusqlite::{Connection, OptionalExtension, Statement};

/// HNSW index parameters
#[derive(Debug, Clone, Copy)]
pub struct HnswParams {
    /// Number of bidirectional links per node (default: 32)
    pub m: i32,
    /// Max connections at layer 0 (default: 64, typically 2*M)
    pub max_m0: i32,
    /// Dynamic candidate list size during construction (default: 400)
    pub ef_construction: i32,
    /// Dynamic candidate list size during search (default: 200)
    pub ef_search: i32,
    /// Maximum hierarchy depth (default: 16)
    pub max_level: i32,
    /// Level generation factor for exponential decay (1/ln(M))
    pub level_factor: f64,
}

impl Default for HnswParams {
    fn default() -> Self {
        HnswParams {
            m: 32,
            max_m0: 64,
            ef_construction: 400,
            ef_search: 200,
            max_level: 16,
            level_factor: 1.0 / (32.0_f64).ln(),
        }
    }
}

impl HnswParams {
    /// Create parameters for high recall (>95%)
    pub fn high_recall() -> Self {
        HnswParams {
            m: 32,
            max_m0: 64,
            ef_construction: 400,
            ef_search: 200,
            ..Default::default()
        }
    }

    /// Create parameters optimized for fast inserts
    pub fn hot_tier() -> Self {
        HnswParams {
            m: 32,
            max_m0: 64,
            ef_construction: 200,
            ef_search: 100,
            ..Default::default()
        }
    }

    /// Create parameters for balanced performance
    pub fn warm_tier() -> Self {
        HnswParams {
            m: 64,
            max_m0: 128,
            ef_construction: 600,
            ef_search: 400,
            ..Default::default()
        }
    }

    /// Create parameters for high quality index
    pub fn cold_tier() -> Self {
        HnswParams {
            m: 96,
            max_m0: 192,
            ef_construction: 1000,
            ef_search: 800,
            ..Default::default()
        }
    }
}

/// HNSW index metadata (page-cache based - only ~64 bytes in memory)
#[derive(Debug, Clone)]
pub struct HnswMetadata {
    /// Index parameters
    pub params: HnswParams,

    /// Entry point node
    pub entry_point_rowid: i64, // -1 if empty
    pub entry_point_level: i32,

    /// Statistics (approximate, not authoritative)
    pub num_nodes: i32,

    /// Vector metadata
    pub dimensions: i32,
    pub element_type: VectorType,
    pub distance_metric: DistanceMetric,

    /// RNG seed for level generation
    pub rng_seed: u32,

    /// Version tracking
    pub hnsw_version: i64,
}

impl HnswMetadata {
    /// Create new metadata with default parameters
    pub fn new(dimensions: i32, element_type: VectorType, distance_metric: DistanceMetric) -> Self {
        use std::collections::hash_map::RandomState;
        use std::hash::BuildHasher;

        // Generate a random seed
        let random_state = RandomState::new();
        let rng_seed = random_state.hash_one(std::time::SystemTime::now()) as u32;

        HnswMetadata {
            params: HnswParams::default(),
            entry_point_rowid: -1,
            entry_point_level: -1,
            num_nodes: 0,
            dimensions,
            element_type,
            distance_metric,
            rng_seed,
            hnsw_version: 1,
        }
    }

    /// Load metadata from shadow table
    pub fn load_from_db(
        db: &Connection,
        table_name: &str,
        column_name: &str,
    ) -> Result<Option<Self>> {
        let meta_table = format!("{}_{}_hnsw_meta", table_name, column_name);

        // Try to read metadata values
        let get_meta = |key: &str| -> Result<Option<String>> {
            let query = format!("SELECT value FROM \"{}\" WHERE key = ?", meta_table);
            match db
                .query_row(&query, [key], |row| row.get::<_, String>(0))
                .optional()
            {
                Ok(opt) => Ok(opt),
                Err(rusqlite::Error::SqliteFailure(err, _))
                    if err.code == rusqlite::ErrorCode::Unknown =>
                {
                    // Table doesn't exist
                    Ok(None)
                }
                Err(e) => Err(Error::Sqlite(e)),
            }
        };

        // If entry_point_rowid doesn't exist, index is not initialized
        let entry_point_rowid = match get_meta("entry_point_rowid")? {
            Some(val) => val.parse::<i64>().unwrap_or(-1),
            None => return Ok(None),
        };

        let params = HnswParams {
            m: get_meta("M")?.and_then(|s| s.parse().ok()).unwrap_or(32),
            max_m0: get_meta("max_M0")?
                .and_then(|s| s.parse().ok())
                .unwrap_or(64),
            ef_construction: get_meta("ef_construction")?
                .and_then(|s| s.parse().ok())
                .unwrap_or(400),
            ef_search: get_meta("ef_search")?
                .and_then(|s| s.parse().ok())
                .unwrap_or(200),
            max_level: get_meta("max_level")?
                .and_then(|s| s.parse().ok())
                .unwrap_or(16),
            level_factor: get_meta("level_factor")?
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.0 / 32.0_f64.ln()),
        };

        Ok(Some(HnswMetadata {
            params,
            entry_point_rowid,
            entry_point_level: get_meta("entry_point_level")?
                .and_then(|s| s.parse().ok())
                .unwrap_or(-1),
            num_nodes: get_meta("num_nodes")?
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            dimensions: get_meta("dimensions")?
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            element_type: get_meta("element_type")?
                .and_then(|s| match s.as_str() {
                    "float32" => Some(VectorType::Float32),
                    "int8" => Some(VectorType::Int8),
                    "bit" => Some(VectorType::Bit),
                    _ => None,
                })
                .unwrap_or(VectorType::Float32),
            distance_metric: get_meta("distance_metric")?
                .and_then(|s| match s.as_str() {
                    "l2" => Some(DistanceMetric::L2),
                    "cosine" => Some(DistanceMetric::Cosine),
                    "l1" => Some(DistanceMetric::L1),
                    _ => None,
                })
                .unwrap_or(DistanceMetric::L2),
            rng_seed: get_meta("rng_seed")?
                .and_then(|s| s.parse().ok())
                .unwrap_or(12345),
            hnsw_version: get_meta("hnsw_version")?
                .and_then(|s| s.parse().ok())
                .unwrap_or(1),
        }))
    }

    /// Save metadata to shadow table (batched multi-row INSERT)
    pub fn save_to_db(&self, db: &Connection, table_name: &str, column_name: &str) -> Result<()> {
        let meta_table = format!("{}_{}_hnsw_meta", table_name, column_name);

        // Batch all metadata in one multi-row INSERT
        let update_sql = format!(
            "INSERT OR REPLACE INTO \"{}\" (key, value) VALUES \
             ('M', ?), ('max_M0', ?), ('ef_construction', ?), ('ef_search', ?), \
             ('max_level', ?), ('level_factor', ?), ('entry_point_rowid', ?), \
             ('entry_point_level', ?), ('num_nodes', ?), ('dimensions', ?), \
             ('element_type', ?), ('distance_metric', ?), ('rng_seed', ?), \
             ('hnsw_version', ?)",
            meta_table
        );

        db.execute(
            &update_sql,
            rusqlite::params![
                self.params.m.to_string(),
                self.params.max_m0.to_string(),
                self.params.ef_construction.to_string(),
                self.params.ef_search.to_string(),
                self.params.max_level.to_string(),
                self.params.level_factor.to_string(),
                self.entry_point_rowid.to_string(),
                self.entry_point_level.to_string(),
                self.num_nodes.to_string(),
                self.dimensions.to_string(),
                self.element_type.as_str(),
                self.distance_metric.as_str(),
                self.rng_seed.to_string(),
                self.hnsw_version.to_string(),
            ],
        )?;

        Ok(())
    }

    /// Save only dynamic fields that change during operations
    /// Uses batched multi-row INSERT for efficiency
    /// Only saves: entry_point, num_nodes, hnsw_version
    pub fn save_dynamic_to_db(
        &self,
        db: &Connection,
        table_name: &str,
        column_name: &str,
    ) -> Result<()> {
        let meta_table = format!("{}_{}_hnsw_meta", table_name, column_name);

        // Batch all dynamic fields in one multi-row INSERT
        let update_sql = format!(
            "INSERT OR REPLACE INTO \"{}\" (key, value) VALUES \
             ('entry_point_rowid', ?), \
             ('entry_point_level', ?), \
             ('num_nodes', ?), \
             ('hnsw_version', ?)",
            meta_table
        );

        db.execute(
            &update_sql,
            rusqlite::params![
                self.entry_point_rowid.to_string(),
                self.entry_point_level.to_string(),
                self.num_nodes.to_string(),
                self.hnsw_version.to_string(),
            ],
        )?;

        Ok(())
    }

    /// Save dynamic metadata using cached prepared statement (FAST PATH)
    /// Matches C implementation: uses single prepared statement for all updates
    ///
    /// # Safety
    /// cached_stmt must be a valid prepared statement for:
    /// INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn save_dynamic_to_db_cached(
        &self,
        cached_stmt: *mut rusqlite::ffi::sqlite3_stmt,
        update_entry_point: bool,
    ) -> Result<()> {
        use rusqlite::ffi;
        use std::ffi::CString;

        // Helper to update one metadata field
        let update_field = |key: &str, value: &str| -> Result<()> {
            ffi::sqlite3_reset(cached_stmt);

            let key_cstr = CString::new(key)
                .map_err(|e| Error::InvalidParameter(format!("Invalid key: {}", e)))?;
            let val_cstr = CString::new(value)
                .map_err(|e| Error::InvalidParameter(format!("Invalid value: {}", e)))?;

            ffi::sqlite3_bind_text(
                cached_stmt,
                1,
                key_cstr.as_ptr(),
                -1,
                ffi::SQLITE_TRANSIENT(),
            );
            ffi::sqlite3_bind_text(
                cached_stmt,
                2,
                val_cstr.as_ptr(),
                -1,
                ffi::SQLITE_TRANSIENT(),
            );

            let rc = ffi::sqlite3_step(cached_stmt);
            ffi::sqlite3_reset(cached_stmt);

            if rc != ffi::SQLITE_DONE {
                return Err(Error::Sqlite(rusqlite::Error::SqliteFailure(
                    rusqlite::ffi::Error::new(rc),
                    None,
                )));
            }
            Ok(())
        };

        // Always update num_nodes and hnsw_version
        update_field("num_nodes", &self.num_nodes.to_string())?;
        update_field("hnsw_version", &self.hnsw_version.to_string())?;

        // Only update entry_point if it changed
        if update_entry_point {
            update_field("entry_point_rowid", &self.entry_point_rowid.to_string())?;
            update_field("entry_point_level", &self.entry_point_level.to_string())?;
        }

        Ok(())
    }

    /// Validate metadata is current and reload if changed by another connection
    /// Matches C implementation: hnsw_validate_and_refresh_caches()
    ///
    /// Returns Ok(true) if metadata was current, Ok(false) if reloaded, Err on failure
    pub fn validate_and_refresh(
        &mut self,
        db: &Connection,
        table_name: &str,
        column_name: &str,
    ) -> Result<bool> {
        let meta_table = format!("{}_{}_hnsw_meta", table_name, column_name);

        // Just check version first (1 query instead of 14)
        let current_version: Option<i64> = db
            .query_row(
                &format!(
                    "SELECT value FROM \"{}\" WHERE key = 'hnsw_version'",
                    meta_table
                ),
                [],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .and_then(|s| s.parse().ok());

        if let Some(curr_ver) = current_version
            && curr_ver != self.hnsw_version
        {
            // Version changed - reload full metadata
            if let Some(current) = HnswMetadata::load_from_db(db, table_name, column_name)? {
                *self = current;
                return Ok(false); // Metadata was stale, reloaded
            }
        }

        Ok(true) // Metadata is current
    }
}

/// Statement cache for HNSW operations (per connection, per vector column)
pub struct HnswStatementCache<'conn> {
    // Read operations
    pub get_node_data: Option<Statement<'conn>>,
    pub get_node_level: Option<Statement<'conn>>,
    pub get_edges: Option<Statement<'conn>>,
    pub get_meta_value: Option<Statement<'conn>>,

    // Write operations
    pub insert_node: Option<Statement<'conn>>,
    pub insert_edge: Option<Statement<'conn>>,
    pub delete_edges_from: Option<Statement<'conn>>,
    pub update_meta: Option<Statement<'conn>>,
}

impl<'conn> HnswStatementCache<'conn> {
    /// Create a new empty statement cache
    pub fn new() -> Self {
        HnswStatementCache {
            get_node_data: None,
            get_node_level: None,
            get_edges: None,
            get_meta_value: None,
            insert_node: None,
            insert_edge: None,
            delete_edges_from: None,
            update_meta: None,
        }
    }

    /// Clear all cached statements
    pub fn clear(&mut self) {
        *self = Self::new();
    }
}

impl<'conn> Default for HnswStatementCache<'conn> {
    fn default() -> Self {
        Self::new()
    }
}

// NOTE: HNSW operations use page-cache based design (no in-memory index struct)
// - insert::insert_hnsw() for vector insertion with HNSW indexing
// - search::search_hnsw() for k-NN queries using HNSW graph
// - rebuild::rebuild_hnsw_index() for rebuilding indexes
// All functions query shadow tables directly rather than maintaining in-memory state

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_params_default() {
        let params = HnswParams::default();
        assert_eq!(params.m, 32);
        assert_eq!(params.max_m0, 64);
        assert_eq!(params.ef_construction, 400);
        assert_eq!(params.ef_search, 200);
    }

    #[test]
    fn test_hnsw_params_high_recall() {
        let params = HnswParams::high_recall();
        assert_eq!(params.m, 32);
        assert_eq!(params.ef_construction, 400);
    }

    #[test]
    fn test_hnsw_params_tiers() {
        let hot = HnswParams::hot_tier();
        assert_eq!(hot.m, 32);
        assert_eq!(hot.ef_construction, 200);

        let warm = HnswParams::warm_tier();
        assert_eq!(warm.m, 64);
        assert_eq!(warm.ef_construction, 600);

        let cold = HnswParams::cold_tier();
        assert_eq!(cold.m, 96);
        assert_eq!(cold.ef_construction, 1000);
    }

    #[test]
    fn test_hnsw_metadata_save_and_load() {
        use crate::shadow;

        let db = Connection::open_in_memory().unwrap();

        // Create HNSW shadow tables
        shadow::create_hnsw_shadow_tables(&db, "test_table", "embedding").unwrap();

        // Create and save metadata
        let metadata = HnswMetadata::new(384, VectorType::Float32, DistanceMetric::L2);
        metadata.save_to_db(&db, "test_table", "embedding").unwrap();

        // Load it back
        let loaded = HnswMetadata::load_from_db(&db, "test_table", "embedding")
            .unwrap()
            .expect("Metadata should exist");

        assert_eq!(loaded.dimensions, 384);
        assert_eq!(loaded.entry_point_rowid, -1);
        assert_eq!(loaded.num_nodes, 0);
        assert_eq!(loaded.params.m, 32);
        assert_eq!(loaded.params.max_m0, 64);
    }

    #[test]
    fn test_hnsw_metadata_load_nonexistent() {
        let db = Connection::open_in_memory().unwrap();

        // Try to load from non-existent table (should return None gracefully)
        let result = HnswMetadata::load_from_db(&db, "nonexistent", "col");

        // Should not panic, just return None
        assert!(result.is_ok());
    }

    #[test]
    fn test_hnsw_statement_cache() {
        let cache = HnswStatementCache::new();
        assert!(cache.get_node_data.is_none());
        assert!(cache.insert_node.is_none());

        let mut cache2 = HnswStatementCache::default();
        cache2.clear();
        assert!(cache2.get_edges.is_none());
    }
}
