//! HNSW (Hierarchical Navigable Small World) index implementation

pub mod cache;
pub mod insert;
pub mod rebuild;
pub mod search;
pub mod storage;

use crate::distance::DistanceMetric;
use crate::error::{Error, Result};
use crate::vector::{IndexQuantization, VectorType};
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
    /// Use simple pruning (just keep closest) instead of RNG heuristic
    /// RNG heuristic is O(n²) and dominates insert time for large M
    /// Setting this to true gives ~2x faster inserts with minimal recall impact
    pub simple_prune: bool,
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
            simple_prune: true, // Default to fast pruning
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

    /// Index storage quantization (e.g., int8 for HNSW nodes)
    pub index_quantization: IndexQuantization,

    /// RNG seed for level generation
    pub rng_seed: u32,

    /// Version tracking
    pub hnsw_version: i64,

    /// Whether to normalize vectors (always true for Cosine to enable L2 internal)
    pub normalize_vectors: bool,
}

impl HnswMetadata {
    /// Get the internal distance metric used for graph operations
    /// For Cosine with normalization, we use L2 internally for numerical stability
    pub fn internal_distance_metric(&self) -> DistanceMetric {
        if self.distance_metric == DistanceMetric::Cosine && self.normalize_vectors {
            DistanceMetric::L2
        } else {
            self.distance_metric
        }
    }

    /// Convert internal L2 distance to the user-requested metric
    /// For Cosine with normalized vectors: cosine_distance = L2² / 2
    pub fn convert_distance_for_output(&self, internal_dist: f32) -> f32 {
        if self.distance_metric == DistanceMetric::Cosine && self.normalize_vectors {
            // L2² / 2 = cosine_distance for normalized vectors
            (internal_dist * internal_dist) / 2.0
        } else {
            internal_dist
        }
    }
}

impl HnswMetadata {
    /// Create new metadata with default parameters
    pub fn new(dimensions: i32, element_type: VectorType, distance_metric: DistanceMetric) -> Self {
        Self::with_index_quantization(
            dimensions,
            element_type,
            distance_metric,
            IndexQuantization::None,
        )
    }

    /// Create new metadata with specific index quantization
    pub fn with_index_quantization(
        dimensions: i32,
        element_type: VectorType,
        distance_metric: DistanceMetric,
        index_quantization: IndexQuantization,
    ) -> Self {
        use std::collections::hash_map::RandomState;
        use std::hash::BuildHasher;

        // Generate a random seed
        let random_state = RandomState::new();
        let rng_seed = random_state.hash_one(std::time::SystemTime::now()) as u32;

        // Enable normalization by default for Cosine distance
        // This allows using L2 internally for better numerical stability
        let normalize_vectors = distance_metric == DistanceMetric::Cosine;

        HnswMetadata {
            params: HnswParams::default(),
            entry_point_rowid: -1,
            entry_point_level: -1,
            num_nodes: 0,
            dimensions,
            element_type,
            distance_metric,
            index_quantization,
            rng_seed,
            hnsw_version: 1,
            normalize_vectors,
        }
    }

    /// Load metadata from shadow table (single row schema)
    pub fn load_from_db(
        db: &Connection,
        table_name: &str,
        column_name: &str,
    ) -> Result<Option<Self>> {
        let meta_table = format!("{}_{}_hnsw_meta", table_name, column_name);

        // Single SELECT for all metadata (index_quantization, normalize_vectors are optional for backwards compat)
        let query = format!(
            "SELECT m, max_m0, ef_construction, ef_search, max_level, level_factor, \
             entry_point_rowid, entry_point_level, num_nodes, dimensions, \
             element_type, distance_metric, rng_seed, hnsw_version, \
             COALESCE(index_quantization, 'none') as index_quantization, \
             COALESCE(normalize_vectors, 1) as normalize_vectors \
             FROM \"{}\" WHERE id = 1",
            meta_table
        );

        let result = db
            .query_row(&query, [], |row| {
                let distance_metric = match row.get::<_, String>(11)?.as_str() {
                    "l2" => DistanceMetric::L2,
                    "cosine" => DistanceMetric::Cosine,
                    "l1" => DistanceMetric::L1,
                    _ => DistanceMetric::L2,
                };
                // Default normalize_vectors based on distance metric for backwards compat
                let normalize_vectors_int: i32 = row.get(15)?;
                let normalize_vectors = if normalize_vectors_int != 0 {
                    // If explicitly set to true or default (1), use it
                    // For cosine, default to true; for others, default to false
                    distance_metric == DistanceMetric::Cosine
                } else {
                    false
                };

                Ok(HnswMetadata {
                    params: HnswParams {
                        m: row.get(0)?,
                        max_m0: row.get(1)?,
                        ef_construction: row.get(2)?,
                        ef_search: row.get(3)?,
                        max_level: row.get(4)?,
                        level_factor: row.get(5)?,
                        simple_prune: true, // Default to fast pruning
                    },
                    entry_point_rowid: row.get(6)?,
                    entry_point_level: row.get(7)?,
                    num_nodes: row.get(8)?,
                    dimensions: row.get(9)?,
                    element_type: match row.get::<_, String>(10)?.as_str() {
                        "float32" => VectorType::Float32,
                        "int8" => VectorType::Int8,
                        "bit" => VectorType::Bit,
                        _ => VectorType::Float32,
                    },
                    distance_metric,
                    rng_seed: row.get::<_, i64>(12)? as u32,
                    hnsw_version: row.get(13)?,
                    index_quantization: match row.get::<_, String>(14)?.as_str() {
                        "int8" => IndexQuantization::Int8,
                        _ => IndexQuantization::None,
                    },
                    normalize_vectors,
                })
            })
            .optional();

        match result {
            Ok(opt) => Ok(opt),
            Err(rusqlite::Error::SqliteFailure(err, _))
                if err.code == rusqlite::ErrorCode::Unknown =>
            {
                // Table doesn't exist
                Ok(None)
            }
            Err(e) => Err(Error::Sqlite(e)),
        }
    }

    /// Save metadata to shadow table (single row schema)
    pub fn save_to_db(&self, db: &Connection, table_name: &str, column_name: &str) -> Result<()> {
        let meta_table = format!("{}_{}_hnsw_meta", table_name, column_name);

        // Single INSERT OR REPLACE for all metadata
        let sql = format!(
            "INSERT OR REPLACE INTO \"{}\" \
             (id, m, max_m0, ef_construction, ef_search, max_level, level_factor, \
              entry_point_rowid, entry_point_level, num_nodes, dimensions, \
              element_type, distance_metric, rng_seed, hnsw_version, index_quantization, normalize_vectors) \
             VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            meta_table
        );

        db.execute(
            &sql,
            rusqlite::params![
                self.params.m,
                self.params.max_m0,
                self.params.ef_construction,
                self.params.ef_search,
                self.params.max_level,
                self.params.level_factor,
                self.entry_point_rowid,
                self.entry_point_level,
                self.num_nodes,
                self.dimensions,
                self.element_type.as_str(),
                self.distance_metric.as_str(),
                self.rng_seed as i64,
                self.hnsw_version,
                self.index_quantization.as_str(),
                self.normalize_vectors as i32,
            ],
        )?;

        Ok(())
    }

    /// Save only dynamic fields that change during operations (single UPDATE)
    /// Only saves: entry_point, num_nodes, hnsw_version
    pub fn save_dynamic_to_db(
        &self,
        db: &Connection,
        table_name: &str,
        column_name: &str,
    ) -> Result<()> {
        let meta_table = format!("{}_{}_hnsw_meta", table_name, column_name);

        // Single UPDATE for dynamic fields
        let sql = format!(
            "UPDATE \"{}\" SET \
             entry_point_rowid = ?, \
             entry_point_level = ?, \
             num_nodes = ?, \
             hnsw_version = ? \
             WHERE id = 1",
            meta_table
        );

        db.execute(
            &sql,
            rusqlite::params![
                self.entry_point_rowid,
                self.entry_point_level,
                self.num_nodes,
                self.hnsw_version,
            ],
        )?;

        Ok(())
    }

    /// Save dynamic metadata using cached prepared statement (FAST PATH)
    /// Uses single UPDATE statement for all dynamic fields
    ///
    /// # Safety
    /// cached_stmt must be a valid prepared statement for:
    /// UPDATE meta SET entry_point_rowid=?, entry_point_level=?, num_nodes=?, hnsw_version=? WHERE id=1
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn save_dynamic_to_db_cached(
        &self,
        cached_stmt: *mut rusqlite::ffi::sqlite3_stmt,
        _update_entry_point: bool, // ignored - always update all dynamic fields
    ) -> Result<()> {
        use rusqlite::ffi;

        ffi::sqlite3_reset(cached_stmt);

        // Bind parameters: entry_point_rowid, entry_point_level, num_nodes, hnsw_version
        ffi::sqlite3_bind_int64(cached_stmt, 1, self.entry_point_rowid);
        ffi::sqlite3_bind_int(cached_stmt, 2, self.entry_point_level);
        ffi::sqlite3_bind_int(cached_stmt, 3, self.num_nodes);
        ffi::sqlite3_bind_int64(cached_stmt, 4, self.hnsw_version);

        let rc = ffi::sqlite3_step(cached_stmt);
        ffi::sqlite3_reset(cached_stmt);

        if rc != ffi::SQLITE_DONE {
            return Err(Error::Sqlite(rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rc),
                None,
            )));
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

        // Single query to check version
        let current_version: Option<i64> = db
            .query_row(
                &format!("SELECT hnsw_version FROM \"{}\" WHERE id = 1", meta_table),
                [],
                |row| row.get(0),
            )
            .optional()?;

        if let Some(curr_ver) = current_version
            && curr_ver != self.hnsw_version
        {
            // Version changed - reload full metadata (single SELECT)
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
