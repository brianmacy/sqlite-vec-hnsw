//! HNSW (Hierarchical Navigable Small World) index implementation

pub mod insert;
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
    pub entry_point_rowid: i64,  // -1 if empty
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
            match db.query_row(&query, [key], |row| row.get::<_, String>(0)).optional() {
                Ok(opt) => Ok(opt),
                Err(rusqlite::Error::SqliteFailure(err, _)) if err.code == rusqlite::ErrorCode::Unknown => {
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
            max_m0: get_meta("max_M0")?.and_then(|s| s.parse().ok()).unwrap_or(64),
            ef_construction: get_meta("ef_construction")?.and_then(|s| s.parse().ok()).unwrap_or(400),
            ef_search: get_meta("ef_search")?.and_then(|s| s.parse().ok()).unwrap_or(200),
            max_level: get_meta("max_level")?.and_then(|s| s.parse().ok()).unwrap_or(16),
            level_factor: get_meta("level_factor")?.and_then(|s| s.parse().ok()).unwrap_or(1.0 / 32.0_f64.ln()),
        };

        Ok(Some(HnswMetadata {
            params,
            entry_point_rowid,
            entry_point_level: get_meta("entry_point_level")?.and_then(|s| s.parse().ok()).unwrap_or(-1),
            num_nodes: get_meta("num_nodes")?.and_then(|s| s.parse().ok()).unwrap_or(0),
            dimensions: get_meta("dimensions")?.and_then(|s| s.parse().ok()).unwrap_or(0),
            element_type: get_meta("element_type")?.and_then(|s| match s.as_str() {
                "float32" => Some(VectorType::Float32),
                "int8" => Some(VectorType::Int8),
                "bit" => Some(VectorType::Bit),
                _ => None,
            }).unwrap_or(VectorType::Float32),
            distance_metric: get_meta("distance_metric")?.and_then(|s| match s.as_str() {
                "l2" => Some(DistanceMetric::L2),
                "cosine" => Some(DistanceMetric::Cosine),
                "l1" => Some(DistanceMetric::L1),
                _ => None,
            }).unwrap_or(DistanceMetric::L2),
            rng_seed: get_meta("rng_seed")?.and_then(|s| s.parse().ok()).unwrap_or(12345),
            hnsw_version: get_meta("hnsw_version")?.and_then(|s| s.parse().ok()).unwrap_or(1),
        }))
    }

    /// Save metadata to shadow table
    pub fn save_to_db(
        &self,
        db: &Connection,
        table_name: &str,
        column_name: &str,
    ) -> Result<()> {
        let meta_table = format!("{}_{}_hnsw_meta", table_name, column_name);
        let update_sql = format!(
            "INSERT OR REPLACE INTO \"{}\" (key, value) VALUES (?, ?)",
            meta_table
        );

        db.execute(&update_sql, ["M", &self.params.m.to_string()])?;
        db.execute(&update_sql, ["max_M0", &self.params.max_m0.to_string()])?;
        db.execute(&update_sql, ["ef_construction", &self.params.ef_construction.to_string()])?;
        db.execute(&update_sql, ["ef_search", &self.params.ef_search.to_string()])?;
        db.execute(&update_sql, ["max_level", &self.params.max_level.to_string()])?;
        db.execute(&update_sql, ["level_factor", &self.params.level_factor.to_string()])?;
        db.execute(&update_sql, ["entry_point_rowid", &self.entry_point_rowid.to_string()])?;
        db.execute(&update_sql, ["entry_point_level", &self.entry_point_level.to_string()])?;
        db.execute(&update_sql, ["num_nodes", &self.num_nodes.to_string()])?;
        db.execute(&update_sql, ["dimensions", &self.dimensions.to_string()])?;
        db.execute(&update_sql, ["element_type", self.element_type.as_str()])?;
        db.execute(&update_sql, ["distance_metric", self.distance_metric.as_str()])?;
        db.execute(&update_sql, ["rng_seed", &self.rng_seed.to_string()])?;
        db.execute(&update_sql, ["hnsw_version", &self.hnsw_version.to_string()])?;

        Ok(())
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

/// HNSW index (page-cache based)
pub struct HnswIndex {
    #[allow(dead_code)]
    metadata: HnswMetadata,
    // TODO: Add shadow table statement caches
}

impl HnswIndex {
    /// Create a new HNSW index
    pub fn new(_params: HnswParams, _dimensions: i32) -> Result<Self> {
        // TODO: Implement HNSW index creation
        Err(Error::NotImplemented(
            "HNSW index creation not yet implemented".to_string(),
        ))
    }

    /// Insert a vector into the index
    pub fn insert(&mut self, _rowid: i64, _vector: &[u8]) -> Result<()> {
        // TODO: Implement HNSW insertion
        Err(Error::NotImplemented(
            "HNSW insert not yet implemented".to_string(),
        ))
    }

    /// Search for k nearest neighbors
    pub fn search(&self, _query: &[u8], _k: usize) -> Result<Vec<(i64, f32)>> {
        // TODO: Implement HNSW search
        Err(Error::NotImplemented(
            "HNSW search not yet implemented".to_string(),
        ))
    }

    /// Rebuild the entire index
    pub fn rebuild(&mut self) -> Result<()> {
        // TODO: Implement index rebuild
        Err(Error::NotImplemented(
            "HNSW rebuild not yet implemented".to_string(),
        ))
    }
}

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
    fn test_hnsw_index_new_not_implemented() {
        let params = HnswParams::default();
        let result = HnswIndex::new(params, 768);
        assert!(result.is_err());
        assert!(matches!(result, Err(Error::NotImplemented(_))));
    }

    #[test]
    fn test_hnsw_metadata_save_and_load() {
        use crate::shadow;

        let db = Connection::open_in_memory().unwrap();

        // Create HNSW shadow tables
        shadow::create_hnsw_shadow_tables(&db, "test_table", "embedding").unwrap();

        // Create and save metadata
        let metadata = HnswMetadata::new(384, VectorType::Float32, DistanceMetric::L2);
        metadata
            .save_to_db(&db, "test_table", "embedding")
            .unwrap();

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
