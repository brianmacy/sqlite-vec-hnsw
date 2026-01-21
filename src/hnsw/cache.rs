//! HNSW Node and Neighbor Cache
//!
//! Caches node data and neighbor lists during HNSW operations to reduce
//! SQLite queries. This mirrors the C implementation's HnswNodeCache.

use std::collections::HashMap;

/// Maximum cache entries (matches C implementation default)
const DEFAULT_MAX_ENTRIES: usize = 10000;

/// Cached node data
#[derive(Debug, Clone)]
pub struct CachedNode {
    pub rowid: i64,
    pub level: i32,
    pub vector: Vec<u8>,
}

/// HNSW node and neighbor cache
///
/// Caches recently accessed nodes and neighbor lists to avoid redundant
/// SQLite queries during graph traversal operations.
pub struct HnswCache {
    /// Node cache: rowid -> CachedNode
    nodes: HashMap<i64, CachedNode>,

    /// Neighbor cache: (rowid, level) -> Vec<neighbor_rowid>
    neighbors: HashMap<(i64, i32), Vec<i64>>,

    /// Neighbor cache with distances: (rowid, level) -> Vec<(neighbor_rowid, distance)>
    neighbors_with_dist: HashMap<(i64, i32), Vec<(i64, f32)>>,

    /// Version tracking for cache invalidation
    version: u32,

    /// Maximum entries (for each cache type)
    max_entries: usize,

    /// Statistics
    pub node_hits: u64,
    pub node_misses: u64,
    pub neighbor_hits: u64,
    pub neighbor_misses: u64,
}

impl Default for HnswCache {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_ENTRIES)
    }
}

impl HnswCache {
    /// Create a new cache with specified max entries
    pub fn new(max_entries: usize) -> Self {
        HnswCache {
            nodes: HashMap::with_capacity(max_entries),
            neighbors: HashMap::with_capacity(max_entries),
            neighbors_with_dist: HashMap::with_capacity(max_entries),
            version: 0,
            max_entries,
            node_hits: 0,
            node_misses: 0,
            neighbor_hits: 0,
            neighbor_misses: 0,
        }
    }

    /// Look up cached node data
    ///
    /// Returns Some if found in cache (cache hit), None if not (cache miss)
    pub fn lookup_node(&mut self, rowid: i64) -> Option<&CachedNode> {
        if let Some(node) = self.nodes.get(&rowid) {
            self.node_hits += 1;
            Some(node)
        } else {
            self.node_misses += 1;
            None
        }
    }

    /// Insert node data into cache
    pub fn insert_node(&mut self, rowid: i64, level: i32, vector: Vec<u8>) {
        // Simple eviction: clear half the cache when full
        if self.nodes.len() >= self.max_entries {
            self.nodes.clear();
        }
        self.nodes.insert(
            rowid,
            CachedNode {
                rowid,
                level,
                vector,
            },
        );
    }

    /// Look up cached neighbor list
    pub fn lookup_neighbors(&mut self, rowid: i64, level: i32) -> Option<&Vec<i64>> {
        if let Some(neighbors) = self.neighbors.get(&(rowid, level)) {
            self.neighbor_hits += 1;
            Some(neighbors)
        } else {
            self.neighbor_misses += 1;
            None
        }
    }

    /// Insert neighbor list into cache
    pub fn insert_neighbors(&mut self, rowid: i64, level: i32, neighbors: Vec<i64>) {
        // Simple eviction: clear half the cache when full
        if self.neighbors.len() >= self.max_entries {
            self.neighbors.clear();
        }
        self.neighbors.insert((rowid, level), neighbors);
    }

    /// Look up cached neighbor list with distances
    pub fn lookup_neighbors_with_dist(
        &mut self,
        rowid: i64,
        level: i32,
    ) -> Option<&Vec<(i64, f32)>> {
        if let Some(neighbors) = self.neighbors_with_dist.get(&(rowid, level)) {
            self.neighbor_hits += 1;
            Some(neighbors)
        } else {
            self.neighbor_misses += 1;
            None
        }
    }

    /// Insert neighbor list with distances into cache
    pub fn insert_neighbors_with_dist(
        &mut self,
        rowid: i64,
        level: i32,
        neighbors: Vec<(i64, f32)>,
    ) {
        // Simple eviction: clear half the cache when full
        if self.neighbors_with_dist.len() >= self.max_entries {
            self.neighbors_with_dist.clear();
        }
        self.neighbors_with_dist.insert((rowid, level), neighbors);
    }

    /// Append a neighbor to cached neighbor list (for insert operations)
    pub fn append_neighbor(&mut self, rowid: i64, level: i32, new_neighbor: i64) {
        if let Some(neighbors) = self.neighbors.get_mut(&(rowid, level)) {
            neighbors.push(new_neighbor);
        }
    }

    /// Append a neighbor with distance to cached neighbor list
    pub fn append_neighbor_with_dist(
        &mut self,
        rowid: i64,
        level: i32,
        new_neighbor: i64,
        distance: f32,
    ) {
        if let Some(neighbors) = self.neighbors_with_dist.get_mut(&(rowid, level)) {
            neighbors.push((new_neighbor, distance));
        }
    }

    /// Invalidate cached neighbors for a node (called when edges change)
    pub fn invalidate_neighbors(&mut self, rowid: i64, level: i32) {
        self.neighbors.remove(&(rowid, level));
        self.neighbors_with_dist.remove(&(rowid, level));
    }

    /// Invalidate all neighbors for a node at all levels
    pub fn invalidate_all_neighbors(&mut self, rowid: i64) {
        self.neighbors.retain(|&(r, _), _| r != rowid);
        self.neighbors_with_dist.retain(|&(r, _), _| r != rowid);
    }

    /// Clear all cached data
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.neighbors.clear();
        self.neighbors_with_dist.clear();
    }

    /// Check version and clear if mismatched
    pub fn check_version(&mut self, db_version: u32) -> bool {
        if self.version != db_version {
            self.clear();
            self.version = db_version;
            false
        } else {
            true
        }
    }

    /// Set cache version
    pub fn set_version(&mut self, version: u32) {
        self.version = version;
    }

    /// Get cache statistics as formatted string
    pub fn stats_string(&self) -> String {
        let node_total = self.node_hits + self.node_misses;
        let neighbor_total = self.neighbor_hits + self.neighbor_misses;

        let node_hit_rate = if node_total > 0 {
            (self.node_hits as f64 / node_total as f64) * 100.0
        } else {
            0.0
        };

        let neighbor_hit_rate = if neighbor_total > 0 {
            (self.neighbor_hits as f64 / neighbor_total as f64) * 100.0
        } else {
            0.0
        };

        format!(
            "Node cache: {} hits, {} misses ({:.1}% hit rate)\n\
             Neighbor cache: {} hits, {} misses ({:.1}% hit rate)\n\
             Entries: {} nodes, {} neighbors",
            self.node_hits,
            self.node_misses,
            node_hit_rate,
            self.neighbor_hits,
            self.neighbor_misses,
            neighbor_hit_rate,
            self.nodes.len(),
            self.neighbors.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_cache() {
        let mut cache = HnswCache::new(100);

        // Miss first
        assert!(cache.lookup_node(1).is_none());
        assert_eq!(cache.node_misses, 1);

        // Insert
        cache.insert_node(1, 2, vec![1, 2, 3, 4]);

        // Hit
        let node = cache.lookup_node(1).unwrap();
        assert_eq!(node.rowid, 1);
        assert_eq!(node.level, 2);
        assert_eq!(node.vector, vec![1, 2, 3, 4]);
        assert_eq!(cache.node_hits, 1);
    }

    #[test]
    fn test_neighbor_cache() {
        let mut cache = HnswCache::new(100);

        // Miss first
        assert!(cache.lookup_neighbors(1, 0).is_none());
        assert_eq!(cache.neighbor_misses, 1);

        // Insert
        cache.insert_neighbors(1, 0, vec![2, 3, 4]);

        // Hit
        let neighbors = cache.lookup_neighbors(1, 0).unwrap();
        assert_eq!(*neighbors, vec![2, 3, 4]);
        assert_eq!(cache.neighbor_hits, 1);
    }

    #[test]
    fn test_append_neighbor() {
        let mut cache = HnswCache::new(100);

        cache.insert_neighbors(1, 0, vec![2, 3]);
        cache.append_neighbor(1, 0, 4);

        let neighbors = cache.lookup_neighbors(1, 0).unwrap();
        assert_eq!(*neighbors, vec![2, 3, 4]);
    }

    #[test]
    fn test_version_check() {
        let mut cache = HnswCache::new(100);
        cache.insert_node(1, 0, vec![1, 2, 3]);
        cache.set_version(1);

        // Same version - should not clear
        assert!(cache.check_version(1));
        assert!(cache.lookup_node(1).is_some());

        // Different version - should clear
        assert!(!cache.check_version(2));
        // After clear, lookup will be a miss (need to reset hit counter mentally)
    }
}
