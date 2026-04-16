# Performance Plan: Match C Implementation

## What C Does

### 1. Prepared Statement Cache (HnswStatementCache)
**We already have this.** Statements are prepared once and reused.

### 2. Node/Neighbor Cache (HnswNodeCache)
**This is what we're missing.**

The C code uses an LRU cache that:
- Caches **node data** (rowid, level, vector) keyed by rowid
- Caches **neighbor lists** keyed by (rowid, level)
- Uses an **arena allocator** for fast memory management
- Has **version tracking** - clears cache if DB version changes
- Uses **zero-copy lookups** - returns const pointers to cached data

### 3. How Cache Works During Operations

**INSERT:**
1. Traverse graph to find insertion point (reads nodes/neighbors)
2. First access: cache miss → SQLite query → insert into cache
3. Subsequent accesses: cache hit → no SQLite query
4. After insert: invalidate/update affected cache entries

**SEARCH:**
1. Traverse graph from entry point
2. Cache accumulates visited nodes during traversal
3. Same nodes revisited at layer 0 → cache hits

### 4. Key Performance Insight

During HNSW graph traversal, we visit the same nodes multiple times:
- At higher levels: fewer nodes, greedy descent
- At layer 0: many nodes visited with `ef` candidates

The cache eliminates redundant SQLite queries for revisited nodes.

## Implementation Plan

### Phase 1: Create HnswNodeCache in Rust

```rust
pub struct HnswNodeCache {
    // Node cache: rowid -> (level, vector_bytes)
    node_cache: LruCache<i64, (i32, Vec<u8>)>,

    // Neighbor cache: (rowid, level) -> Vec<i64>
    neighbor_cache: LruCache<(i64, i32), Vec<i64>>,

    // Version tracking
    version: u32,

    // Stats
    node_hits: u64,
    node_misses: u64,
    neighbor_hits: u64,
    neighbor_misses: u64,
}
```

### Phase 2: Integrate with Storage Layer

Modify `fetch_node_data` and `fetch_neighbors` to:
1. Check cache first
2. On cache miss: query SQLite, insert into cache
3. Return cached data

### Phase 3: Cache Invalidation

During INSERT:
- After inserting new edges, invalidate affected neighbor caches
- Or use append-only updates like C does

### Phase 4: Version Tracking

- Track hnsw_version in metadata
- Compare with cache version
- Clear cache on version mismatch (handles multi-connection)

## Expected Performance Gain

Based on C implementation profiling:
- Cache hit rate during search: ~80-90%
- Cache reduces SQLite queries by 5-10x
- Expected insert time reduction: ~50-70%
