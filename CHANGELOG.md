# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-01-30

### Added
- StmtHandleGuard for RAII-based statement reset in shared-cache mode
- Non-vector column support with proper type preservation and cleanup

### Changed
- **Major Architecture Refactor**: Unified storage architecture for improved maintainability
- **Performance Optimization**: Zero-copy vectors and search caching in HNSW
- **FFI Optimization**: Small batch path + skip redundant resets for improved performance
- **HNSW Prune Optimization**: Use stored edge distances for 2x speedup

### Fixed
- Improved error messaging across the codebase
- Fixed shadow table column indexing for tables with non-vector columns first
- Fixed multithread stress test: enable HNSW index

### Performance Improvements
- 2x speedup in HNSW pruning operations
- Reduced FFI barrier overhead with smart batching
- Zero-copy vector handling eliminates unnecessary allocations

## [0.1.0] - 2026-01-19

### Added
- **KNN Query Support**: Full implementation of MATCH operator for k-nearest neighbor queries
  - Queries like `WHERE vector MATCH vec_f32('[...]') AND k = 10` now work
  - HNSW search when index available
  - Automatic brute force fallback for small datasets or when HNSW not built

- **Transaction Support**: Implemented TransactionVTab trait
  - begin(), sync(), commit(), rollback() hooks
  - Full ACID guarantees through shadow table persistence
  - Proper cleanup on transaction boundaries

- **C Compatibility**: Full compatibility with original C sqlite-vec implementation
  - Successfully read C-created databases (verified with 24,902 vectors)
  - Shadow table schemas match C version exactly
  - HNSW metadata format compatible
  - Can query HNSW graph structure from C databases

- **Brute Force Fallback**: Automatic graceful degradation
  - Scans all vectors from shadow tables when HNSW not available
  - Perfect for small datasets (< 1000 vectors)
  - Seamless experience for users

- **Index Management**:
  - vec_rebuild_hnsw() function to rebuild HNSW indexes
  - Support for updating M and ef_construction parameters
  - PRAGMA integrity_check support for validating indexes

- **Comprehensive Testing**:
  - 109+ tests covering all core functionality
  - Unit tests (79)
  - Integration tests (21)
  - KNN query tests
  - Rebuild function tests (4)
  - Shadow table tests (2)
  - C compatibility tests (2)
  - Scale tests (10K and 100K vectors)

- **Examples**:
  - basic_usage.rs - Demonstrates CRUD operations, shadow tables, HNSW stats
  - similarity_search.rs - Shows HNSW vs brute force with recall measurement

### Changed
- Updated README with production-ready status
- Improved documentation with Rust API examples
- Applied cargo fmt formatting across codebase

### Fixed
- MATCH operator registration now works in both create() and connect()
- Fixed k column index calculation for hidden columns
- Resolved clippy warnings throughout codebase

### Technical Details
- Shadow table-based persistence (scales to millions of vectors)
- Page-cache based HNSW design (minimal memory footprint)
- SIMD-optimized distance calculations using simsimd
- Chunk-based vector storage with validity bitmaps
- Transaction hooks for proper ACID semantics

### Performance Characteristics
- Insert rate: ~100-200 vectors/sec with HNSW indexing (768D)
- Query latency: ~2-5ms for k=10 on 100K vectors (768D)
- Recall: >95% at k=10 with default HNSW parameters
- Memory: Minimal (only metadata ~64 bytes per index in memory)

### Known Limitations
- Partition keys not yet implemented
- Metadata filtering in KNN queries not yet implemented
- Tested up to 768D vectors
- HNSW is approximate (does not guarantee exact KNN results)

## [0.0.1] - 2026-01-15

### Added
- Initial implementation
- Basic shadow table structure
- HNSW index building
- SQL scalar functions (vec_f32, vec_distance_l2, etc.)
- Virtual table infrastructure
- Basic CRUD operations

[Unreleased]: https://github.com/brianmacy/sqlite-vec-hnsw/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/brianmacy/sqlite-vec-hnsw/releases/tag/v0.4.0
[0.1.0]: https://github.com/brianmacy/sqlite-vec-hnsw/releases/tag/v0.1.0
[0.0.1]: https://github.com/brianmacy/sqlite-vec-hnsw/releases/tag/v0.0.1
