# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rust port of sqlite-vec, a SQLite extension providing vector search with HNSW indexing. Original C implementation at `~/dev/G2/dev/libs/external/sqlite-vec` (from https://github.com/asg017/sqlite-vec, modified by Senzing).

## Build Commands

```bash
# Build and test
cargo build
cargo test

# Lint (MUST pass)
cargo clippy --all-targets --all-features -- -D warnings

# Release build
cargo build --release
```

**Rust Edition:** 2024

## Architecture

### Virtual Table: `vec0`

Implements SQLite virtual table for vector storage and KNN search. Key concepts:

**Column Types:**
- Vector columns: `embedding float[768]` (also supports `int8[N]`, `bit[N]`)
- Partition keys: `user_id INTEGER PARTITION KEY` (for multi-tenant sharding)
- Auxiliary columns: `+contents TEXT` (stored but not indexed)
- Metadata columns: `category TEXT` (indexed for KNN filtering)

**Query Syntax:**
```sql
-- IMPORTANT: Use "AND k = N" constraint, NOT "LIMIT N"
SELECT rowid, distance
FROM table
WHERE embedding MATCH '[0.1, 0.2, ...]'
  AND k = 10
  AND user_id = 123  -- partition key filter
ORDER BY distance;
```

### Shadow Tables

Each `vec0` table creates shadow tables for persistence:
- `*_chunks` - Chunk metadata (chunk_id, size, validity bitmap, rowids)
- `*_rowids` - Rowid to chunk mapping
- `*_<column>_chunksNN` - Vector data storage
- `*_auxiliary` - Non-indexed column data
- `*_hnsw_nodes_<column>` - HNSW graph nodes (when use_hnsw=1)
- `*_hnsw_edges_<column>` - HNSW graph edges
- `*_hnsw_meta_<column>` - Index parameters

### Query Planning (idxStr Protocol)

The `idxStr` encodes query plans as a header character + 4-char blocks:
- Header: `'1'`=fullscan, `'2'`=point query, `'3'`=KNN
- Blocks map to `argv[i]` constraints:
  - `'{___'` - Query vector
  - `'}___'` - k limit
  - `']Xop_'` - Partition constraint (X=column index, op=operator)
  - `'&Xop_'` - Metadata constraint

### HNSW Implementation

**Key Parameters:**
- `M` (default: 32) - Links per node (higher = better recall, more memory)
- `ef_construction` (default: 400) - Build quality (higher = better graph, slower inserts)
- `ef_search` (default: 200) - Query quality (higher = better recall, slower search)

**Page-Cache Based:** Only stores metadata in memory, fetches nodes/edges from shadow tables on demand. This enables scaling to millions of vectors.

**Distance Metrics:** L2 (Euclidean), L1 (Manhattan), Cosine, Hamming. Should support SIMD (AVX512/AVX2/SSE on x86_64, NEON on ARM64).

## Testing Requirements

**Critical from global standards:**
- 100% test coverage, all tests must pass
- No mock tests - use real implementations only
- Recall quality MUST exceed 95% at production scale (100K+ vectors, 768D)
- HNSW must be truly optimized (not acting as flat index)
- Multi-process access must work correctly (transactional safety)
- Benchmarks must use actual data, not estimates

**Test against:** Original C implementation at `~/dev/G2/dev/libs/external/sqlite-vec`

## Reference Implementation

Key files in C codebase:
- `sqlite-vec.c` (14K LOC) - Main implementation
- `sqlite-vec-hnsw.c` (2K LOC) - HNSW algorithm
- `sqlite-vec-hnsw.h` - Data structures
- `CLAUDE.md` - Detailed development notes
- `ARCHITECTURE.md` - Internal architecture
- `test.sql` - SQL usage examples

## SQL Functions to Implement

**Vector constructors:** `vec_f32()`, `vec_int8()`, `vec_bit()`
**Distance functions:** `vec_distance_l2()`, `vec_distance_cosine()`, `vec_distance_l1()`, `vec_distance_hamming()`
**Operations:** `vec_length()`, `vec_type()`, `vec_to_json()`, `vec_add()`, `vec_sub()`, `vec_normalize()`
**Quantization:** `vec_quantize_int8()`, `vec_quantize_binary()`
**Management:** `vec_rebuild_hnsw()`, `vec_version()`, `vec_debug()`
