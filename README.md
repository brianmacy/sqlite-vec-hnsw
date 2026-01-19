# sqlite-vec-hnsw

A Rust-based SQLite extension for vector similarity search with HNSW (Hierarchical Navigable Small World) indexing.

> **Note:** This is a Rust port of [sqlite-vec](https://github.com/asg017/sqlite-vec) with modifications from Senzing. This project is pre-v1 and may have breaking changes.

## Features

- **Vector Storage**: Store float32, int8, and binary vectors in SQLite tables
- **Fast KNN Search**: K-nearest neighbor queries using HNSW approximate indexing
- **Distance Metrics**: L2 (Euclidean), L1 (Manhattan), Cosine, and Hamming distance
- **Multi-Tenant Support**: Partition keys for isolated vector indexes per tenant/user
- **Metadata Filtering**: Filter KNN results by metadata columns
- **Memory Efficient**: Quantized int8 vectors use 4× less memory than float32
- **SIMD Optimized**: Fast distance calculations using AVX/NEON instructions

## Installation

```bash
# Build the extension
cargo build --release

# The compiled extension will be at:
# target/release/libsqlite_vec_hnsw.so (Linux)
# target/release/libsqlite_vec_hnsw.dylib (macOS)
# target/release/sqlite_vec_hnsw.dll (Windows)
```

## Quick Start

### Loading the Extension

```sql
-- In sqlite3 CLI
.load target/release/libsqlite_vec_hnsw

-- Or programmatically
-- (method varies by language/SQLite library)
```

### Creating a Vector Table

```sql
CREATE VIRTUAL TABLE vec_documents USING vec0(
  embedding float[384]
);
```

### Inserting Vectors

```sql
-- Insert vectors as JSON arrays
INSERT INTO vec_documents(rowid, embedding) VALUES
  (1, '[-0.200, 0.250, 0.341, -0.211, ...]'),
  (2, '[0.443, -0.501, 0.355, -0.771, ...]'),
  (3, '[0.716, -0.927, 0.134, 0.052, ...]');

-- Or use compact binary format (more efficient)
INSERT INTO vec_documents(rowid, embedding)
  VALUES (4, vec_f32('[0.1, 0.2, 0.3, ...]'));
```

### Querying (K-Nearest Neighbors)

```sql
-- Find the 5 most similar vectors
SELECT rowid, distance
FROM vec_documents
WHERE embedding MATCH '[0.890, 0.544, 0.825, ...]'
  AND k = 5
ORDER BY distance;
```

**Important:** Always use `WHERE ... AND k = N`, not `LIMIT N` for KNN queries.

## Common Use Cases

### 1. Semantic Document Search

```sql
-- Create table with document metadata
CREATE VIRTUAL TABLE vec_docs USING vec0(
  doc_id INTEGER PRIMARY KEY,
  embedding float[768],
  +title TEXT,              -- auxiliary column (not indexed)
  +content TEXT,            -- auxiliary column
  category TEXT,            -- metadata (filterable in KNN)
  published_date TEXT
);

-- Insert documents
INSERT INTO vec_docs(doc_id, embedding, title, content, category, published_date)
VALUES
  (1, '[...]', 'Introduction to AI', 'AI is...', 'technology', '2024-01-15'),
  (2, '[...]', 'Cooking Tips', 'Here are...', 'food', '2024-02-20');

-- Search with metadata filtering
SELECT doc_id, title, category, distance
FROM vec_docs
WHERE embedding MATCH '[0.1, 0.2, ...]'
  AND k = 10
  AND category = 'technology'
  AND published_date >= '2024-01-01'
ORDER BY distance;
```

### 2. Multi-Tenant Vector Search

```sql
-- Create table with partition key for user isolation
CREATE VIRTUAL TABLE vec_user_docs USING vec0(
  user_id INTEGER PARTITION KEY,  -- shard index by user
  +contents TEXT,
  embedding float[1024]
);

-- Insert documents for different users
INSERT INTO vec_user_docs(rowid, user_id, contents, embedding) VALUES
  (1, 123, 'User 123 document 1', '[...]'),
  (2, 123, 'User 123 document 2', '[...]'),
  (3, 456, 'User 456 document 1', '[...]');

-- Query only user 123's vectors (efficient - uses partitioned index)
SELECT rowid, contents, distance
FROM vec_user_docs
WHERE embedding MATCH '[...]'
  AND user_id = 123
  AND k = 5
ORDER BY distance;
```

### 3. Movie Recommendation System

```sql
CREATE VIRTUAL TABLE vec_movies USING vec0(
  movie_id INTEGER PRIMARY KEY,
  synopsis_embedding float[768],
  +title TEXT,
  genre TEXT,
  rating FLOAT,
  num_reviews INTEGER
);

INSERT INTO vec_movies(movie_id, synopsis_embedding, title, genre, rating, num_reviews)
VALUES
  (1, '[...]', 'Interstellar', 'scifi', 5.0, 532),
  (2, '[...]', 'The Matrix', 'scifi', 4.5, 423),
  (3, '[...]', 'Get Out', 'horror', 4.9, 88);

-- Find similar sci-fi movies with good ratings
SELECT movie_id, title, genre, rating, distance
FROM vec_movies
WHERE synopsis_embedding MATCH '[...]'
  AND k = 10
  AND genre = 'scifi'
  AND rating > 4.0
  AND num_reviews > 100
ORDER BY distance;
```

## Vector Types and Quantization

### Float32 (Default)

```sql
CREATE VIRTUAL TABLE vec_float USING vec0(
  embedding float[128]  -- 4 bytes per dimension
);
```

### Int8 (Quantized - 4× Memory Savings)

```sql
CREATE VIRTUAL TABLE vec_quantized USING vec0(
  embedding int8[128]  -- 1 byte per dimension
);

-- Quantize float vectors to int8
INSERT INTO vec_quantized(rowid, embedding)
  SELECT rowid, vec_quantize_int8(embedding)
  FROM vec_float;
```

### Binary Vectors (32× Memory Savings)

```sql
CREATE VIRTUAL TABLE vec_binary USING vec0(
  embedding bit[1024]  -- 1 bit per dimension
);
```

## HNSW Index Configuration

```sql
CREATE VIRTUAL TABLE vec_optimized USING vec0(
  embedding float[768],
  use_hnsw=1,              -- Enable HNSW indexing (default: 1)
  hnsw_m=32,               -- Links per node (default: 32)
  hnsw_ef_construction=400 -- Build quality (default: 400)
);
```

**Parameter Guidelines:**
- **M**: Higher = better recall, more memory (typical: 16-96)
- **ef_construction**: Higher = better index quality, slower inserts (typical: 100-1000)
- **ef_search**: Adjustable per-query for recall/speed tradeoff (default: 200)

**Presets:**
- Fast inserts: `M=32, ef_construction=200`
- Balanced: `M=64, ef_construction=600`
- High quality: `M=96, ef_construction=1000`
- High recall (>95%): `M=32, ef_construction=400`

## Distance Functions

```sql
-- L2 (Euclidean) - default for KNN queries
SELECT vec_distance_l2('[1,2,3]', '[4,5,6]');

-- Cosine similarity (for normalized embeddings)
SELECT vec_distance_cosine('[1,2,3]', '[4,5,6]');

-- L1 (Manhattan)
SELECT vec_distance_l1('[1,2,3]', '[4,5,6]');

-- Hamming (for binary vectors)
SELECT vec_distance_hamming(vec_bit('...'), vec_bit('...'));
```

## Vector Operations

```sql
-- Get vector metadata
SELECT vec_length(embedding);        -- number of dimensions
SELECT vec_type(embedding);          -- 'float32', 'int8', or 'bit'

-- Convert to JSON
SELECT vec_to_json(embedding);

-- Vector arithmetic
SELECT vec_add('[1,2,3]', '[4,5,6]');     -- [5,7,9]
SELECT vec_sub('[4,5,6]', '[1,2,3]');     -- [3,3,3]

-- Normalization
SELECT vec_normalize('[3,4]');            -- [0.6, 0.8]

-- Slicing
SELECT vec_slice('[1,2,3,4,5]', 1, 3);    -- [2,3,4]
```

## Table Schema Options

```sql
CREATE VIRTUAL TABLE vec_example USING vec0(
  -- Primary key (optional)
  doc_id INTEGER PRIMARY KEY,

  -- Vector columns (required, at least one)
  title_embedding float[384],
  content_embedding float[768],

  -- Partition keys (optional, for multi-tenant)
  tenant_id INTEGER PARTITION KEY,
  category TEXT PARTITION KEY,

  -- Auxiliary columns (optional, stored but not indexed)
  +title TEXT,
  +content TEXT,
  +metadata JSON,

  -- Metadata columns (optional, filterable in KNN)
  author TEXT,
  rating FLOAT,
  published_date TEXT,

  -- Table options
  chunk_size=32,               -- chunk size for storage (default: varies)
  use_hnsw=1,                  -- enable HNSW index (default: 1)
  hnsw_m=32,                   -- HNSW M parameter
  hnsw_ef_construction=400     -- HNSW ef_construction parameter
);
```

## Performance Tips

1. **Use int8 quantization** for large datasets (4× memory reduction with minimal accuracy loss)
2. **Set appropriate k values** - larger k = slower but more comprehensive results
3. **Use partition keys** for multi-tenant scenarios to avoid scanning irrelevant data
4. **Tune HNSW parameters**:
   - Increase `ef_construction` for better recall (slower builds)
   - Increase `M` for better recall (more memory)
5. **Filter with metadata columns** instead of post-filtering in application code

## Troubleshooting

### Poor Recall Quality

- Increase `hnsw_ef_construction` when creating the table
- Increase `ef_search` for queries (if implemented)
- Use higher `M` value (e.g., 64 or 96)

### Slow Inserts

- Decrease `hnsw_ef_construction`
- Decrease `M` parameter
- Consider batch inserts in transactions

### High Memory Usage

- Use int8 quantization instead of float32
- Reduce `M` parameter
- Use partition keys to shard large indexes

## Limitations

- Pre-v1 software: expect breaking changes
- Maximum vector dimensions: implementation-dependent
- HNSW is approximate: does not guarantee exact KNN results

## Reference

For implementation details, see:
- Original C project: https://github.com/asg017/sqlite-vec
- Senzing modifications: `~/dev/G2/dev/libs/external/sqlite-vec`
- Architecture documentation: See `CLAUDE.md`

## License

Follows the licensing of the original sqlite-vec project (Apache 2.0 / MIT dual license).
