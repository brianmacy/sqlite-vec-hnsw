# Detailed Pseudo-Code: C Implementation HNSW Insert Path

## Overview
This document describes EXACTLY how the C sqlite-vec implementation inserts a vector into the HNSW index, including all IO, memory allocation, caching, and SQL operations.

## Key Data Structures

```c
struct HnswStatementCache {
    sqlite3_stmt *get_node_data;      // SELECT rowid, level, vector FROM nodes WHERE rowid=?
    sqlite3_stmt *get_edges;          // SELECT to_rowid FROM edges WHERE from_rowid=? AND level=?
    sqlite3_stmt *insert_node;        // INSERT INTO nodes (rowid, level, vector) VALUES (?,?,?)
    sqlite3_stmt *insert_edge;        // INSERT INTO edges (from_rowid, to_rowid, level) VALUES (?,?,?)
    sqlite3_stmt *delete_edges_from;  // DELETE FROM edges WHERE from_rowid=? AND level=?
    sqlite3_stmt *update_meta;        // INSERT OR REPLACE INTO meta (key,value) VALUES (?,?)

    HnswNodeCache *node_cache;        // NULL if SQLITE_VEC_DISABLE_HNSW_CACHE is defined
    char *table_name;
    char *column_name;
}

struct HnswMetadata {
    i32 M;                    // Links per node (default: 32)
    i32 max_M0;               // Max links at layer 0 (default: 64)
    i32 ef_construction;      // Build quality (default: 400)
    i32 ef_search;            // Search quality (default: 200)
    f64 level_factor;         // 1/ln(M) for level generation

    i64 entry_point_rowid;    // Current entry point (-1 if empty)
    i32 entry_point_level;    // Level of entry point
    i32 num_nodes;            // Node count

    i32 dimensions;
    enum VectorType element_type;
    enum DistanceMetric distance_metric;
    u32 rng_seed;
    i64 hnsw_version;         // Incremented on each modification
}

// Priority queue for search (max-heap by distance)
struct HnswCandidateQueue {
    struct { i64 rowid; f32 distance; } *items;
    i32 size;
    i32 capacity;
}

// Visited set (hash table for O(1) lookup)
struct HnswVisitedSet {
    i64 *rowids;
    i32 *slots_used;
    i32 capacity;
    i32 mask;  // capacity - 1 (capacity is power of 2)
}
```

## Main Insert Function: hnsw_insert_query_based()

```
FUNCTION hnsw_insert_query_based(db, stmts, meta, rowid, vector, vector_size):
    // ============================================================
    // PHASE 0: VALIDATE CACHES (multi-connection safety)
    // ============================================================

    // Check if DB has changed since we last read metadata
    CALL hnsw_validate_and_refresh_caches(db, stmts, meta, "INSERT")
    // This function:
    //   1. Reads PRAGMA data_version (SQLite's DB change counter)
    //   2. Reads hnsw_version from meta table (our HNSW change counter)
    //   3. If either changed: clears cache, reloads metadata
    //   4. Returns SQLITE_OK or error

    IF error THEN RETURN error

    // ============================================================
    // PHASE 1: GENERATE LEVEL & INSERT NODE
    // ============================================================

    // Generate random level using exponential decay
    level = hnsw_gen_level_query(meta)
    // Algorithm:
    //   uniform_random = random_float(0, 1) using meta->rng_seed
    //   level = floor(-ln(uniform_random) * meta->level_factor)
    //   CLAMP level to [0, meta->max_level]

    // Insert node into shadow table using PREPARED STATEMENT
    CALL sqlite3_reset(stmts->insert_node)
    CALL sqlite3_bind_int64(stmts->insert_node, 1, rowid)
    CALL sqlite3_bind_int(stmts->insert_node, 2, level)
    CALL sqlite3_bind_blob(stmts->insert_node, 3, vector, vector_size, SQLITE_TRANSIENT)
    rc = sqlite3_step(stmts->insert_node)
    CALL sqlite3_reset(stmts->insert_node)

    IF rc != SQLITE_DONE THEN RETURN error

    // SQLite IO: 1 INSERT (goes to WAL journal, not yet committed)
    // Memory: No heap allocation in this step

    // If cache is enabled, add node to cache (DISABLED BY DEFAULT)
    IF stmts->node_cache != NULL THEN
        hnsw_cache_insert_node(stmts->node_cache, rowid, level, vector, vector_size)
        // This makes a COPY of the vector into cache's arena allocator
    END IF

    // ============================================================
    // PHASE 2: HANDLE FIRST NODE (EMPTY INDEX)
    // ============================================================

    IF meta->entry_point_rowid == -1 THEN
        // This is the first node - set as entry point
        meta->entry_point_rowid = rowid
        meta->entry_point_level = level
        meta->num_nodes = 1

        // Write to metadata table (3 SQL statements)
        sprintf(value_buf, "%lld", rowid)
        CALL hnsw_update_metadata_value(stmts->update_meta, "entry_point_rowid", value_buf)

        sprintf(value_buf, "%d", level)
        CALL hnsw_update_metadata_value(stmts->update_meta, "entry_point_level", value_buf)

        sprintf(value_buf, "1")
        CALL hnsw_update_metadata_value(stmts->update_meta, "num_nodes", value_buf)

        // SQLite IO: 3 INSERT OR REPLACE into meta table
        // Memory: No heap allocation

        GOTO finish  // Done - no edges to create for first node
    END IF

    // ============================================================
    // PHASE 3: INCREMENT NODE COUNT
    // ============================================================

    meta->num_nodes++
    sprintf(value_buf, "%d", meta->num_nodes)
    CALL hnsw_update_metadata_value(stmts->update_meta, "num_nodes", value_buf)

    // SQLite IO: 1 INSERT OR REPLACE into meta table

    // ============================================================
    // PHASE 4: FIND INSERTION POINT (greedy traversal from top)
    // ============================================================

    current_nearest = meta->entry_point_rowid

    // Traverse from entry level down to insertion level+1
    FOR lc FROM meta->entry_point_level DOWN TO (level + 1):
        // Search this layer to find nearest node
        ALLOCATE layer_neighbors = NULL
        ALLOCATE layer_distances = NULL
        layer_count = 0

        rc = hnsw_search_layer_query(db, stmts, meta, current_nearest, vector, lc, 1,
                                      &layer_neighbors, &layer_distances, &layer_count)

        // hnsw_search_layer_query() allocates:
        //   - layer_neighbors: malloc(layer_count * sizeof(i64))
        //   - layer_distances: malloc(layer_count * sizeof(f32))
        // We must FREE these after use

        IF rc == SQLITE_OK AND layer_count > 0 THEN
            current_nearest = layer_neighbors[0]  // Move to nearest
            FREE layer_neighbors
            FREE layer_distances
        ELSE
            // On error, stay at current position
            IF layer_neighbors != NULL THEN FREE layer_neighbors
            IF layer_distances != NULL THEN FREE layer_distances
        END IF
    END FOR

    // SQLite IO during this phase: See hnsw_search_layer_query() details below
    // Memory: Allocates/frees temporary arrays for each layer

    // ============================================================
    // PHASE 5: INSERT AT EACH LEVEL (from insertion level down to 0)
    // ============================================================

    FOR lc FROM level DOWN TO 0:
        // ---------------------------------------------------------
        // STEP 5a: Find ef_construction nearest neighbors
        // ---------------------------------------------------------

        ALLOCATE candidates = NULL
        ALLOCATE distances = NULL
        candidate_count = 0

        rc = hnsw_search_layer_query(db, stmts, meta, current_nearest, vector, lc,
                                      meta->params.ef_construction,
                                      &candidates, &distances, &candidate_count)

        IF rc != SQLITE_OK OR candidate_count == 0 THEN
            IF candidates != NULL THEN FREE candidates
            IF distances != NULL THEN FREE distances
            CONTINUE  // Skip this level on error
        END IF

        // SQLite IO: See hnsw_search_layer_query() details below
        // Memory: candidates and distances are malloc'd by search_layer

        // ---------------------------------------------------------
        // STEP 5b: Select M neighbors from candidates
        // ---------------------------------------------------------

        M = (lc == 0) ? meta->params.max_M0 : meta->params.M
        selected_count = 0
        selected = hnsw_select_neighbors(candidates, distances, candidate_count, M, &selected_count)

        // hnsw_select_neighbors() does:
        //   1. ALLOCATE selected = malloc(min(candidate_count, M) * sizeof(i64))
        //   2. Bubble sort candidates by distance (IN-PLACE modification)
        //   3. Copy first M rowids to selected array
        //   4. Return selected (caller must FREE)

        IF selected == NULL THEN
            FREE candidates
            FREE distances
            CONTINUE
        END IF

        // Memory: selected is malloc'd, we must FREE it

        // ---------------------------------------------------------
        // STEP 5c: Create bidirectional edges
        // ---------------------------------------------------------

        FOR i FROM 0 TO selected_count - 1:
            neighbor_rowid = selected[i]

            // Add edge: new_node -> neighbor
            CALL sqlite3_reset(stmts->insert_edge)
            CALL sqlite3_bind_int64(stmts->insert_edge, 1, rowid)  // from
            CALL sqlite3_bind_int64(stmts->insert_edge, 2, neighbor_rowid)  // to
            CALL sqlite3_bind_int(stmts->insert_edge, 3, lc)  // level
            rc = sqlite3_step(stmts->insert_edge)
            CALL sqlite3_reset(stmts->insert_edge)

            // SQLite IO: 1 INSERT into edges table

            // Update cache if enabled
            IF stmts->node_cache != NULL THEN
                hnsw_cache_append_neighbor(stmts->node_cache, rowid, lc, neighbor_rowid)
            END IF

            // Add edge: neighbor -> new_node (bidirectional)
            CALL sqlite3_reset(stmts->insert_edge)
            CALL sqlite3_bind_int64(stmts->insert_edge, 1, neighbor_rowid)  // from
            CALL sqlite3_bind_int64(stmts->insert_edge, 2, rowid)  // to
            CALL sqlite3_bind_int(stmts->insert_edge, 3, lc)  // level
            rc = sqlite3_step(stmts->insert_edge)
            CALL sqlite3_reset(stmts->insert_edge)

            // SQLite IO: 1 INSERT into edges table

            // Update cache if enabled
            IF stmts->node_cache != NULL THEN
                hnsw_cache_append_neighbor(stmts->node_cache, neighbor_rowid, lc, rowid)
            END IF

            // ---------------------------------------------------------
            // STEP 5d: Prune neighbor's connections (maintain small-world)
            // ---------------------------------------------------------

            CALL hnsw_prune_neighbor_connections(db, stmts, meta, neighbor_rowid, rowid, lc, M)
            // See detailed pseudo-code below
            // SQLite IO: Variable (fetches + deletes + re-inserts edges)
            // Memory: Allocates temp arrays
        END FOR

        // Update current_nearest for next level
        IF selected_count > 0 THEN
            current_nearest = selected[0]
        END IF

        FREE selected
        FREE candidates
        FREE distances

        // SQLite IO total per level:
        //   - 1 search_layer_query (many SELECTs, see below)
        //   - 2 * selected_count INSERTs into edges
        //   - selected_count * prune operations (see prune pseudo-code)

    END FOR

    // ============================================================
    // PHASE 6: UPDATE ENTRY POINT IF NEEDED
    // ============================================================

    IF level > meta->entry_point_level THEN
        meta->entry_point_rowid = rowid
        meta->entry_point_level = level

        sprintf(value_buf, "%lld", rowid)
        CALL hnsw_update_metadata_value(stmts->update_meta, "entry_point_rowid", value_buf)

        sprintf(value_buf, "%d", level)
        CALL hnsw_update_metadata_value(stmts->update_meta, "entry_point_level", value_buf)

        // SQLite IO: 2 INSERT OR REPLACE into meta table
    END IF

    // ============================================================
    // PHASE 7: INCREMENT HNSW VERSION
    // ============================================================

finish:
    // Read current version from DB
    current_hnsw_version = 0
    CALL hnsw_get_hnsw_version(db, stmts->table_name, stmts->column_name, &current_hnsw_version)
    // This does: SELECT value FROM meta WHERE key='hnsw_version'

    new_hnsw_version = current_hnsw_version + 1

    // Write new version
    CALL hnsw_set_hnsw_version(stmts->update_meta, new_hnsw_version)
    // This does: INSERT OR REPLACE INTO meta (key,value) VALUES ('hnsw_version', ?)

    // Update cache version to match
    IF stmts->node_cache != NULL THEN
        hnsw_cache_set_version(stmts->node_cache, new_hnsw_version)
    END IF

    // SQLite IO: 1 SELECT + 1 INSERT OR REPLACE

    RETURN SQLITE_OK
END FUNCTION


## Helper Function: hnsw_search_layer_query()

This is the CORE traversal algorithm that gets called multiple times during insert.

```
FUNCTION hnsw_search_layer_query(db, stmts, meta, entry_rowid, query_vector, level, ef,
                                   out_rowids, out_distances, out_count):

    // ============================================================
    // INITIALIZATION
    // ============================================================

    // Allocate priority queues (max-heaps)
    ALLOCATE candidates = malloc(sizeof(HnswCandidateQueue))
    CALL hnsw_queue_init(&candidates, ef * 2)
    // Allocates: candidates.items = malloc(ef * 2 * sizeof(struct {i64, f32}))

    ALLOCATE results = malloc(sizeof(HnswCandidateQueue))
    CALL hnsw_queue_init(&results, ef)
    // Allocates: results.items = malloc(ef * sizeof(struct {i64, f32}))

    // Allocate visited set (hash table)
    ALLOCATE visited = malloc(sizeof(HnswVisitedSet))
    CALL hnsw_visited_init(&visited, ef * 4)
    // Allocates:
    //   visited.rowids = malloc(ef * 4 * sizeof(i64))
    //   visited.slots_used = malloc(ef * 4 * sizeof(i32))
    // Initializes all to -1 (empty)

    // ============================================================
    // FETCH ENTRY POINT
    // ============================================================

    entry_vector_ptr = NULL
    entry_vector_size = 0
    entry_level = 0

    rc = hnsw_fetch_node_data(stmts, entry_rowid, &entry_level, &entry_vector_ptr, &entry_vector_size)
    // See hnsw_fetch_node_data() pseudo-code below
    // SQLite IO: 1 SELECT from nodes table (unless in cache)
    // Memory: Allocates entry_vector_ptr = malloc(entry_vector_size)

    IF rc != SQLITE_OK THEN
        CLEANUP and RETURN error
    END IF

    // Calculate distance to entry point
    entry_distance = hnsw_calc_distance(meta, query_vector, entry_vector_ptr)
    // No IO, pure computation (SIMD if available)

    // Free vector immediately - we only needed it for distance
    FREE entry_vector_ptr

    // Add entry point to queues
    CALL hnsw_queue_push(&candidates, entry_rowid, -entry_distance)  // Negative for max-heap
    CALL hnsw_queue_push(&results, entry_rowid, entry_distance)
    CALL hnsw_visited_add(&visited, entry_rowid)
    // Hash function: (rowid * 0x9E3779B97F4A7C15ULL) & mask

    // ============================================================
    // GREEDY SEARCH LOOP
    // ============================================================

    WHILE candidates.size > 0:
        // ---------------------------------------------------------
        // Pop best candidate (heap pop)
        // ---------------------------------------------------------

        current_rowid = candidates.items[0].rowid
        current_neg_dist = candidates.items[0].distance

        // Remove from heap (move last to root, heapify down)
        candidates.items[0] = candidates.items[candidates.size - 1]
        candidates.size--

        // Heapify down (standard max-heap)
        idx = 0
        WHILE idx < candidates.size:
            left = 2 * idx + 1
            right = 2 * idx + 2
            largest = idx

            IF left < candidates.size AND candidates.items[left].distance > candidates.items[largest].distance THEN
                largest = left
            END IF
            IF right < candidates.size AND candidates.items[right].distance > candidates.items[largest].distance THEN
                largest = right
            END IF

            IF largest == idx THEN BREAK

            SWAP candidates.items[idx] WITH candidates.items[largest]
            idx = largest
        END WHILE

        // ---------------------------------------------------------
        // Early termination check
        // ---------------------------------------------------------

        IF results.size >= ef AND -current_neg_dist > results.items[0].distance THEN
            BREAK  // All remaining candidates are farther than worst result
        END IF

        // ---------------------------------------------------------
        // Fetch current node's vector
        // ---------------------------------------------------------

        current_vector = NULL
        current_vector_size = 0
        current_level = 0

        rc = hnsw_fetch_node_data(stmts, current_rowid, &current_level, &current_vector, &current_vector_size)
        // SQLite IO: 1 SELECT from nodes table (unless in cache)
        // Memory: Allocates current_vector = malloc(current_vector_size)

        IF rc != SQLITE_OK THEN
            IF current_vector != NULL THEN FREE current_vector
            CONTINUE  // Skip on error
        END IF

        // ---------------------------------------------------------
        // Fetch current node's neighbors at this level
        // ---------------------------------------------------------

        neighbors = NULL
        neighbor_count = 0

        rc = hnsw_fetch_neighbors(stmts, current_rowid, level, &neighbors, &neighbor_count)
        // See hnsw_fetch_neighbors() pseudo-code below
        // SQLite IO: 1 SELECT query that fetches multiple rows (unless in cache)
        // Memory: Allocates neighbors = malloc(neighbor_count * sizeof(i64))

        IF rc != SQLITE_OK THEN
            FREE current_vector
            CONTINUE  // Skip on error
        END IF

        // ---------------------------------------------------------
        // Process each neighbor
        // ---------------------------------------------------------

        FOR i FROM 0 TO neighbor_count - 1:
            neighbor_rowid = neighbors[i]

            // Check if already visited (O(1) hash lookup)
            IF hnsw_visited_contains(&visited, neighbor_rowid) THEN
                CONTINUE
            END IF

            // Mark as visited
            CALL hnsw_visited_add(&visited, neighbor_rowid)

            // Fetch neighbor's vector
            neighbor_vector_ptr = NULL
            neighbor_vector_size = 0
            neighbor_level = 0

            rc = hnsw_fetch_node_data(stmts, neighbor_rowid, &neighbor_level, &neighbor_vector_ptr, &neighbor_vector_size)
            // SQLite IO: 1 SELECT from nodes table (unless in cache)
            // Memory: Allocates neighbor_vector_ptr = malloc(neighbor_vector_size)

            IF rc != SQLITE_OK THEN CONTINUE

            // Calculate distance
            neighbor_distance = hnsw_calc_distance(meta, query_vector, neighbor_vector_ptr)

            // Free vector immediately
            FREE neighbor_vector_ptr

            // Add to queues if it improves results
            IF results.size < ef OR neighbor_distance < results.items[0].distance THEN
                CALL hnsw_queue_push(&candidates, neighbor_rowid, -neighbor_distance)
                CALL hnsw_queue_push(&results, neighbor_rowid, neighbor_distance)

                // If results heap exceeds ef, remove worst
                IF results.size > ef THEN
                    // Pop root (max element) from results heap
                    results.items[0] = results.items[results.size - 1]
                    results.size--
                    // Heapify down (same as above)
                END IF
            END IF
        END FOR

        // Free current iteration's allocations
        FREE current_vector
        FREE neighbors

        // SQLite IO per iteration:
        //   - 1 SELECT for current node vector (unless cached)
        //   - 1 SELECT for current node neighbors (unless cached)
        //   - neighbor_count SELECTs for neighbor vectors (unless cached)
        // Total per iteration: 2 + neighbor_count SELECTs (typical M=32, so ~34 SELECTs)

    END WHILE

    // ============================================================
    // EXTRACT RESULTS
    // ============================================================

    *out_count = results.size
    ALLOCATE *out_rowids = malloc(results.size * sizeof(i64))
    ALLOCATE *out_distances = malloc(results.size * sizeof(f32))

    IF *out_rowids == NULL OR *out_distances == NULL THEN
        CLEANUP and RETURN SQLITE_NOMEM
    END IF

    // Copy from heap to output arrays
    FOR i FROM 0 TO results.size - 1:
        (*out_rowids)[i] = results.items[i].rowid
        (*out_distances)[i] = results.items[i].distance
    END FOR

    // ============================================================
    // CLEANUP
    // ============================================================

    CALL hnsw_visited_cleanup(&visited)
    // Frees: visited.rowids, visited.slots_used

    CALL hnsw_queue_cleanup(&candidates)
    // Frees: candidates.items

    CALL hnsw_queue_cleanup(&results)
    // Frees: results.items

    RETURN SQLITE_OK

    // CRITICAL PERFORMANCE NOTES:
    // 1. This function makes MANY SQLite queries (potentially hundreds per insert)
    // 2. Each query goes through prepared statement -> SQLite page cache
    // 3. The HnswNodeCache (if enabled) adds overhead that slows this down 24-42%
    // 4. SQLite's page cache is what makes this fast - recently accessed pages stay in memory
    // 5. The prepared statements avoid SQL parsing overhead
END FUNCTION


## Helper Function: hnsw_fetch_node_data()

```
FUNCTION hnsw_fetch_node_data(stmts, rowid, out_level, out_vector, out_vector_size):

    // ============================================================
    // TRY CACHE FIRST (if enabled)
    // ============================================================

    IF stmts->node_cache != NULL THEN
        cached_vector = NULL
        cached_size = 0
        cached_level = 0

        found = hnsw_cache_lookup_node(stmts->node_cache, rowid, &cached_level, &cached_vector, &cached_size)
        // O(1) hash table lookup in cache
        // Returns pointer to cached data (ZERO-COPY)

        IF found THEN
            // Cache hit - make a copy for caller
            *out_level = cached_level
            *out_vector_size = cached_size

            IF out_vector != NULL AND cached_vector != NULL AND cached_size > 0 THEN
                ALLOCATE vector_copy = malloc(cached_size)
                IF vector_copy == NULL THEN RETURN SQLITE_NOMEM
                memcpy(vector_copy, cached_vector, cached_size)
                *out_vector = vector_copy
                // Caller must FREE this
            END IF

            RETURN SQLITE_OK  // Cache hit - no SQLite IO
        END IF
    END IF

    // ============================================================
    // CACHE MISS - QUERY SQLITE
    // ============================================================

    stmt = stmts->get_node_data
    // Prepared statement: SELECT rowid, level, vector FROM nodes WHERE rowid=?

    retries = 0
    max_retries = 5

    WHILE retries < max_retries:
        CALL sqlite3_reset(stmt)
        // Resets statement but keeps it prepared

        CALL sqlite3_bind_int64(stmt, 1, rowid)
        // Bind parameter

        rc = sqlite3_step(stmt)
        // CRITICAL: This is where SQLite IO happens
        // - SQLite checks page cache first
        // - If page not in cache, reads from disk/WAL
        // - Page stays in cache (LRU eviction policy)

        IF rc == SQLITE_ROW THEN
            // Extract results
            IF out_level != NULL THEN
                *out_level = sqlite3_column_int(stmt, 1)
            END IF

            // Get blob pointer (points into SQLite's page cache)
            blob_ptr = sqlite3_column_blob(stmt, 2)
            blob_size = sqlite3_column_bytes(stmt, 2)
            level_val = sqlite3_column_int(stmt, 1)

            IF out_level != NULL THEN
                *out_level = level_val
            END IF

            // Insert into cache BEFORE copying (if cache enabled)
            IF stmts->node_cache != NULL AND blob_ptr != NULL AND blob_size > 0 THEN
                hnsw_cache_insert_node(stmts->node_cache, rowid, level_val, blob_ptr, blob_size)
                // Makes a copy into cache's arena allocator
            END IF

            // Make copy for caller BEFORE reset
            // (blob_ptr becomes invalid after reset)
            vector_copy = NULL
            IF out_vector != NULL AND out_vector_size != NULL AND blob_ptr != NULL AND blob_size > 0 THEN
                ALLOCATE vector_copy = malloc(blob_size)
                IF vector_copy == NULL THEN
                    CALL sqlite3_reset(stmt)
                    RETURN SQLITE_NOMEM
                END IF
                memcpy(vector_copy, blob_ptr, blob_size)
                *out_vector = vector_copy
                *out_vector_size = blob_size
                // Caller must FREE vector_copy
            END IF

            // Reset statement immediately to release WAL read lock
            CALL sqlite3_reset(stmt)

            RETURN SQLITE_OK

        ELSE IF rc == SQLITE_DONE THEN
            // No row found
            CALL sqlite3_reset(stmt)
            *out_vector = NULL
            *out_vector_size = 0
            RETURN SQLITE_OK

        ELSE IF (rc == SQLITE_BUSY OR rc == SQLITE_BUSY_SNAPSHOT) AND retries < max_retries - 1 THEN
            // Retry with exponential backoff
            retries++
            random_val = random()
            sleep_ms = (random_val % (retries * 100)) + 1
            CALL sqlite3_sleep(sleep_ms)
            CONTINUE

        ELSE
            // Error
            CALL sqlite3_reset(stmt)
            RETURN rc
        END IF
    END WHILE

    CALL sqlite3_reset(stmt)
    RETURN rc

    // MEMORY: Allocates vector_copy = malloc(blob_size), caller must FREE
    // SQLite IO: 1 SELECT (unless cached by SQLite page cache)
END FUNCTION


## Helper Function: hnsw_fetch_neighbors()

```
FUNCTION hnsw_fetch_neighbors(stmts, rowid, level, out_neighbors, out_count):

    // ============================================================
    // TRY CACHE FIRST (if enabled)
    // ============================================================

    IF stmts->node_cache != NULL THEN
        cached_neighbors = NULL
        cached_count = 0

        found = hnsw_cache_lookup_neighbors(stmts->node_cache, rowid, level, &cached_neighbors, &cached_count)
        // O(1) hash table lookup in cache
        // Returns pointer to cached data (ZERO-COPY)

        IF found THEN
            // Cache hit - make a copy for caller
            *out_count = cached_count

            IF out_neighbors != NULL AND cached_neighbors != NULL AND cached_count > 0 THEN
                size = cached_count * sizeof(i64)
                ALLOCATE neighbors_copy = malloc(size)
                IF neighbors_copy == NULL THEN RETURN SQLITE_NOMEM
                memcpy(neighbors_copy, cached_neighbors, size)
                *out_neighbors = neighbors_copy
                // Caller must FREE this
            END IF

            RETURN SQLITE_OK  // Cache hit - no SQLite IO
        END IF
    END IF

    // ============================================================
    // CACHE MISS - QUERY SQLITE
    // ============================================================

    stmt = stmts->get_edges
    // Prepared statement: SELECT to_rowid FROM edges WHERE from_rowid=? AND level=?

    count = 0
    capacity = 16
    ALLOCATE neighbors = malloc(capacity * sizeof(i64))
    IF neighbors == NULL THEN RETURN SQLITE_NOMEM

    retries = 0
    max_retries = 5

    WHILE retries < max_retries:
        CALL sqlite3_reset(stmt)
        CALL sqlite3_bind_int64(stmt, 1, rowid)
        CALL sqlite3_bind_int(stmt, 2, level)
        count = 0  // Reset on retry

        // Fetch all rows
        WHILE (rc = sqlite3_step(stmt)) == SQLITE_ROW:
            // Grow array if needed
            IF count >= capacity THEN
                capacity *= 2
                REALLOCATE neighbors to capacity * sizeof(i64)
                IF realloc failed THEN
                    FREE neighbors
                    RETURN SQLITE_NOMEM
                END IF
            END IF

            neighbors[count++] = sqlite3_column_int64(stmt, 0)
        END WHILE

        IF rc == SQLITE_DONE THEN
            // Success - got all rows
            CALL sqlite3_reset(stmt)
            *out_neighbors = neighbors
            *out_count = count

            // Insert into cache for future lookups
            IF stmts->node_cache != NULL THEN
                hnsw_cache_insert_neighbors(stmts->node_cache, rowid, level, neighbors, count)
                // Makes a copy into cache's arena allocator
            END IF

            RETURN SQLITE_OK

        ELSE IF (rc == SQLITE_BUSY OR rc == SQLITE_BUSY_SNAPSHOT) AND retries < max_retries - 1 THEN
            // Retry with exponential backoff
            retries++
            random_val = random()
            sleep_ms = (random_val % (retries * 100)) + 1
            CALL sqlite3_sleep(sleep_ms)
            CONTINUE

        ELSE
            // Error
            CALL sqlite3_reset(stmt)
            FREE neighbors
            RETURN rc
        END IF
    END WHILE

    CALL sqlite3_reset(stmt)
    FREE neighbors
    RETURN rc

    // MEMORY: Allocates neighbors array (dynamically grown), caller must FREE
    // SQLite IO: 1 SELECT that returns multiple rows (typical M=32 rows)
END FUNCTION


## Helper Function: hnsw_prune_neighbor_connections()

```
FUNCTION hnsw_prune_neighbor_connections(db, stmts, meta, neighbor_rowid, new_node_rowid, level, max_connections):

    // This function ensures neighbor doesn't exceed max_connections
    // Uses RNG heuristic to select diverse set of neighbors

    // ============================================================
    // FETCH NEIGHBOR'S EXISTING EDGES
    // ============================================================

    existing_neighbors = NULL
    existing_count = 0

    rc = hnsw_fetch_neighbors(stmts, neighbor_rowid, level, &existing_neighbors, &existing_count)
    // SQLite IO: 1 SELECT (unless cached)
    // Memory: Allocates existing_neighbors array

    IF rc != SQLITE_OK THEN RETURN rc

    // Check if pruning needed
    IF existing_count <= max_connections THEN
        FREE existing_neighbors
        RETURN SQLITE_OK  // No pruning needed
    END IF

    // ============================================================
    // FETCH NEIGHBOR'S VECTOR (center point)
    // ============================================================

    neighbor_vector = NULL
    neighbor_size = 0
    neighbor_level = 0

    rc = hnsw_fetch_node_data(stmts, neighbor_rowid, &neighbor_level, &neighbor_vector, &neighbor_size)
    // SQLite IO: 1 SELECT (unless cached)
    // Memory: Allocates neighbor_vector

    IF rc != SQLITE_OK OR neighbor_vector == NULL THEN
        FREE existing_neighbors
        RETURN rc
    END IF

    // ============================================================
    // BUILD CANDIDATE POOL
    // ============================================================

    pool_size = existing_count + 1  // Existing + new node
    ALLOCATE candidate_pool = malloc(pool_size * sizeof(i64))
    ALLOCATE candidate_distances = malloc(pool_size * sizeof(f32))

    // Copy existing neighbors to pool
    memcpy(candidate_pool, existing_neighbors, existing_count * sizeof(i64))
    FREE existing_neighbors  // Done with this

    // Add new node to pool
    candidate_pool[pool_size - 1] = new_node_rowid

    // ============================================================
    // COMPUTE DISTANCES TO CENTER (neighbor)
    // ============================================================

    FOR i FROM 0 TO pool_size - 1:
        cand_vector = NULL
        cand_size = 0
        cand_level = 0

        rc = hnsw_fetch_node_data(stmts, candidate_pool[i], &cand_level, &cand_vector, &cand_size)
        // SQLite IO: 1 SELECT per candidate (unless cached)
        // Memory: Allocates cand_vector

        IF rc == SQLITE_OK AND cand_vector != NULL THEN
            candidate_distances[i] = hnsw_calc_distance(meta, neighbor_vector, cand_vector)
            FREE cand_vector
        ELSE
            candidate_distances[i] = FLT_MAX
        END IF
    END FOR

    FREE neighbor_vector  // Done with center vector

    // SQLite IO so far: existing_count + 1 SELECTs

    // ============================================================
    // SORT BY DISTANCE TO CENTER
    // ============================================================

    // Simple bubble sort (pool_size is small, typically ~33)
    FOR i FROM 0 TO pool_size - 2:
        FOR j FROM i + 1 TO pool_size - 1:
            IF candidate_distances[j] < candidate_distances[i] THEN
                SWAP candidate_distances[i] WITH candidate_distances[j]
                SWAP candidate_pool[i] WITH candidate_pool[j]
            END IF
        END FOR
    END FOR

    // ============================================================
    // APPLY RNG HEURISTIC
    // ============================================================

    selected = NULL
    selected_count = 0

    rc = hnsw_select_neighbors_by_heuristic(candidate_pool, candidate_distances, pool_size,
                                             neighbor_vector, max_connections,
                                             stmts, meta, &selected, &selected_count)

    // This function:
    //   1. Processes candidates from closest to farthest
    //   2. For each candidate, checks if adding it maintains "small world" property
    //   3. Discards candidates that are too close to already-selected ones
    //   4. May fetch more vectors for distance comparisons
    //   5. Returns selected = malloc(selected_count * sizeof(i64))

    FREE candidate_pool
    FREE candidate_distances

    IF rc != SQLITE_OK THEN
        IF selected != NULL THEN FREE selected
        RETURN rc
    END IF

    // ============================================================
    // DELETE ALL EXISTING EDGES
    // ============================================================

    CALL sqlite3_reset(stmts->delete_edges_from)
    CALL sqlite3_bind_int64(stmts->delete_edges_from, 1, neighbor_rowid)
    CALL sqlite3_bind_int(stmts->delete_edges_from, 2, level)
    rc = sqlite3_step(stmts->delete_edges_from)
    CALL sqlite3_reset(stmts->delete_edges_from)

    // SQLite IO: 1 DELETE (removes all edges at this level)

    IF rc != SQLITE_DONE THEN
        FREE selected
        RETURN SQLITE_ERROR
    END IF

    // ============================================================
    // RE-INSERT SELECTED EDGES
    // ============================================================

    FOR i FROM 0 TO selected_count - 1:
        CALL sqlite3_reset(stmts->insert_edge)
        CALL sqlite3_bind_int64(stmts->insert_edge, 1, neighbor_rowid)
        CALL sqlite3_bind_int64(stmts->insert_edge, 2, selected[i])
        CALL sqlite3_bind_int(stmts->insert_edge, 3, level)
        rc = sqlite3_step(stmts->insert_edge)
        CALL sqlite3_reset(stmts->insert_edge)

        // SQLite IO: 1 INSERT per selected neighbor

        IF rc != SQLITE_OK THEN
            FREE selected
            RETURN rc
        END IF
    END FOR

    // ============================================================
    // INVALIDATE CACHE FOR THIS NEIGHBOR
    // ============================================================

    IF stmts->node_cache != NULL THEN
        hnsw_cache_invalidate_neighbors(stmts->node_cache, neighbor_rowid, level)
        // Removes this (rowid, level) from neighbor cache
    END IF

    FREE selected
    RETURN SQLITE_OK

    // TOTAL SQLite IO:
    //   - 1 SELECT for existing neighbors (unless cached)
    //   - 1 SELECT for neighbor's vector (unless cached)
    //   - pool_size SELECTs for candidate vectors (unless cached)
    //   - Variable SELECTs in heuristic (depends on algorithm)
    //   - 1 DELETE for all edges
    //   - selected_count INSERTs for new edges
    // Typical: ~40 SELECTs + 1 DELETE + 32 INSERTs
END FUNCTION


## Cache Behavior (DISABLED BY DEFAULT)

The C implementation has `SQLITE_VEC_DISABLE_HNSW_CACHE` defined by default because:

1. **Cache is 24-42% SLOWER than no cache**
2. **SQLite's page cache already does this job well**
3. **Prepared statements avoid SQL parsing overhead**

When cache IS enabled:
- `HnswNodeCache` uses arena allocator for fast memory management
- Hash table with open addressing (linear probing)
- LRU eviction when cache is full
- Version tracking for multi-connection safety
- Zero-copy lookups (returns pointers to cached data)

But the overhead of checking cache + copying data + managing eviction is slower than just letting SQLite handle it with its highly optimized page cache.

## Critical Performance Factors

1. **Prepared Statements**: Avoid SQL parsing overhead (~90% of C code benefit)
2. **SQLite Page Cache**: Recently accessed pages stay in memory (most important!)
3. **Immediate sqlite3_reset()**: Releases WAL locks quickly
4. **Minimal memory copies**: Only copy when needed for caller
5. **Efficient data structures**: Max-heaps, hash tables for O(1)/O(log n) operations
6. **SIMD distance calculations**: AVX512/AVX2/SSE/NEON when available

## Total SQLite IO for Typical Insert (100k vectors, M=32, ef=400)

Assuming level=2 (3 levels: 0, 1, 2), inserting into established index:

- Phase 1: 1 INSERT (node)
- Phase 2: Skip (not first node)
- Phase 3: 1 INSERT OR REPLACE (num_nodes)
- Phase 4: 2 search_layer calls (levels 3→2, 2→1)
  - Each search_layer: ~200 SELECTs (ef=400, but early termination)
  - Total: ~400 SELECTs
- Phase 5: 3 levels (2, 1, 0)
  - Level 2: search_layer (~200 SELECTs) + 2*32 INSERTs + 32 prunes (~1000 SELECTs total)
  - Level 1: search_layer (~200 SELECTs) + 2*32 INSERTs + 32 prunes (~1000 SELECTs total)
  - Level 0: search_layer (~200 SELECTs) + 2*64 INSERTs + 64 prunes (~2000 SELECTs total)
  - Total: ~4200 SELECTs + 256 INSERTs
- Phase 7: 1 SELECT + 1 INSERT OR REPLACE (hnsw_version)

**Grand total: ~4601 SELECTs + 257 INSERTs + 1 DELETE per insert**

But thanks to SQLite's page cache, most of these SELECTs hit cached pages!

## Key Insight

The C implementation is fast because:
1. It uses prepared statements (no SQL parsing)
2. It relies on SQLite's page cache (not custom cache)
3. It minimizes memory copies
4. It uses efficient algorithms (max-heaps, hash tables)

The Rust implementation should do the SAME - don't try to outsmart SQLite's page cache!
