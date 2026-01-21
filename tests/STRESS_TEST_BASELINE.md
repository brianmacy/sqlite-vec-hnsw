# Multi-threaded Stress Test Baseline

**Date:** 2025-01-21
**Test:** `cargo test test_multithread_long_running --release -- --ignored --nocapture`

## Configuration

| Parameter | Value |
|-----------|-------|
| Duration | 60 seconds |
| Insert Threads | 16 |
| Search Threads | 4 |
| Vector Dimensions | 128 |
| k (neighbors) | 50 |
| Search Sleep | 10ms between queries |
| SQLite Mode | WAL |

## Baseline Results

```
============================================================
                    BASELINE RESULTS
============================================================
Duration:        60.34s
Total Inserts:   8404 (0 errors)
Total Searches:  13581 (0 errors)
Insert Rate:     139 vec/sec
Search Rate:     225 queries/sec
------------------------------------------------------------
Vectors in DB:   8404
HNSW Nodes:      8404
HNSW Edges:      309164
Integrity:       PASS
============================================================
```

## Summary

| Metric | Value |
|--------|-------|
| Insert Rate | 139 vec/sec |
| Search Rate | 225 queries/sec |
| Search:Insert Ratio | 1.6:1 |
| Total Operations | 21,985 |
| Error Rate | 0% |
| Data Integrity | PASS |

## Notes

- 16 insert threads competing for SQLite write lock
- 4 search threads with 10ms sleep between queries to allow fair lock acquisition
- WAL mode enables concurrent readers with single writer
- All vectors correctly indexed in HNSW graph (nodes = vectors)
