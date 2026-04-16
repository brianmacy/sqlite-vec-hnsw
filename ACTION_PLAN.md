# Action Plan: Match C Implementation Exactly

## Current Status

**Performance:**
- Early (100 nodes): 3.53 ms/vector → 1.45x slower ✅ Getting close!
- Late (1000 nodes): 8.02 ms/vector → 3.30x slower ❌ Degrading badly
- Search: 1.04 ms → Faster than C! ✅

**Issue:** 2.27x performance degradation as graph grows (C stays constant)

---

## Critical Differences Found

### 1. No SQLITE_BUSY Retry Logic
**C:** Every DB operation has retry with exponential backoff
**Rust:** Fail immediately on BUSY
**Fix:** Add retry logic to all storage operations

### 2. Pruning May Be Broken
**C:** Computes distances from candidates to center
**Rust:** Tries to use distances from DB (NULL) → broken sorting
**Fix:** Implement proper distance computation in pruning OR disable pruning temporarily to test

### 3. Extra Statement (get_edges_with_dist)
**C:** Only has get_edges (no distances)
**Rust:** Has both get_edges and get_edges_with_dist
**Impact:** Minor overhead preparing extra statement
**Fix:** Can remove or leave for now

---

## Immediate Actions

### Step 1: Verify Edge Counts (Diagnose Pruning)
Run test to check actual edge counts after 1000 inserts.

Expected (C with proper pruning):
- Avg edges per node: ~88
- Max edges per node: ≤128 (max_M0)

If we see:
- Max > 128: Pruning is broken
- Avg growing over time: Edges accumulating

### Step 2: Fix Pruning
C's approach: Fetch neighbor vectors, compute distances to center, apply heuristic

Our current approach: Fetch NULL distances, sort by NULL (broken!)

**Fix options:**
A. Implement C's full RNG heuristic (complex)
B. Implement simple greedy with computed distances (simpler)
C. Temporarily disable pruning to isolate issue

**Recommendation:** Try B first (greedy with computed distances)

### Step 3: Add Retry Logic
C retries on SQLITE_BUSY/SQLITE_BUSY_SNAPSHOT with exponential backoff.

Add to:
- insert_node
- insert_edge
- fetch operations
- delete operations

### Step 4: Test After Each Change
After each fix:
- Run profile_insert_timing test
- Check early vs late performance
- Verify degradation is fixed

---

## Implementation Order

1. ✅ Test edge counts to confirm pruning issue
2. ✅ Fix pruning to compute distances (not fetch NULL)
3. ✅ Test - verify degradation reduced
4. ✅ Add SQLITE_BUSY retry logic
5. ✅ Test - verify performance matches C
6. ✅ Clean up extra statements if needed

---

## Expected Results After Fixes

If pruning is the issue:
- Edge counts will be controlled (≤128 max)
- Late inserts will speed up (no bloated graph)
- Performance will stay constant like C

Target: **< 3ms per vector constant** (match C's 2.43ms)
