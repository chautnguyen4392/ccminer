# Why Batching is Used in titan_scrypt_core_kernelA_LG

> **Note:** The production kernel now always starts from iteration `0` and
> processes a full range specified by an `iterations` parameter. The material
> below documents the legacy batching/resume approach for historical reference.

## Overview

The `titan_scrypt_core_kernelA_LG` kernel processes iterations in **batches** (using `begin` and `end` parameters) instead of processing all N iterations continuously in a single kernel launch. This document explains why this design choice was made.

## The Batching Implementation

```cuda
template <int ALGO, MemoryAccess SCHEME> __global__
void titan_scrypt_core_kernelA_LG(const uint32_t *d_idata, int begin, int end, unsigned int LOOKUP_GAP)
{
    // ... setup code ...
    
    int i = begin;  // Start from 'begin', not 0
    
    // Handle resume from previous batch
    if (i == 0) {
        load_key<ALGO>(d_idata, b, bx);
        write_keys_direct<SCHEME>(b, bx, start);
        ++i;
    } else {
        // Resume from stored state
        int pos = (i-1)/LOOKUP_GAP, loop = (i-1)-pos*LOOKUP_GAP;
        read_keys_direct<SCHEME>(b, bx, start+32*pos);
        while(loop--) block_mixer<ALGO>(b, bx, x1, x2, x3);
    }
    
    // Process batch [begin, end)
    while (i < end) {
        block_mixer<ALGO>(b, bx, x1, x2, x3);
        if (i % LOOKUP_GAP == 0)
            write_keys_direct<SCHEME>(b, bx, start+32*(i/LOOKUP_GAP));
        ++i;
    }
}
```

**Host-side batching loop:**
```cuda
int batch = device_batchsize[thr_id];  // Default: 1024
unsigned int pos = 0;

do {
    titan_scrypt_core_kernelA_LG<...> <<< grid, threads, 0, stream >>>(
        d_idata, pos, min(pos+batch, N), LOOKUP_GAP);
    pos += batch;
} while (pos < N);
```

## Reasons for Batching

### 1. **Kernel Timeout Prevention (Windows)**

**Problem:**
- Windows has a **5-10 second timeout** for GPU kernel execution
- If a kernel runs longer than this, Windows will reset the GPU (TDR - Timeout Detection and Recovery)
- For large N values (e.g., N=4,194,304 with Nfactor=21), a single kernel could easily exceed this timeout

**Solution:**
- By processing in batches of 1024 iterations, each kernel launch completes quickly
- Example: With N=4,194,304, there are 4,096 batches (4,194,304 / 1024)
- Each batch processes only 1024 iterations, well within timeout limits

### 2. **System Responsiveness**

**Problem:**
- A single long-running kernel can make the system unresponsive
- The GPU is locked during kernel execution
- User interaction and other GPU operations are blocked

**Solution:**
- Batching allows the system to:
  - Process other GPU work between batches
  - Respond to user input
  - Handle system events
  - Allow other applications to use the GPU

### 3. **Memory Pressure Management**

**Problem:**
- Very large N values require massive scratchpad memory
- With LOOKUP_GAP=32 and N=4,194,304:
  - Scratchpad size = (N/32) × 128 bytes = 16 MB per hash
  - For multiple concurrent hashes, this can be hundreds of MB
- Processing all iterations at once can cause:
  - Memory fragmentation
  - Allocation failures
  - Out-of-memory errors

**Solution:**
- Batching allows:
  - Better memory management
  - Gradual memory allocation
  - Memory cleanup between batches if needed
  - More predictable memory usage patterns

### 4. **Progress Reporting and Monitoring**

**Problem:**
- A single long-running kernel provides no progress feedback
- Difficult to estimate completion time
- Hard to detect if kernel is stuck or making progress

**Solution:**
- Batching enables:
  - Progress tracking (e.g., "Processing batch 500 of 4096")
  - Performance monitoring per batch
  - Early detection of performance issues
  - Better debugging and profiling

### 5. **Error Recovery and Robustness**

**Problem:**
- If a single kernel fails or crashes, all progress is lost
- No way to resume from a checkpoint
- Must restart entire computation

**Solution:**
- Batching provides:
  - Checkpointing between batches (state stored in scratchpad)
  - Ability to resume from last successful batch
  - Better error isolation (one batch failure doesn't affect others)
  - Easier debugging (identify which batch failed)

### 6. **Register Pressure and Resource Management**

**Problem:**
- Very long loops can cause:
  - High register usage
  - Register spilling to local memory (slow)
  - Reduced occupancy (fewer concurrent threads)

**Solution:**
- Shorter batch loops:
  - Lower register pressure
  - Better register allocation
  - Higher GPU occupancy
  - Better performance

### 7. **Resume Logic for Continuity**

The kernel includes special logic to handle resuming from a previous batch:

```cuda
if (i == 0) {
    // First batch: Load initial key from input
    load_key<ALGO>(d_idata, b, bx);
    write_keys_direct<SCHEME>(b, bx, start);
    ++i;
} else {
    // Resume batch: Reconstruct state from scratchpad
    int pos = (i-1)/LOOKUP_GAP, loop = (i-1)-pos*LOOKUP_GAP;
    read_keys_direct<SCHEME>(b, bx, start+32*pos);
    while(loop--) block_mixer<ALGO>(b, bx, x1, x2, x3);
}
```

This ensures:
- **Continuity**: Each batch continues exactly where the previous batch left off
- **Correctness**: The computation is identical to processing all iterations at once
- **State preservation**: Uses LOOKUP_GAP storage to reconstruct intermediate state

## Example: N=4,194,304 with LOOKUP_GAP=32, batch=1024

**Total iterations:** 4,194,304  
**Batch size:** 1024  
**Number of batches:** 4,194,304 / 1024 = 4,096 batches

**Execution pattern:**
```
Batch 0:   Process iterations [0, 1024)
Batch 1:   Process iterations [1024, 2048)  (resumes from iteration 1024)
Batch 2:   Process iterations [2048, 3072)  (resumes from iteration 2048)
...
Batch 4095: Process iterations [4193280, 4194304)  (final batch)
```

**Storage pattern (with LOOKUP_GAP=32):**
- Stores every 32nd iteration in scratchpad
- Batch 0 stores: iterations 0, 32, 64, ..., 992 (32 values)
- Batch 1 stores: iterations 1024, 1056, 1088, ..., 2016 (32 values)
- Each batch stores 32 values (1024 / 32 = 32)

## Performance Impact

**Overhead:**
- Minimal: Each batch launch has ~microsecond overhead
- Total overhead: 4,096 batches × ~1μs = ~4ms (negligible for 4M iterations)

**Benefits:**
- Prevents timeouts (critical for Windows)
- Better system responsiveness
- More predictable performance
- Easier debugging and monitoring

## Comparison: Continuous vs Batched

| Aspect | Continuous (0 to N) | Batched (1024 iterations) |
|-------|-------------------|---------------------------|
| **Kernel timeout** | ❌ Risk of timeout | ✅ Safe |
| **System responsiveness** | ❌ Blocked | ✅ Responsive |
| **Progress tracking** | ❌ None | ✅ Per batch |
| **Error recovery** | ❌ All-or-nothing | ✅ Per batch |
| **Memory management** | ❌ All at once | ✅ Gradual |
| **Register pressure** | ⚠️ Higher | ✅ Lower |
| **Code complexity** | ✅ Simpler | ⚠️ More complex |

## Conclusion

Batching is **essential** for:
1. **Windows compatibility** (timeout prevention)
2. **System responsiveness**
3. **Robustness and error recovery**
4. **Better resource management**

The small code complexity increase is well worth these benefits, especially for large N values common in scrypt-jane mining (Nfactor 18-22, resulting in N = 2^19 to 2^23 = 524,288 to 8,388,608).

