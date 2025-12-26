# Scrypt-Jane GPU Kernel Execution Details
## Main Mining Loop and GPU Kernel Flow for Nfactor=21 on GTX 1070

This document provides detailed analysis of the main mining loop and GPU kernel execution for scrypt-jane algorithm with Nfactor=21 on a GeForce GTX 1070.

---

## 1. The `parallel` Variable

### Definition and Purpose

```c
// ccminer.cpp:153
int parallel = 2; // All should be made on GPU
```

**Meaning:**
- `parallel = 2`: **Full GPU mode** - All operations (PBKDF2 + ROMix) run on GPU
- `parallel < 2`: **Hybrid mode** - PBKDF2 runs on CPU, only ROMix runs on GPU
- `parallel = 1`: **Half CPU mode** - Some operations split between CPU and GPU

**Why Hardcoded to 2:**
- Modern GPUs (Pascal and newer) have sufficient compute power and memory bandwidth
- GPU-accelerated PBKDF2 (Keccak-512) is faster than CPU implementation
- Reduces CPU-GPU data transfers
- Better overall performance for scrypt-jane

---

## 2. Main Mining Loop Architecture

### Double-Buffering Design

The mining loop uses **double-buffering** to overlap computation and memory transfers:

```c
// scrypt-jane.cpp:637-638
int cur = 0, nxt = 1;  // Current and next buffer indices
int iteration = 0;     // Iteration counter
```

**Buffer Structure:**
- **Two streams** (0 and 1) for asynchronous execution
- **Two data buffers** (`data[0]` and `data[1]`)
- **Two hash buffers** (`hash[0]` and `hash[1]`)
- **Two X buffers** (`Xbuf[0]` and `Xbuf[1]`)

**Execution Pattern:**
```
Iteration N:   Process buffer[cur],  Launch buffer[nxt]
Iteration N+1: Process buffer[nxt], Launch buffer[cur]
```

This allows:
- GPU processing buffer `cur` while preparing buffer `nxt`
- Overlapping memory transfers with computation
- Maximizing GPU utilization

---

## 3. Execution Path: `parallel == 2` (Full GPU Mode)

### Code Flow (scrypt-jane.cpp:706-765)

When `parallel == 2`, the entire scrypt-jane computation runs on GPU:

```c
// Line 706-765: Full GPU execution path
if (parallel == 2) {
    // All operations on GPU
    
    // 1. Update nonce for next batch
    n += throughput;
    
    // 2. Serialize execution (prevent kernel overlap)
    cuda_scrypt_serialize(thr_id, nxt);
    
    // 3. Pre-Keccak512: PBKDF2 on GPU
    pre_keccak512(thr_id, nxt, nonce[nxt], throughput, block_header_size);
    
    // 4. ROMix: Sequential writes + random reads
    cuda_scrypt_core(thr_id, nxt, N);
    
    // 5. Synchronize before post-processing
    if (!cuda_scrypt_sync(thr_id, nxt)) {
        break;
    }
    
    // 6. Post-Keccak512: Final PBKDF2 on GPU
    post_keccak512(thr_id, nxt, nonce[nxt], throughput, block_header_size);
    
    // 7. Mark stream as done
    cuda_scrypt_done(thr_id, nxt);
    
    // 8. Transfer results back to host
    cuda_scrypt_DtoH(thr_id, hash[nxt], nxt, true);
    
    // 9. Final synchronization
    if (!cuda_scrypt_sync(thr_id, nxt)) {
        break;
    }
}
```

### Step-by-Step GPU Execution

#### Step 1: Serialization (`cuda_scrypt_serialize`)

```c
// scrypt/salsa_kernel.cu:772-778
void cuda_scrypt_serialize(int thr_id, int stream)
{
    // Wait for other stream to finish if device supports concurrent kernels
    if (context_concurrent[thr_id] || device_interactive[thr_id])
        cudaStreamWaitEvent(context_streams[stream][thr_id], 
                           context_serialize[(stream+1)&1][thr_id], 0);
}
```

**Purpose:** Prevents kernel overlap on devices that don't support true concurrency, ensuring sequential execution of ROMix operations.

#### Step 2: Pre-Keccak512 (`pre_keccak512`)

**Function:** `pre_keccak512(thr_id, stream, nonce, throughput, block_header_size)`

**GPU Kernel:** `cuda_pre_keccak512` (keccak.cu:442-504)

**What it does:**
1. Each CUDA thread processes one work unit (nonce)
2. Loads block header data from constant memory (`c_data`)
3. Inserts thread-specific nonce
4. Performs PBKDF2 with Keccak-512 HMAC:
   - `HMAC(password=block_header, salt=block_header)`
   - Generates 128 bytes of output (2 × 64-byte Keccak-512 blocks)
5. Stores result in `context_idata[stream][thr_id]`

**Kernel Launch:**
```c
// keccak.cu:559-565
dim3 block(128);  // 128 threads per block
dim3 grid((throughput+127)/128);  // Number of blocks

cuda_pre_keccak512<<<grid, block, 0, context_streams[stream][thr_id]>>>(
    context_idata[stream][thr_id], nonce, keylen);
```

**For Nfactor=21:**
- `throughput` typically = 128-256 threads
- Each thread processes one nonce
- Output: 128 bytes per thread (scrypt input)

#### Step 3: ROMix Core (`cuda_scrypt_core`)

**Function:** `cuda_scrypt_core(thr_id, stream, N)`

**For GTX 1070 (Pascal SM 6.1):** Uses `TitanKernel`

**Kernel Selection:**
```c
// scrypt/salsa_kernel.cu:54-80
KernelInterface *Best_Kernel_Heuristics(cudaDeviceProp *props)
{
    uint64_t N = 1UL << (opt_nfactor+1);  // N = 2^22 = 4,194,304
    
    if (IS_SCRYPT_JANE() && N > 8192) {
        // High N-factor scrypt-jane = low register count kernels
        if (props->major > 3 || (props->major == 3 && props->minor >= 5))
            kernel = new TitanKernel();  // ← GTX 1070 uses this
    }
}
```

**TitanKernel Execution (titan_kernel.cu:718-748):**

**Phase 1: Sequential Writes (kernelA)**
```c
// Sequential writes to scratchpad
unsigned int pos = 0;
do {
    if (LOOKUP_GAP == 1) {
        titan_scrypt_core_kernelA<A_SCRYPT_JANE, SIMPLE>
            <<< grid, threads, 0, stream >>>(d_idata, pos, min(pos+batch, N));
    }
    pos += batch;
} while (pos < N);
```

**What happens:**
- For each position `i` from 0 to N-1:
  1. Read `X[i]` from input buffer
  2. Apply ChaCha20 mixing function (20 rounds)
  3. Write result to `V[i]` in scratchpad memory
- **Total iterations:** N = 4,194,304
- **Memory access:** Sequential writes (cache-friendly)
- **Batch processing:** Processes in chunks (default batch = 1024)

**Phase 2: Random Reads (kernelB)**
```c
// Random read access from scratchpad
pos = 0;
do {
    if (LOOKUP_GAP == 1) {
        titan_scrypt_core_kernelB<A_SCRYPT_JANE, SIMPLE>
            <<< grid, threads, 0, stream >>>(d_odata, pos, min(pos+batch, N));
    }
    pos += batch;
} while (pos < N);
```

**What happens:**
- For each position `i` from 0 to N-1:
  1. Calculate random index: `j = X[i] & (N-1)`
  2. Read `V[j]` from scratchpad (random access)
  3. XOR with `X[i]`: `X[i] = X[i] ⊕ V[j]`
  4. Apply ChaCha20 mixing function
  5. Write result to output buffer
- **Total iterations:** N = 4,194,304
- **Memory access:** Random reads (memory bandwidth bottleneck)
- **Scratchpad size:** N × 128 bytes = 536,870,912 bytes (~512 MB)

**ChaCha20 Mixing Function:**
- Used instead of Salsa20/8 for scrypt-jane
- 20 rounds of mixing (vs 8 for standard scrypt)
- More computationally intensive but provides better security

#### Step 4: Post-Keccak512 (`post_keccak512`)

**Function:** `post_keccak512(thr_id, stream, nonce, throughput, block_header_size)`

**GPU Kernel:** `cuda_post_keccak512` (keccak.cu:508-541)

**What it does:**
1. Each CUDA thread processes one work unit
2. Loads block header from constant memory
3. Performs final PBKDF2 with Keccak-512:
   - `HMAC(password=block_header, salt=ROMix_output)`
   - Generates 32-byte final hash
4. Stores result in `context_hash[stream][thr_id]`

**Output:** 32-byte hash per thread (final scrypt-jane hash)

#### Step 5: Result Transfer (`cuda_scrypt_DtoH`)

```c
// scrypt/salsa_kernel.cu:809-820
void cuda_scrypt_DtoH(int thr_id, uint32_t *X, int stream, bool postSHA)
{
    unsigned int mem_size = WU_PER_LAUNCH * sizeof(uint32_t) * 8;  // 32 bytes per hash
    
    if (postSHA) {
        // Transfer final hashes
        cudaMemcpyAsync(X, context_hash[stream][thr_id], mem_size, 
                       cudaMemcpyDeviceToHost, context_streams[stream][thr_id]);
    } else {
        // Transfer intermediate data
        cudaMemcpyAsync(X, context_odata[stream][thr_id], mem_size, 
                       cudaMemcpyDeviceToHost, context_streams[stream][thr_id]);
    }
}
```

**Purpose:** Asynchronously transfer computed hashes from GPU to CPU for validation.

---

## 4. Solution Checking and Validation

### Quick Check (scrypt-jane.cpp:767-813)

```c
// Line 767-813: Check for solutions
for (int i=0; iteration > 0 && i<throughput; i++)
{
    // Quick check: Compare highest 32 bits
    if (hash[cur][8*i+7] <= Htarg && fulltest(&hash[cur][8*i], ptarget))
    {
        // Potential solution found!
        
        // Validate on CPU using reference implementation
        uint32_t thash[8], tdata[(block_header_size / 4)];
        uint32_t tmp_nonce = nonce[cur] + i;
        
        // Reconstruct block header with found nonce
        for(int z=0; z<(block_header_size / 4 - 1); z++)
            tdata[z] = bswap_32x4(pdata[z]);
        tdata[(block_header_size / 4 - 1)] = bswap_32x4(tmp_nonce);
        
        // CPU computation for verification
        scrypt_pbkdf2_1((unsigned char *)tdata, block_header_size, 
                       (unsigned char *)tdata, block_header_size, 
                       Xbuf[cur].ptr + 128 * i, 128);
        scrypt_ROMix_1((scrypt_mix_word_t *)(Xbuf[cur].ptr + 128 * i), 
                      (scrypt_mix_word_t *)(Ybuf.ptr), 
                      (scrypt_mix_word_t *)(Vbuf.ptr), N);
        scrypt_pbkdf2_1((unsigned char *)tdata, block_header_size, 
                       Xbuf[cur].ptr + 128 * i, 128, 
                       (unsigned char *)thash, 32);
        
        // Compare GPU and CPU results
        if (memcmp(thash, &hash[cur][8*i], 32) == 0) {
            // Valid solution!
            work->nonces[0] = tmp_nonce;
            return 1;
        }
    }
}
```

**Validation Process:**
1. **Quick check:** `hash[7] <= Htarg` (fast rejection)
2. **Full check:** `fulltest(hash, target)` (complete comparison)
3. **CPU verification:** Recompute on CPU to ensure correctness
4. **Comparison:** GPU hash must match CPU hash exactly

---

## 5. Memory Layout and Data Flow

### GPU Memory Allocation (for Nfactor=21, N=4,194,304)

```
┌─────────────────────────────────────────────────────────┐
│ GPU Global Memory                                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ context_idata[0/1][thr_id]  (Input buffers)            │
│   Size: throughput × 128 bytes                          │
│   Purpose: Block headers + Pre-Keccak512 output          │
│                                                         │
│ context_odata[0/1][thr_id]  (Output buffers)           │
│   Size: throughput × 128 bytes                          │
│   Purpose: ROMix output                                 │
│                                                         │
│ context_hash[0/1][thr_id]   (Hash buffers)             │
│   Size: throughput × 32 bytes                           │
│   Purpose: Final hashes (Post-Keccak512 output)         │
│                                                         │
│ h_V[thr_id][0..N-1]         (Scratchpad)               │
│   Size: N × 128 bytes = 536,870,912 bytes (~512 MB)     │
│   Purpose: ROMix scratchpad (V array)                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
CPU → GPU:
  Block Header (84 bytes) → Constant Memory (c_data)
  
GPU Processing:
  ┌─────────────────────────────────────────┐
  │ 1. Pre-Keccak512                        │
  │    Block Header → 128 bytes (PBKDF2)    │
  │    → context_idata[stream]              │
  └─────────────────────────────────────────┘
           ↓
  ┌─────────────────────────────────────────┐
  │ 2. ROMix Phase A (Sequential Writes)    │
  │    context_idata → Scratchpad (V)       │
  │    N iterations: V[i] = ChaCha(X[i])    │
  └─────────────────────────────────────────┘
           ↓
  ┌─────────────────────────────────────────┐
  │ 3. ROMix Phase B (Random Reads)         │
  │    Scratchpad (V) → context_odata       │
  │    N iterations: X[i] = X[i] ⊕ V[j]     │
  └─────────────────────────────────────────┘
           ↓
  ┌─────────────────────────────────────────┐
  │ 4. Post-Keccak512                       │
  │    context_odata → 32 bytes (PBKDF2)    │
  │    → context_hash[stream]               │
  └─────────────────────────────────────────┘

GPU → CPU:
  context_hash[stream] → hash[stream] (32 bytes per thread)
```

---

## 6. Performance Characteristics (GTX 1070, Nfactor=21)

### Memory Bandwidth Analysis

**Scratchpad Access Pattern:**
- **Sequential writes:** ~512 MB written sequentially (cache-friendly)
- **Random reads:** ~512 MB read randomly (memory bandwidth limited)
- **Total memory traffic:** ~1 GB per hash computation

**GTX 1070 Specifications:**
- Memory: 8 GB GDDR5
- Memory Bandwidth: 256 GB/s
- Memory Clock: 8 Gbps
- CUDA Cores: 1920
- Memory Bus: 256-bit

**Bottleneck:**
- Random memory access in ROMix Phase B is the primary bottleneck
- Memory bandwidth utilization: ~60-80% (depending on intensity)

### Computational Load

**Per Hash Computation:**
- **Pre-Keccak512:** ~2,000 operations
- **ROMix Phase A:** 4,194,304 × 20 ChaCha rounds = 83,886,080 operations
- **ROMix Phase B:** 4,194,304 × 20 ChaCha rounds = 83,886,080 operations
- **Post-Keccak512:** ~1,000 operations
- **Total:** ~168 million operations per hash

**Throughput:**
- Typical: 128-256 threads per launch
- Hashrate: 50-150 H/s (highly dependent on intensity and memory bandwidth)

---

## 7. Complete Execution Timeline

### Double-Buffered Execution

```
Time →

Stream 0 (cur=0):  [Pre-Keccak] [ROMix A] [ROMix B] [Post-Keccak] [DtoH] [Check]
                    └────────────────────────────────────────────────┘
                                                                      ↓
Stream 1 (nxt=1):                    [Pre-Keccak] [ROMix A] [ROMix B] [Post-Keccak] [DtoH] [Check]
                                      └────────────────────────────────────────────────┘
                                                                                      ↓
Stream 0 (cur=0):                                                                    [Pre-Keccak] [ROMix A] ...
                                                                                      └──────────────────────┘
```

**Key Points:**
- While Stream 0 processes results, Stream 1 computes next batch
- Memory transfers overlap with computation
- GPU is kept busy continuously
- CPU validates results while GPU computes next batch

---

## 8. Kernel Launch Configuration

### Grid and Block Dimensions

**For GTX 1070 with throughput=256:**

```c
// Pre/Post-Keccak512 kernels
dim3 block(128);                    // 128 threads per block
dim3 grid((256+127)/128) = 2;      // 2 blocks

// ROMix kernels (TitanKernel)
dim3 grid(WU_PER_LAUNCH/WU_PER_BLOCK, 1, 1);
dim3 threads(THREADS_PER_WU*WU_PER_BLOCK, 1, 1);
```

**Typical Values:**
- `WU_PER_LAUNCH` = 256 (throughput)
- `WU_PER_BLOCK` = 4-8 (depends on kernel)
- `THREADS_PER_WU` = 1 (for TitanKernel)
- `WARPS_PER_BLOCK` = 4-8

**Total Threads:**
- Pre/Post-Keccak512: 256 threads
- ROMix: 256 threads (one per work unit)

---

## 9. Key Differences: `parallel == 2` vs `parallel < 2`

### `parallel == 2` (Full GPU - Current Implementation)

**Advantages:**
- ✅ Maximum GPU utilization
- ✅ Minimal CPU-GPU transfers
- ✅ Faster overall performance
- ✅ Better for high-intensity mining

**Execution:**
- All PBKDF2 operations on GPU (Keccak-512)
- ROMix on GPU
- Only final hashes transferred to CPU

### `parallel < 2` (Hybrid Mode)

**Execution (scrypt-jane.cpp:643-705):**
```c
if (parallel < 2) {
    // PBKDF2 on CPU
    for(int i=0; i<throughput; ++i) {
        scrypt_pbkdf2_1(..., Xbuf[nxt].ptr + 128 * i, 128);
    }
    
    // Transfer to GPU
    memcpy(cuda_X[nxt], Xbuf[nxt].ptr, 128 * throughput);
    cuda_scrypt_HtoD(thr_id, cuda_X[nxt], nxt);
    
    // ROMix on GPU only
    cuda_scrypt_core(thr_id, nxt, N);
    
    // Transfer back
    cuda_scrypt_DtoH(thr_id, cuda_X[nxt], nxt, false);
    
    // Final PBKDF2 on CPU
    for(int i=0; i<throughput; ++i) {
        scrypt_pbkdf2_1(..., (unsigned char*)&hash[cur][8*i], 32);
    }
}
```

**Disadvantages:**
- ❌ More CPU-GPU transfers
- ❌ CPU becomes bottleneck
- ❌ Lower overall performance

**Why Not Used:**
- Modern GPUs (Pascal+) have sufficient compute for Keccak-512
- GPU implementation is optimized and faster
- `parallel = 2` is hardcoded for optimal performance

---

## 10. Summary

### Main Mining Loop Flow (parallel=2)

1. **Initialize:** Allocate buffers, set up double-buffering
2. **Loop:**
   - Update nonces for next batch
   - Serialize execution
   - **Pre-Keccak512:** PBKDF2 on GPU (128 bytes output)
   - **ROMix Phase A:** Sequential writes to scratchpad (N iterations)
   - **ROMix Phase B:** Random reads from scratchpad (N iterations)
   - **Post-Keccak512:** Final PBKDF2 on GPU (32 bytes output)
   - Transfer results to CPU
   - Check for solutions
   - Swap buffers (cur ↔ nxt)
3. **Validation:** CPU verification of potential solutions
4. **Submission:** Submit valid solutions

### GPU Kernel Execution (GTX 1070, Nfactor=21)

- **Kernel:** TitanKernel (optimized for Pascal architecture)
- **Scratchpad:** 512 MB (N × 128 bytes)
- **Iterations:** 4,194,304 per hash
- **Mixing Function:** ChaCha20 (20 rounds)
- **Memory Pattern:** Sequential writes, random reads
- **Bottleneck:** Memory bandwidth for random reads

### Why `parallel = 2`?

- **Performance:** Full GPU utilization maximizes hashrate
- **Efficiency:** Reduces CPU-GPU transfers
- **Modern GPUs:** Pascal+ architecture handles Keccak-512 efficiently
- **Best Practice:** Industry standard for scrypt-jane mining

---

## Conclusion

The scrypt-jane implementation with `parallel = 2` leverages the full computational power of modern GPUs like the GTX 1070. The double-buffered design ensures continuous GPU utilization while the CPU validates results. The TitanKernel is specifically optimized for Pascal architecture, providing optimal performance for the memory-intensive ROMix operations required by scrypt-jane with Nfactor=21.


