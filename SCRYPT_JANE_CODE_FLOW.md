# ccminer Scrypt-Jane Algorithm Code Flow Analysis
## GeForce GTX 1070, Nfactor = 21

This document provides a detailed code flow analysis for ccminer running the scrypt-jane algorithm with Nfactor=21 on a GeForce GTX 1070.

---

## 1. Program Initialization

### Entry Point: `main()` (ccminer.cpp:4136)

**Initial Setup:**
```c
// Line 4145: Print version info
printf("*** ccminer " PACKAGE_VERSION " for nVidia GPUs by tpruvot@github ***\n");

// Line 4159-4162: Initialize global strings
rpc_user = strdup("");
rpc_pass = strdup("");
rpc_url = strdup("");
jane_params = strdup("");

// Line 4164-4168: Initialize mutexes
pthread_mutex_init(&applog_lock, NULL);
pthread_mutex_init(&stratum_sock_lock, NULL);
pthread_mutex_init(&stratum_work_lock, NULL);
pthread_mutex_init(&stats_lock, NULL);
pthread_mutex_init(&g_work_lock, NULL);
```

**CPU Detection:**
```c
// Line 4171-4185: Detect number of CPU cores
#if defined(WIN32)
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    num_cpus = sysinfo.dwNumberOfProcessors;
#elif defined(_SC_NPROCESSORS_CONF)
    num_cpus = sysconf(_SC_NPROCESSORS_CONF);
#endif
```

**GPU Device Initialization:**
```c
// Line 4188: Detect CUDA devices
active_gpus = cuda_num_devices();

// Line 4190-4203: Initialize device arrays
for (i = 0; i < MAX_GPUS; i++) {
    device_map[i] = i % active_gpus;
    device_name[i] = NULL;
    device_config[i] = NULL;
    device_backoff[i] = is_windows() ? 12 : 2;
    device_bfactor[i] = is_windows() ? 11 : 0;
    device_lookup_gap[i] = 1;
    device_batchsize[i] = 1024;
    device_interactive[i] = -1;
    device_singlememory[i] = -1;
    device_pstate[i] = -1;
    device_led[i] = -1;
}

// Line 4205: Get device names
cuda_devicenames();
```

**Key Global Variables:**
- `opt_algo` (line 119): Algorithm selection (ALGO_SCRYPT_JANE = 50)
- `opt_nfactor` (line 159): N-factor for scrypt-jane (default 14, set to 21)
- `opt_n_threads` (line 120): Number of mining threads
- `device_map[]` (line 133): Maps thread ID to GPU device ID
- `gpus_intensity[]` (line 136): GPU intensity settings
- `pools[]` (line 164): Pool configuration array
- `g_work` (line 517): Global work structure shared by all threads

---

## 2. Command-Line Parsing

### Entry: `parse_cmdline()` (ccminer.cpp:4038)

**Algorithm Selection:**
```c
// Line 3328-3352: Parse --algo argument
case 'a': /* --algo */
    p = strstr(arg, ":"); // Optional factor
    if (p) *p = '\0';
    
    i = algo_to_int(arg);  // Converts "scrypt-jane" to ALGO_SCRYPT_JANE
    if (i >= 0)
        opt_algo = (enum sha_algos)i;
    
    if (p) {
        opt_nfactor = atoi(p + 1);  // Extract Nfactor from "scrypt-jane:21"
        if (opt_algo == ALGO_SCRYPT_JANE) {
            free(jane_params);
            jane_params = strdup(p+1);  // Store "21" in jane_params
        }
    }
    
    // Set default Nfactor if not specified
    if (!opt_nfactor) {
        switch (opt_algo) {
        case ALGO_SCRYPT:      opt_nfactor = 9;  break;
        case ALGO_SCRYPT_JANE: opt_nfactor = 14; break;
        }
    }
```

**Example Command:**
```bash
./ccminer -a scrypt-jane:21 -o http://yacoind:8332 -u user -p pass
```

**Parsed Values:**
- `opt_algo = ALGO_SCRYPT_JANE` (50)
- `opt_nfactor = 21`
- `jane_params = "21"`

**Other Important Parsing:**
- `-d, --devices`: GPU device selection (line 3897-3931)
- `-i, --intensity`: GPU intensity (line 3453-3483)
- `-o, --url`: Pool URL (line 3562-3610)
- `-u, --user`: Username (line 3557-3560)
- `-p, --pass`: Password (line 3510-3513)

---

## 3. GPU Detection

### Function: `cuda_num_devices()` (called from main:4188)

**Detection Process:**
```c
// Line 4188: Count CUDA devices
active_gpus = cuda_num_devices();

// Line 4205: Get device names and properties
cuda_devicenames();  // Populates device_name[] array

// Line 4331-4338: Validate GPU count
if (active_gpus == 0) {
    applog(LOG_ERR, "No CUDA devices found! terminating.");
    exit(1);
}
if (!opt_n_threads)
    opt_n_threads = active_gpus;  // Default: one thread per GPU
```

**Device Mapping:**
```c
// Line 4190-4191: Map threads to devices
for (i = 0; i < MAX_GPUS; i++) {
    device_map[i] = i % active_gpus;  // Round-robin assignment
}
```

**For GTX 1070:**
- Device detected: GeForce GTX 1070
- Compute Capability: 6.1 (Pascal architecture)
- Memory: 8GB GDDR5
- `device_map[0] = 0` (first thread uses GPU 0)

**Device Properties Stored:**
- `device_sm[]` (line 134): Compute capability (SM version)
- `device_mpcount[]` (line 135): Multiprocessor count
- `device_name[]` (line 132): Device name string

---

## 4. GPU Setup

### Thread Creation: `main()` (ccminer.cpp:4492-4510)

**Mining Thread Initialization:**
```c
// Line 4492-4510: Create mining threads
for (i = 0; i < opt_n_threads; i++) {
    thr = &thr_info[i];
    
    thr->id = i;
    thr->gpu.thr_id = i;
    thr->gpu.gpu_id = (uint8_t) device_map[i];  // Maps to GPU 0
    thr->gpu.gpu_arch = (uint16_t) device_sm[device_map[i]];  // SM 6.1
    
    thr->q = tq_new();  // Create thread queue
    
    pthread_mutex_init(&thr->gpu.monitor.lock, NULL);
    pthread_cond_init(&thr->gpu.monitor.sampling_signal, NULL);
    
    pthread_create(&thr->pth, NULL, miner_thread, thr);
}
```

### GPU Context Setup: `scanhash_scrypt_jane()` (scrypt-jane.cpp:584-596)

**First-Time GPU Initialization:**
```c
// Line 583-596: Initialize GPU context (first call only)
static __thread int throughput = 0;
if(!init[thr_id]) {
    int dev_id = device_map[thr_id];
    
    // Reset GPU context
    cudaSetDevice(dev_id);
    cudaDeviceSynchronize();
    cudaDeviceReset();
    cudaSetDevice(dev_id);
    
    // Initialize GPU throughput and memory
    throughput = cuda_throughput(thr_id);  // ← Critical GPU setup
    gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", 
           throughput2intensity(throughput), throughput);
    
    init[thr_id] = true;
}
```

**GPU Throughput Calculation: `cuda_throughput()` (scrypt/salsa_kernel.cu:137)**

This function:
1. Determines optimal CUDA thread configuration
2. Allocates GPU memory for scrypt-jane (N=2^22 = 4,194,304)
3. Selects appropriate kernel (Titan, Kepler, Fermi, etc.)
4. For GTX 1070 (Pascal): Uses TitanKernel or optimized Pascal kernel

**Memory Allocation:**
- **Vbuf**: N × 128 bytes = 4,194,304 × 128 = ~536 MB (scratchpad)
- **Xbuf**: 2 × throughput × 128 bytes (double buffering)
- **Ybuf**: 128 bytes (temporary)
- **Total**: ~550-600 MB per thread

**Key Structures:**
```c
struct thr_info {
    int id;
    struct cgpu_info gpu;
    pthread_t pth;
    struct tq *q;
};

struct cgpu_info {
    uint8_t gpu_id;
    uint16_t gpu_arch;  // SM version
    int thr_id;
    struct gpu_monitor monitor;
};
```

---

## 5. Worker Threads Getting Work from yacoind

### Work I/O Thread: `workio_thread()` (ccminer.cpp:1521-1576)

**Thread Creation:**
```c
// Line 4392-4403: Create workio thread
work_thr_id = opt_n_threads;
thr = &thr_info[work_thr_id];
thr->id = work_thr_id;
thr->q = tq_new();

pthread_create(&thr->pth, NULL, workio_thread, thr);
```

**Work Retrieval Loop:**
```c
// Line 1533-1576: Main workio thread loop
while (ok && !abort_flag) {
    struct workio_cmd *wc;
    
    wc = (struct workio_cmd *)tq_pop(mythr->q, NULL);  // Wait for command
    
    switch (wc->cmd) {
    case WC_GET_WORK:
        ok = workio_get_work(wc, curl);  // ← Get work from pool
        break;
    case WC_SUBMIT_WORK:
        ok = workio_submit_work(wc, curl);  // Submit solution
        break;
    }
}
```

### Getting Work: `workio_get_work()` (ccminer.cpp:1453-1493)

```c
// Line 1458-1460: Allocate work structure
ret_work = (struct work*)aligned_calloc(sizeof(struct work));

// Line 1463: Assign pool number
ret_work->pooln = wc->pooln;

// Line 1467: Get work from yacoind
while (!get_upstream_work(curl, ret_work)) {
    // Retry logic on failure
    if (unlikely((opt_retries >= 0) && (++failures > opt_retries))) {
        applog(LOG_ERR, "get_work json_rpc_call failed");
        aligned_free(ret_work);
        return false;
    }
    sleep(opt_fail_pause);
}

// Line 1489: Send work to requesting thread
if (!tq_push(wc->thr->q, ret_work))
    aligned_free(ret_work);
```

### Upstream Work Retrieval: `get_upstream_work()` (ccminer.cpp:1362-1417)

```c
// Line 1370: Start timing
gettimeofday(&tv_start, NULL);

// Line 1390: Make JSON-RPC call to yacoind
val = json_rpc_call_pool(curl, pool, rpc_req, want_longpoll, have_longpoll, NULL);

// Line 1402: Decode work from JSON response
rc = work_decode(json_object_get(val, "result"), work);

// Line 1413-1414: Get additional info
get_mininginfo(curl, work);      // Network difficulty, hashrate
get_blocktemplate(curl, work);   // Block height
```

### Work Decoding: `work_decode()` (ccminer.cpp:744-878)

**For scrypt-jane (default case):**
```c
// Line 750-771: Determine data size
switch (opt_algo) {
    // ... other algorithms ...
    default:
        data_size = 128;      // 128 bytes for scrypt-jane
        adata_sz = data_size / 4;  // 32 uint32_t words
}

// Line 773-786: Decode "data" field (block header)
if (!jobj_binary(val, "data", work->data, data_size)) {
    // Handle hex string conversion
    hex2bin((uchar*)work->data, hexstr, data_size);
}

// Line 788-791: Decode "target" field
if (!jobj_binary(val, "target", work->target, target_size)) {
    applog(LOG_ERR, "JSON invalid target");
    return false;
}

// Line 799-802: Convert endianness
for (i = 0; i < adata_sz; i++)
    work->data[i] = le32dec(work->data + i);  // Little-endian decode
for (i = 0; i < atarget_sz; i++)
    work->target[i] = le32dec(work->target + i);

// Line 813: Calculate target difficulty
work->targetdiff = target_to_diff(work->target);

// Line 849: Generate job ID from ntime
cbin2hex(work->job_id, (const char*)&work->data[17], 4);
```

**Work Structure:**
```c
struct work {
    uint32_t data[32];        // Block header (128 bytes for scrypt-jane)
    uint32_t target[8];       // Target difficulty
    double targetdiff;         // Calculated difficulty
    uint32_t height;          // Block height
    char job_id[65];          // Job identifier
    uint32_t nonces[4];       // Found nonces
    int pooln;                 // Pool number
    // ... other fields ...
};
```

**JSON-RPC Request to yacoind:**
```json
{
    "method": "getwork",
    "params": [],
    "id": 0
}
```

**JSON-RPC Response from yacoind:**
```json
{
    "result": {
        "data": "00000020...",  // 256 hex chars = 128 bytes
        "target": "00000000...", // 64 hex chars = 32 bytes
        "hash1": "..."
    },
    "error": null,
    "id": 0
}
```

---

## 6. Mining Threads Processing the Work

### Mining Thread: `miner_thread()` (ccminer.cpp:1938-2934)

**Main Loop Structure:**
```c
// Line 2006: Main mining loop
while (!abort_flag) {
    struct timeval tv_start, tv_end, diff;
    unsigned long hashes_done;
    uint32_t start_nonce, max_nonce;
    
    // ... work retrieval and preparation ...
    
    // Line 2534: Call algorithm-specific scanhash function
    switch (opt_algo) {
        case ALGO_SCRYPT_JANE:
            rc = scanhash_scrypt_jane(thr_id, &work, max_nonce, &hashes_done,
                                      NULL, &tv_start, &tv_end, nVersion);
            break;
    }
    
    // ... solution submission ...
}
```

### Work Retrieval in Mining Thread (ccminer.cpp:2087-2123)

```c
// Line 2089-2121: Get work from global work structure
pthread_mutex_lock(&g_work_lock);

secs = (uint32_t) (time(NULL) - g_work_time);
if (secs >= scan_time || nonceptr[0] >= (end_nonce - 0x100)) {
    // Obtain new work from workio thread
    if (!get_work(mythr, &g_work)) {
        pthread_mutex_unlock(&g_work_lock);
        continue;
    }
    g_work_time = time(NULL);
}

// Line 2126-2141: Update work if changed
if (!opt_benchmark && (g_work.height != work.height || 
                       memcmp(work.target, g_work.target, sizeof(work.target)))) {
    memcpy(work.target, g_work.target, sizeof(work.target));
    work.targetdiff = g_work.targetdiff;
    work.height = g_work.height;
}

pthread_mutex_unlock(&g_work_lock);
```

### Scrypt-Jane Specific Work Preparation (ccminer.cpp:2044-2163)

```c
// Line 2026: Extract block version
int nVersion = swab32(work.data[0]);

// Line 2044-2052: Handle hardfork (version >= 7)
if (opt_algo == ALGO_SCRYPT_JANE) {
    if (nVersion >= 7) {
        // 64-bit nTime, nonce at offset 80
        nonceptr = (uint32_t*) (((char*)work.data) + 80);
        wcmplen = 80;
    }
}

// Line 2149-2163: Detect hardfork at runtime
static __thread int hardFork = false;
if (!hardFork && opt_algo == ALGO_SCRYPT_JANE) {
    nVersion = swab32(g_work.data[0]);
    if (nVersion >= 7) {
        nonceptr = (uint32_t*) (((char*)work.data) + 80);
        wcmplen = 80;
        hardFork = true;
        applog(LOG_NOTICE, "yacoin hardfork detected, nVersion = %d", nVersion);
    }
}
```

### Scrypt-Jane Mining: `scanhash_scrypt_jane()` (scrypt-jane.cpp:459-829)

**Function Signature:**
```c
int scanhash_scrypt_jane(int thr_id, struct work *work, uint32_t max_nonce,
                         unsigned long *hashes_done, unsigned char *scratchbuf,
                         struct timeval *tv_start, struct timeval *tv_end,
                         int block_version)
```

**Step-by-Step Execution:**

**1. Extract Parameters:**
```c
// Line 462-465: Get work data and target
uint32_t *pdata = work->data;
uint32_t *ptarget = work->target;
const uint32_t Htarg = ptarget[7];  // Hash target for quick comparison

// Line 550-579: Calculate N-factor
int Nfactor = GetNfactor(pdata[17], minn, maxn, starttime);
// For Nfactor=21: N = 2^(21+1) = 2^22 = 4,194,304

uint32_t N = (1 << (Nfactor + 1));
int block_header_size = (block_version >= 7) ? 84 : 80;
```

**2. GPU Initialization (First Call):**
```c
// Line 583-596: Initialize GPU context
if(!init[thr_id]) {
    int dev_id = device_map[thr_id];
    cudaSetDevice(dev_id);
    cudaDeviceSynchronize();
    cudaDeviceReset();
    cudaSetDevice(dev_id);
    
    throughput = cuda_throughput(thr_id);  // Get optimal thread count
    init[thr_id] = true;
}
```

**3. Allocate Memory:**
```c
// Line 603-604: Allocate host memory for block headers
uint32_t *data[2] = { 
    new uint32_t[(block_header_size/4)*throughput], 
    new uint32_t[(block_header_size/4)*throughput] 
};

// Line 604: Get GPU hash buffers
uint32_t* hash[2] = { 
    cuda_hashbuffer(thr_id, 0), 
    cuda_hashbuffer(thr_id, 1) 
};

// Line 626-628: Allocate scrypt buffers
scrypt_aligned_alloc Xbuf[2] = { 
    scrypt_alloc(128 * throughput), 
    scrypt_alloc(128 * throughput) 
};
scrypt_aligned_alloc Vbuf = scrypt_alloc(N * 128);  // ~536 MB
scrypt_aligned_alloc Ybuf = scrypt_alloc(128);
```

**4. Prepare Work Data:**
```c
// Line 619-623: Byte-swap and duplicate block headers
for (int k=0; k<2; ++k) {
    for(int z=0; z<(block_header_size/4); z++) 
        data[k][z] = bswap_32x4(pdata[z]);
    for(int i=1; i<throughput; ++i) 
        memcpy(&data[k][(block_header_size/4)*i], &data[k][0], 
               (block_header_size/4)*sizeof(uint32_t));
}

// Line 624: Prepare Keccak constants for GPU
if (parallel == 2) 
    prepare_keccak512(thr_id, pdata, block_header_size);
```

**5. Main Mining Loop:**
```c
// Line 630-818: Double-buffered mining loop
uint32_t nonce[2];
uint32_t* cuda_X[2] = { cuda_scrypt_buffer_0(thr_id), cuda_scrypt_buffer_1(thr_id) };
int cur = 0, nxt = 1;
uint32_t n = pdata[(block_header_size/4 - 1)];
int iteration = 0;

do {
    // Update nonces for this batch
    for (int i=0; i<throughput; i++) {
        data[cur][(block_header_size/4 - 1) + (block_header_size/4)*i] = 
            bswap_32x4(n + i);
    }
    nonce[cur] = n;
    
    // Transfer data to GPU
    cuda_scrypt_HtoD(thr_id, cuda_X[cur], data[cur], 
                     (block_header_size/4)*throughput);
    
    // Launch GPU kernel
    cuda_scrypt_core(thr_id, throughput, Nfactor, block_header_size, 
                     cuda_X[cur], cuda_hashbuffer(thr_id, cur));
    
    // Process previous batch results
    if (iteration > 0) {
        cuda_scrypt_sync(thr_id, nxt);
        
        // Check for solutions
        for (int i=0; i<throughput; i++) {
            if (hash[cur][8*i+7] <= Htarg && 
                fulltest(&hash[cur][8*i], ptarget)) {
                // Validate on CPU
                // ... validation code ...
                if (valid) {
                    work->nonces[0] = nonce[cur] + i;
                    return 1;  // Solution found!
                }
            }
        }
    }
    
    // Swap buffers
    cur = (cur+1)&1;
    nxt = (nxt+1)&1;
    n += throughput;
    ++iteration;
    
} while (n <= max_nonce && !work_restart[thr_id].restart);
```

**6. GPU Kernel Execution: `cuda_scrypt_core()`**

The GPU kernel performs:
1. **Pre-Keccak512**: PBKDF2 with Keccak-512 to generate scrypt input
2. **ROMix**: Sequential writes + random reads (N=4,194,304 iterations)
   - Uses ChaCha20 mixing function
   - Requires ~536 MB scratchpad memory
3. **Post-Keccak512**: Final PBKDF2 with Keccak-512 to generate hash

**7. Solution Validation:**
```c
// Line 769-812: Validate potential solutions
if (hash[cur][8*i+7] <= Htarg && fulltest(&hash[cur][8*i], ptarget)) {
    // Recompute on CPU for verification
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
        work->nonces[0] = tmp_nonce;
        return 1;  // Valid solution!
    }
}
```

**8. Cleanup:**
```c
// Line 820-823: Free allocated memory
scrypt_free(&Vbuf);
scrypt_free(&Ybuf);
scrypt_free(&Xbuf[0]); 
scrypt_free(&Xbuf[1]);
delete[] data[0]; 
delete[] data[1];

*hashes_done = n - pdata[(block_header_size / 4 - 1)];
pdata[(block_header_size / 4 - 1)] = n;
gettimeofday(tv_end, NULL);
return 0;  // No solution found in this range
```

---

## 7. Solution Submission

### Submission Flow: `miner_thread()` → `submit_work()` (ccminer.cpp:2888-2924)

**When Solution Found:**
```c
// Line 2889-2899: Submit first nonce
if (rc > 0 && !opt_benchmark) {
    uint32_t curnonce = nonceptr[0];  // Save current position
    
    work.submit_nonce_id = 0;
    nonceptr[0] = work.nonces[0];  // Set found nonce
    
    if (!submit_work(mythr, &work))
        break;
    
    nonceptr[0] = curnonce;  // Restore position
}
```

### Submit Work Function: `submit_work()` (ccminer.cpp:1626-1652)

```c
// Line 1629-1641: Create work submission command
struct workio_cmd *wc;
wc = (struct workio_cmd *)calloc(1, sizeof(*wc));

wc->u.work = (struct work *)aligned_calloc(sizeof(*work_in));
memcpy(wc->u.work, work_in, sizeof(struct work));

wc->cmd = WC_SUBMIT_WORK;
wc->thr = thr;
wc->pooln = work_in->pooln;

// Line 1644: Send to workio thread
if (!tq_push(thr_info[work_thr_id].q, wc))
    goto err_out;
```

### Work Submission: `submit_upstream_work()` (ccminer.cpp:940-1235)

**For scrypt-jane (getwork mode):**
```c
// Line 1054-1067: Handle scrypt-jane submission
case ALGO_SCRYPT_JANE:
    nVersion = swab32(work->data[0]);
    if (nVersion >= 7) {
        // 64-bit nTime, nonce at offset 80
        le32enc(&ntime, work->data[17]);
        le32enc(&nonce, work->data[19]);
    } else {
        // 32-bit nTime, nonce at offset 76
        le32enc(&ntime, work->data[18]);
        le32enc(&nonce, work->data[20]);
    }
    break;
```

**JSON-RPC Submission:**
```c
// Line 1208-1211: Build JSON-RPC request
sprintf(s, "{\"method\": \"getwork\", \"params\": [\"%s\"], \"id\":10}\r\n", str);

// Line 1214: Send to yacoind
val = json_rpc_call_pool(curl, pool, s, false, false, NULL);
```

**Response Handling:**
```c
// Line 1220-1227: Process response
res = json_object_get(val, "result");
reason = json_object_get(val, "reject-reason");

share_result(json_is_true(res), work->pooln, work->sharediff[0],
             reason ? json_string_value(reason) : NULL);
```

---

## 8. Key Data Structures

### Global Work Structure
```c
// Line 517-518: Global work shared by all threads
struct work _ALIGN(64) g_work;
volatile time_t g_work_time;
pthread_mutex_t g_work_lock;
```

### Work Structure (miner.h:730-764)
```c
struct work {
    uint32_t data[48];           // Block header (128 bytes for scrypt-jane)
    uint32_t target[8];          // Target difficulty (32 bytes)
    uint32_t maxvote;            // Heavycoin voting
    
    char job_id[128];            // Job identifier
    size_t xnonce2_len;          // Stratum extranonce2 length
    uchar xnonce2[32];           // Stratum extranonce2
    
    uint8_t pooln;               // Pool number
    uint8_t valid_nonces;       // Number of valid nonces found
    uint8_t submit_nonce_id;    // Which nonce to submit
    uint8_t job_nonce_id;       // Job nonce ID
    
    uint32_t nonces[MAX_NONCES]; // Found nonces (up to 2)
    double sharediff[MAX_NONCES]; // Share difficulty
    double shareratio[MAX_NONCES]; // Share ratio
    double targetdiff;           // Target difficulty
    
    uint32_t height;             // Block height
    uint32_t scanned_from;       // Nonce range start
    uint32_t scanned_to;         // Nonce range end
};
```

### Thread Information Structure (miner.h:517-522)
```c
struct thr_info {
    int id;                      // Thread ID
    pthread_t pth;               // POSIX thread handle
    struct thread_q *q;          // Thread queue
    struct cgpu_info gpu;        // GPU information
};
```

### GPU Information Structure (miner.h:436-469)
```c
struct cgpu_info {
    uint8_t gpu_id;              // CUDA device ID
    uint8_t thr_id;              // Thread ID
    uint16_t gpu_arch;           // SM version (610 for GTX 1070)
    uint32_t gpu_clock;           // GPU clock (MHz)
    uint32_t gpu_memclock;        // Memory clock (MHz)
    uint64_t gpu_mem;             // Total memory (MB)
    uint64_t gpu_memfree;         // Free memory (MB)
    uint32_t throughput;          // CUDA threads
    double intensity;             // Mining intensity
    struct monitor_info monitor;  // Monitoring data
};
```

### Pool Information Structure (miner.h:770-813)
```c
struct pool_infos {
    uint8_t id;                  // Pool ID
    uint8_t type;                // POOL_GETWORK | POOL_STRATUM | POOL_LONGPOLL
    uint16_t status;             // Pool status flags
    int algo;                    // Algorithm
    char url[512];               // Pool URL
    char user[192];              // Username
    char pass[384];              // Password
    double max_diff;             // Max network difficulty
    double max_rate;             // Max network hashrate
    uint32_t accepted_count;     // Accepted shares
    uint32_t rejected_count;    // Rejected shares
    uint32_t solved_count;       // Blocks solved
};
```

---

## 9. Memory Layout for Scrypt-Jane (Nfactor=21)

### Block Header Layout
```
Offset  Size  Field
------  ----  -----
0       4     Version (nVersion)
4       32    Previous Block Hash (hashPrevBlock)
36      32    Merkle Root (hashMerkleRoot)
68      4/8   Timestamp (nTime) - 4 bytes if version < 7, 8 bytes if >= 7
72/76   4     Bits (nBits)
76/80   4     Nonce (nNonce)
```

**For Version >= 7 (Hardfork):**
- Total size: 84 bytes
- Nonce at offset 80

**For Version < 7:**
- Total size: 80 bytes
- Nonce at offset 76

### GPU Memory Allocation

**For Nfactor=21 (N = 2^22 = 4,194,304):**

1. **Vbuf (Scratchpad):**
   - Size: N × 128 bytes = 4,194,304 × 128 = 536,870,912 bytes (~512 MB)
   - Purpose: ROMix scratchpad for sequential writes and random reads

2. **Xbuf (Double Buffer):**
   - Size: 2 × throughput × 128 bytes
   - Example: 2 × 256 × 128 = 65,536 bytes (~64 KB)
   - Purpose: Input/output buffer for scrypt operations

3. **Ybuf (Temporary):**
   - Size: 128 bytes
   - Purpose: Temporary buffer for ROMix operations

4. **Hash Buffers:**
   - Size: 2 × throughput × 32 bytes (for final hashes)
   - Purpose: Store computed hashes from GPU

5. **Data Buffers:**
   - Size: 2 × throughput × block_header_size
   - Purpose: Block header data for parallel processing

**Total GPU Memory per Thread:**
- ~550-600 MB (depending on throughput)

---

## 10. Complete Execution Flow Summary

### High-Level Flow

```
1. Program Initialization (main:4136)
   ├── Parse command line arguments
   ├── Initialize CUDA runtime
   ├── Detect GPUs (cuda_num_devices)
   ├── Initialize device arrays
   └── Create threads

2. Command-Line Parsing (parse_cmdline:4038)
   ├── Parse --algo=scrypt-jane:21
   ├── Set opt_algo = ALGO_SCRYPT_JANE
   ├── Set opt_nfactor = 21
   └── Parse pool URL, credentials

3. GPU Detection (cuda_num_devices:28)
   ├── Query CUDA driver version
   ├── Get device count
   ├── Get device properties (GTX 1070: SM 6.1, 8GB)
   └── Populate device_map[]

4. Thread Creation (main:4492)
   ├── Create workio thread (workio_thread)
   ├── Create stratum thread (stratum_thread)
   ├── Create longpoll thread (longpoll_thread)
   └── Create mining threads (miner_thread)

5. Work Retrieval (workio_thread:1521)
   ├── Wait for WC_GET_WORK command
   ├── Call get_upstream_work()
   │   ├── JSON-RPC call to yacoind
   │   ├── Parse response (work_decode)
   │   └── Store in g_work
   └── Send work to mining thread

6. Mining Loop (miner_thread:1938)
   ├── Get work from g_work (get_work:1578)
   ├── Prepare nonce range
   ├── Call scanhash_scrypt_jane()
   │   ├── Initialize GPU (first call)
   │   ├── Allocate memory
   │   ├── Prepare block headers
   │   ├── Launch GPU kernels
   │   │   ├── Pre-Keccak512 (PBKDF2)
   │   │   ├── ROMix (4.2M iterations)
   │   │   └── Post-Keccak512 (PBKDF2)
   │   ├── Check for solutions
   │   └── Validate on CPU
   └── Submit solutions (submit_work:1626)

7. Solution Submission (submit_upstream_work:940)
   ├── Build JSON-RPC request
   ├── Send to yacoind
   ├── Process response
   └── Update statistics
```

### Detailed Scrypt-Jane Mining Flow

```
scanhash_scrypt_jane(thr_id, work, max_nonce, ...)
│
├─→ Extract Nfactor from block header
│   └─→ N = 2^(Nfactor+1) = 2^22 = 4,194,304
│
├─→ Initialize GPU (first call only)
│   ├─→ cudaSetDevice(device_map[thr_id])
│   ├─→ cuda_throughput(thr_id)
│   │   ├─→ Select kernel (TitanKernel for Pascal)
│   │   ├─→ Find optimal block count
│   │   └─→ Allocate GPU memory
│   └─→ init[thr_id] = true
│
├─→ Allocate memory
│   ├─→ Vbuf: N × 128 bytes (~512 MB)
│   ├─→ Xbuf[2]: 2 × throughput × 128 bytes
│   └─→ Ybuf: 128 bytes
│
├─→ Prepare work data
│   ├─→ Byte-swap block headers
│   ├─→ Duplicate for throughput
│   └─→ Prepare Keccak constants
│
└─→ Main mining loop (double-buffered)
    │
    ├─→ Iteration N:
    │   ├─→ Update nonces in data[cur]
    │   ├─→ cuda_scrypt_HtoD() - Transfer to GPU
    │   ├─→ cuda_scrypt_core() - Launch GPU kernel
    │   │   ├─→ Pre-Keccak512 (PBKDF2)
    │   │   ├─→ ROMix (sequential writes)
    │   │   ├─→ ROMix (random reads)
    │   │   └─→ Post-Keccak512 (PBKDF2)
    │   └─→ Process results from iteration N-1
    │
    ├─→ Check solutions
    │   ├─→ Quick check: hash[7] <= Htarg
    │   ├─→ Full check: fulltest(hash, target)
    │   └─→ CPU validation
    │
    └─→ Swap buffers (cur ↔ nxt)
```

---

## 11. Important Global Variables

### Algorithm Configuration
- `opt_algo` (ccminer.cpp:119): Current algorithm (ALGO_SCRYPT_JANE = 50)
- `opt_nfactor` (ccminer.cpp:159): N-factor (21 for this case)
- `jane_params` (ccminer.cpp:161): Scrypt-jane parameters string

### GPU Configuration
- `device_map[]` (ccminer.cpp:133): Maps thread ID to GPU device ID
- `device_sm[]` (ccminer.cpp:134): SM version per device
- `device_mpcount[]` (ccminer.cpp:135): Multiprocessor count
- `gpus_intensity[]` (ccminer.cpp:136): GPU intensity settings
- `device_config[]` (ccminer.cpp:154): Kernel launch configuration

### Work Management
- `g_work` (ccminer.cpp:517): Global work structure
- `g_work_time` (ccminer.cpp:518): Timestamp of last work update
- `g_work_lock` (ccminer.cpp:519): Mutex for g_work
- `work_restart[]` (ccminer.cpp:197): Restart flags per thread

### Pool Management
- `pools[]` (ccminer.cpp:164): Pool configuration array
- `cur_pooln` (ccminer.cpp:166): Current active pool
- `num_pools` (ccminer.cpp:165): Number of configured pools

### Threading
- `thr_info[]` (ccminer.cpp:188): Thread information array
- `opt_n_threads` (ccminer.cpp:120): Number of mining threads
- `work_thr_id` (ccminer.cpp:189): Work I/O thread ID
- `stratum_thr_id` (ccminer.cpp:192): Stratum thread ID

---

## 12. Performance Characteristics (GTX 1070, Nfactor=21)

### Memory Requirements
- **Scratchpad (Vbuf):** ~512 MB per thread
- **Total GPU Memory:** ~550-600 MB per thread
- **Throughput:** Typically 128-256 CUDA threads

### Computational Complexity
- **N = 2^22 = 4,194,304** iterations per hash
- **Each iteration:** ChaCha20 mixing (20 rounds)
- **Total operations:** ~84 million per hash

### Expected Performance
- **Hashrate:** ~50-150 H/s (highly dependent on intensity)
- **Memory bandwidth:** Critical bottleneck
- **Optimal intensity:** Auto-tuned by `cuda_throughput()`

---

## Conclusion

This document provides a comprehensive code flow analysis for ccminer running scrypt-jane with Nfactor=21 on a GeForce GTX 1070. The flow covers:

1. ✅ Program initialization and setup
2. ✅ Command-line parsing
3. ✅ GPU detection and configuration
4. ✅ GPU context setup and memory allocation
5. ✅ Work retrieval from yacoind
6. ✅ Mining thread processing
7. ✅ Solution submission

The scrypt-jane algorithm requires significant GPU memory (~512 MB per thread for Nfactor=21) and performs computationally intensive ROMix operations with 4.2 million iterations per hash attempt.