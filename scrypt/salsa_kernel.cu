//
// Contains the autotuning logic and some utility functions.
// Note that all CUDA kernels have been moved to other .cu files
//

#include <stdio.h>
#include <stdarg.h>
#include <map>
#include <algorithm>
#include <unistd.h> // usleep
#include <ctype.h> // tolower
#include <sys/sysinfo.h> // sysinfo
#include <pthread.h> // pthread_mutex for thread safety
#include "cuda_helper.h"

#include "salsa_kernel.h"

#include "volta_kernel.h"
#include "pascal_kernel.h"

#include "miner.h"

#if defined(_WIN64) || defined(__x86_64__) || defined(__64BIT__)
#define MAXMEM 0x300000000ULL  // 12 GB (the largest Kepler)
#else
#define MAXMEM  0xFFFFFFFFULL  // nearly 4 GB (32 bit limitations)
#endif

// require CUDA 5.5 driver API
#define DMAJ 5
#define DMIN 5

// define some error checking macros
#define DELIMITER '/'
#define __FILENAME__ ( strrchr(__FILE__, DELIMITER) != NULL ? strrchr(__FILE__, DELIMITER)+1 : __FILE__ )

#undef checkCudaErrors
#define checkCudaErrors(x) \
{ \
	cudaGetLastError(); \
	x; \
	cudaError_t err = cudaGetLastError(); \
	if (err != cudaSuccess && !abort_flag) \
		applog(LOG_ERR, "GPU #%d: Err %d: %s (%s:%d)", device_map[thr_id], err, cudaGetErrorString(err), __FILENAME__, __LINE__); \
}

// some globals containing pointers to device memory (for chunked allocation)
// [MAX_GPUS] indexes up to MAX_GPUS threads (0...MAX_GPUS-1)
int       MAXWARPS[MAX_GPUS];
int       MAXWARPS_RAM[MAX_GPUS];                      // Maximum warps using system RAM
uint32_t* h_V[MAX_GPUS][TOTAL_WARP_LIMIT*64];          // NOTE: the *64 prevents buffer overflow for --keccak
                                                       // h_V[0..MAXWARPS-1] = VRAM buffers, h_V[MAXWARPS..MAXWARPS+MAXWARPS_RAM-1] = system RAM buffers
uint32_t  h_V_extra[MAX_GPUS][TOTAL_WARP_LIMIT*64];    //       with really large kernel launch configurations

KernelInterface *Best_Kernel_Heuristics(cudaDeviceProp *props)
{
	KernelInterface *kernel = NULL;

	// Select kernel based on compute capability
	if (props->major >= 7)
	{
		// For compute capability >= 7.0, use VoltaKernel
		kernel = new VoltaKernel();
	}
	else
	{
		// For compute capability < 7.0, use PascalKernel
		kernel = new PascalKernel();
	}
	return kernel;
}


bool validate_config(char *config, int &b, int &w, KernelInterface **kernel = NULL, cudaDeviceProp *props = NULL)
{
	bool success = false;
	char kernelid = ' ';
	if (config != NULL)
	{
		if (config[0] == 'V' || config[0] == 'T' || config[0] == 'v' ||
			config[0] == 'P' || config[0] == 't' || config[0] == 'p') {
			kernelid = config[0];
			config++;
		}

		if (config[0] >= '0' && config[0] <= '9')
			if (sscanf(config, "%dx%d", &b, &w) == 2)
				success = true;

		if (success && kernel != NULL)
		{
			switch (kernelid)
			{
				case 'V': case 'T': case 'v': // VoltaKernel identifiers
					*kernel = new VoltaKernel(); break;
				case 'P': case 't': case 'p': // PascalKernel identifiers
					*kernel = new PascalKernel(); break;
				case ' ': // choose based on device architecture
					*kernel = Best_Kernel_Heuristics(props);
				break;
			}
		}
	}
	return success;
}

std::map<int, int> context_blocks;
std::map<int, int> context_wpb;
std::map<int, int> context_threads_per_warp;
std::map<int, int> context_wu_per_warp;
std::map<int, int> context_wu_per_block;
std::map<int, int> context_wu_per_launch;
std::map<int, bool> context_concurrent;
std::map<int, KernelInterface *> context_kernel;
std::map<int, uint32_t *> context_idata[2];
std::map<int, uint32_t *> context_odata[2];
std::map<int, cudaStream_t> context_streams[2];
std::map<int, uint32_t *> context_X[2];
std::map<int, uint32_t *> context_H[2];
std::map<int, cudaEvent_t> context_serialize[2];

// for scrypt-jane hashing on GPU
std::map<int, uint32_t *> context_hash[2];

int find_optimal_blockcount(int thr_id, KernelInterface* &kernel, bool &concurrent, int &wpb);
static void log_gpu_memory_info(int thr_id, const char *prefix_format, ...);

int cuda_throughput(int thr_id)
{
	int GRID_BLOCKS, WARPS_PER_BLOCK;
	if (context_blocks.find(thr_id) == context_blocks.end())
	{
		checkCudaErrors(cudaSetDevice(device_map[thr_id]));
		// Set device flags BEFORE creating context
		// If using system RAM, enable host memory mapping
		unsigned int flags = cudaDeviceScheduleYield;
		if (opt_use_system_ram) {
			flags |= cudaDeviceMapHost;
		}
		checkCudaErrors(cudaSetDeviceFlags(flags));

		KernelInterface *kernel;
		bool concurrent;
		GRID_BLOCKS = find_optimal_blockcount(thr_id, kernel, concurrent, WARPS_PER_BLOCK);

		if(GRID_BLOCKS == 0)
			return 0;

		unsigned int THREADS_PER_WU = kernel->threads_per_wu();
		unsigned int THREADS_PER_WARP = device_threads_per_warp[thr_id];
		unsigned int WU_PER_WARP = THREADS_PER_WARP / THREADS_PER_WU;
		unsigned int WU_PER_BLOCK = WU_PER_WARP * WARPS_PER_BLOCK;
		unsigned int WU_PER_LAUNCH = GRID_BLOCKS * WU_PER_BLOCK;
		
		unsigned int mem_size = WU_PER_LAUNCH * sizeof(uint32_t) * 32; // BLOCK DATA RECEIVED FROM YACOIND IS 128 BYTES
		unsigned int state_size = WU_PER_LAUNCH * sizeof(uint32_t) * 8;

		// allocate device memory for scrypt_core inputs and outputs
		uint32_t *tmp;
		checkCudaErrors(cudaMalloc((void **) &tmp, mem_size)); context_idata[0][thr_id] = tmp; // INPUT DATA 1
		checkCudaErrors(cudaMalloc((void **) &tmp, mem_size)); context_idata[1][thr_id] = tmp; // INPUT DATA 2
		checkCudaErrors(cudaMalloc((void **) &tmp, mem_size)); context_odata[0][thr_id] = tmp; // OUTPUT DATA 1
		checkCudaErrors(cudaMalloc((void **) &tmp, mem_size)); context_odata[1][thr_id] = tmp; // OUTPUT DATA 2

		// allocate pinned host memory for scrypt hashes
		checkCudaErrors(cudaHostAlloc((void **) &tmp, state_size, cudaHostAllocDefault)); context_H[0][thr_id] = tmp;
		checkCudaErrors(cudaHostAlloc((void **) &tmp, state_size, cudaHostAllocDefault)); context_H[1][thr_id] = tmp;

		// allocate pinned host memory for scrypt_core input/output (scrypt-jane)
		checkCudaErrors(cudaHostAlloc((void **) &tmp, mem_size, cudaHostAllocDefault)); context_X[0][thr_id] = tmp;
		checkCudaErrors(cudaHostAlloc((void **) &tmp, mem_size, cudaHostAllocDefault)); context_X[1][thr_id] = tmp;

		checkCudaErrors(cudaMalloc((void **) &tmp, state_size)); context_hash[0][thr_id] = tmp;
		checkCudaErrors(cudaMalloc((void **) &tmp, state_size)); context_hash[1][thr_id] = tmp;

		// create two CUDA streams
		cudaStream_t tmp2;
		checkCudaErrors( cudaStreamCreate(&tmp2) ); context_streams[0][thr_id] = tmp2;
		checkCudaErrors( cudaStreamCreate(&tmp2) ); context_streams[1][thr_id] = tmp2;

		// events used to serialize the kernel launches (we don't want any overlapping of kernels)
		cudaEvent_t tmp4;
		checkCudaErrors(cudaEventCreateWithFlags(&tmp4, cudaEventDisableTiming)); context_serialize[0][thr_id] = tmp4;
		checkCudaErrors(cudaEventCreateWithFlags(&tmp4, cudaEventDisableTiming)); context_serialize[1][thr_id] = tmp4;
		checkCudaErrors(cudaEventRecord(context_serialize[1][thr_id]));

		// Store calculated values in context
		context_kernel[thr_id] = kernel;
		context_concurrent[thr_id] = concurrent;
		context_blocks[thr_id] = GRID_BLOCKS;
		context_wpb[thr_id] = WARPS_PER_BLOCK;
		context_threads_per_warp[thr_id] = THREADS_PER_WARP;
		context_wu_per_warp[thr_id] = WU_PER_WARP;
		context_wu_per_block[thr_id] = WU_PER_BLOCK;
		context_wu_per_launch[thr_id] = WU_PER_LAUNCH;
	}

	return context_wu_per_launch[thr_id];
}

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct {
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = {
		{ 0x10, 8   }, // Tesla Generation (SM 1.0) G80 class
		{ 0x11, 8   }, // Tesla Generation (SM 1.1) G8x class
		{ 0x12, 8   }, // Tesla Generation (SM 1.2) G9x class
		{ 0x13, 8   }, // Tesla Generation (SM 1.3) GT200 class
		{ 0x20, 32  }, // Fermi Generation (SM 2.0) GF100 class
		{ 0x21, 48  }, // Fermi Generation (SM 2.1) GF10x class
		{ 0x30, 192 }, // Kepler Generation (SM 3.0) GK10x class - GK104 = 1536 cores / 8 SMs
		{ 0x35, 192 }, // Kepler Generation (SM 3.5) GK11x class
		{ 0x50, 128 }, // Maxwell First Generation (SM 5.0) GTX750/750Ti
		{ 0x52, 128 }, // Maxwell Second Generation (SM 5.2) GTX980 = 2048 cores / 16 SMs - GTX970 1664 cores / 13 SMs
		{ 0x61, 128 }, // Pascal GeForce (SM 6.1)
		{ 0x75, 64 }, // Turing RTX 2070 Super (SM 7.5)
		{ 0x86, 128 }, // Ampere RTX A5000 (SM 8.6)
		{ -1, -1 },
	};

	int index = 0;
	while (nGpuArchCoresPerSM[index].SM != -1)
	{
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
			return nGpuArchCoresPerSM[index].Cores;
		}
		index++;
	}

	// If we don't find the values, we default use the previous one to run properly
	applog(LOG_WARNING, "MapSMtoCores for SM %d.%d is undefined. Default to use %d Cores/SM", major, minor, 128);
	return 128;
}

#ifdef WIN32
#include <windows.h>
static int console_width() {
	CONSOLE_SCREEN_BUFFER_INFO csbi;
	GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
	return csbi.srWindow.Right - csbi.srWindow.Left + 1;
}
#else
static inline int console_width() {
	return 999;
}
#endif

// Calculate available system RAM per GPU
// Returns available RAM in bytes per GPU (distributed equally among all GPUs)
// Reads from /proc/meminfo first, falls back to sysinfo if unavailable
// Thread-safe: calculates value only once and caches it for subsequent calls
static size_t get_available_system_ram_per_gpu(int dev_id, const char *dev_name)
{
	// Static variables for thread-safe caching
	static size_t cached_ram_per_gpu = 0;
	static bool cached_value_valid = false;
	static pthread_mutex_t cache_mutex = PTHREAD_MUTEX_INITIALIZER;
	
	// Try to acquire lock and check cache
	pthread_mutex_lock(&cache_mutex);
	
	// If value is already cached, return it immediately
	if (cached_value_valid) {
		pthread_mutex_unlock(&cache_mutex);
		return cached_ram_per_gpu;
	}
	
	// Value not cached yet, calculate it now
	unsigned long mem_available = 0;
	unsigned long mem_total = 0;
	unsigned long mem_free = 0;
	FILE *f;
	char line[256];
	
	// Try reading from /proc/meminfo first
	f = fopen("/proc/meminfo", "r");
	if (f) {
		while (fgets(line, sizeof(line), f)) {
			if (sscanf(line, "MemTotal: %lu kB", &mem_total) == 1) {
				// Convert from KB to bytes
				mem_total *= 1024;
			} else if (sscanf(line, "MemFree: %lu kB", &mem_free) == 1) {
				// Convert from KB to bytes
				mem_free *= 1024;
			} else if (sscanf(line, "MemAvailable: %lu kB", &mem_available) == 1) {
				// Convert from KB to bytes
				mem_available *= 1024;
			}
		}
		fclose(f);
		
		if (mem_available > 0) {
			applog(LOG_INFO, "GPU #%d (%s): System RAM: MemTotal=%lu MB, MemFree=%lu MB, MemAvailable=%lu MB",
			       dev_id, dev_name,
			       (unsigned long)(mem_total / (1024 * 1024)),
			       (unsigned long)(mem_free / (1024 * 1024)),
			       (unsigned long)(mem_available / (1024 * 1024)));
		} else {
			// Fallback: use MemFree if MemAvailable not found
			mem_available = mem_free;
			applog(LOG_INFO, "GPU #%d (%s): System RAM: MemTotal=%lu MB, MemFree=%lu MB (MemAvailable not found, using MemFree)",
			       dev_id, dev_name,
			       (unsigned long)(mem_total / (1024 * 1024)),
			       (unsigned long)(mem_free / (1024 * 1024)));
		}
	}
	
	// Fallback to sysinfo if /proc/meminfo failed or MemAvailable not found
	if (mem_available == 0) {
		struct sysinfo si;
		if (sysinfo(&si) == 0) {
			// freeram + bufferram gives available memory
			mem_available = (unsigned long)si.freeram * si.mem_unit + (unsigned long)si.bufferram * si.mem_unit;
			mem_total = (unsigned long)si.totalram * si.mem_unit;
			mem_free = (unsigned long)si.freeram * si.mem_unit;
			applog(LOG_INFO, "GPU #%d (%s): System RAM (from sysinfo): MemTotal=%lu MB, MemFree=%lu MB, Available=%lu MB",
			       dev_id, dev_name,
			       (unsigned long)(mem_total / (1024 * 1024)),
			       (unsigned long)(mem_free / (1024 * 1024)),
			       (unsigned long)(mem_available / (1024 * 1024)));
		} else {
			applog(LOG_ERR, "GPU #%d (%s): Failed to get system memory information", dev_id, dev_name);
			pthread_mutex_unlock(&cache_mutex);
			return 0;
		}
	}
	
	// Count number of enabled GPU threads (opt_n_threads)
	// Note: opt_n_threads represents the number of enabled GPU threads
	int num_gpus = opt_n_threads;
	if (num_gpus <= 0) {
		// Fallback: use cuda_num_devices() if opt_n_threads not yet initialized
		num_gpus = cuda_num_devices();
		if (num_gpus <= 0) {
			applog(LOG_ERR, "GPU #%d (%s): No GPUs detected, cannot distribute system RAM", dev_id, dev_name);
			pthread_mutex_unlock(&cache_mutex);
			return 0;
		}
		applog(LOG_DEBUG, "GPU #%d (%s): opt_n_threads not yet initialized, using cuda_num_devices() count: %d", dev_id, dev_name, num_gpus);
	}
	
	// Reserve system RAM if --use-system-ram is enabled and --reserve-ram > 0
	if (opt_use_system_ram && opt_reserve_ram > 0) {
		size_t reserve_bytes = (size_t)opt_reserve_ram * 1024ULL * 1024ULL;
		if (mem_available > reserve_bytes) {
			mem_available -= reserve_bytes;
			applog(LOG_INFO, "GPU #%d (%s): Reserving %d MB system RAM, remaining system RAM: %lu MB",
			       dev_id, dev_name, opt_reserve_ram, (unsigned long)(mem_available / (1024 * 1024)));
		} else {
			applog(LOG_WARNING, "GPU #%d (%s): Requested reserve (%d MB) exceeds available system RAM (%lu MB), using all available",
			       dev_id, dev_name, opt_reserve_ram, (unsigned long)(mem_available / (1024 * 1024)));
			mem_available = 0;
		}
	}
	
	// Distribute available RAM equally among all enabled GPUs
	size_t ram_per_gpu = (size_t)(mem_available / num_gpus);
	applog(LOG_INFO, "GPU #%d (%s): Distributing system RAM: %zu MB per GPU (%d enabled GPU thread(s) total)",
	       dev_id, dev_name, ram_per_gpu / (1024 * 1024), num_gpus);
	
	// Cache the calculated value
	cached_ram_per_gpu = ram_per_gpu;
	cached_value_valid = true;
	
	// Release lock before returning
	pthread_mutex_unlock(&cache_mutex);
	
	return ram_per_gpu;
}

// Query and log GPU memory information at runtime
static void log_gpu_memory_info(int thr_id, const char *prefix_format, ...)
{
	char prefix[256];
	va_list args;
	va_start(args, prefix_format);
	vsnprintf(prefix, sizeof(prefix), prefix_format, args);
	va_end(args);

	size_t free_mem = 0, total_mem = 0;
	cudaError_t mem_err = cudaMemGetInfo(&free_mem, &total_mem);
	int dev_id = device_map[thr_id];
	const char* dev_name = device_name[dev_id] ? device_name[dev_id] : "Unknown";
	if (mem_err == cudaSuccess) {
		double free_mb = (double)free_mem / (1024.0 * 1024.0);
		double total_mb = (double)total_mem / (1024.0 * 1024.0);
		double used_mb = total_mb - free_mb;
		applog(LOG_INFO, "GPU #%d (%s): %sMemory - Total: %.2f MB, Available: %.2f MB, Used: %.2f MB (%.1f%%)",
			dev_id, dev_name, prefix, total_mb, free_mb, used_mb, (used_mb / total_mb) * 100.0);
	} else {
		applog(LOG_WARNING, "GPU #%d (%s): %sFailed to query memory info: %s", dev_id, dev_name, prefix, cudaGetErrorString(mem_err));
	}
}

int find_optimal_blockcount(int thr_id, KernelInterface* &kernel, bool &concurrent, int &WARPS_PER_BLOCK)
{
	int cw = console_width();
	int optimal_blocks = 0;

	cudaDeviceProp props;
	checkCudaErrors(cudaGetDeviceProperties(&props, device_map[thr_id]));
	concurrent = (props.concurrentKernels > 0);

	WARPS_PER_BLOCK = -1;

	// if not specified, use interactive mode for devices that have the watchdog timer enabled
	if (device_interactive[thr_id] == -1)
		device_interactive[thr_id] = props.kernelExecTimeoutEnabled;

	// figure out which kernel implementation to use
	bool kernel_auto_selected = false;
	if (!validate_config(device_config[thr_id], optimal_blocks, WARPS_PER_BLOCK, &kernel, &props)) {
		kernel = NULL;
		if (device_config[thr_id] != NULL) {
			if (device_config[thr_id][0] == 'V' || device_config[thr_id][0] == 'T' || device_config[thr_id][0] == 'v')
				kernel = new VoltaKernel();
			else if (device_config[thr_id][0] == 'P' || device_config[thr_id][0] == 't' || device_config[thr_id][0] == 'p')
				kernel = new PascalKernel();
		}
		if (kernel == NULL) {
			kernel = Best_Kernel_Heuristics(&props);
			kernel_auto_selected = true;
		}
	}

	if (kernel->get_major_version() > props.major || kernel->get_major_version() == props.major && kernel->get_minor_version() > props.minor)
	{
		applog(LOG_ERR, "GPU #%d: FATAL: the '%c' kernel requires %d.%d capability!", device_map[thr_id], kernel->get_identifier(), kernel->get_major_version(), kernel->get_minor_version());
		return 0;
	}

	// Apply auto-selection logic for device_threads_per_warp and device_lookup_gap
	if (kernel_auto_selected) {
		char kernel_id = kernel->get_identifier();
		if (kernel_id == 'V') {
			// VoltaKernel: set device_threads_per_warp to 16 if -1, set device_lookup_gap to 64 if 1
			if (device_threads_per_warp[thr_id] == -1)
				device_threads_per_warp[thr_id] = 16;
			if (device_lookup_gap[thr_id] == 1)
				device_lookup_gap[thr_id] = 64;
		} else if (kernel_id == 'P') {
			// PascalKernel: set device_threads_per_warp to 32 if -1, set device_lookup_gap to 64 if 1
			if (device_threads_per_warp[thr_id] == -1)
				device_threads_per_warp[thr_id] = 32;
			if (device_lookup_gap[thr_id] == 1)
				device_lookup_gap[thr_id] = 64;
		}
	}

	// Ensure device_threads_per_warp is set based on kernel if still -1
	if (device_threads_per_warp[thr_id] == -1) {
		char kernel_id = kernel->get_identifier();
		if (kernel_id == 'V') {
			device_threads_per_warp[thr_id] = 16;
		} else if (kernel_id == 'P') {
			device_threads_per_warp[thr_id] = 32;
		} else {
			// fallback to 32 for unknown kernels
			device_threads_per_warp[thr_id] = 32;
		}
	}

	// set whatever cache configuration and shared memory bank mode the kernel prefers
	checkCudaErrors(cudaDeviceSetCacheConfig(kernel->cache_config()));
	checkCudaErrors(cudaDeviceSetSharedMemConfig(kernel->shared_mem_config()));

	if (device_lookup_gap[thr_id] == 0) device_lookup_gap[thr_id] = 1;
	if (!kernel->support_lookup_gap() && device_lookup_gap[thr_id] > 1)
	{
		applog(LOG_WARNING, "GPU #%d: the '%c' kernel does not support a lookup gap", device_map[thr_id], kernel->get_identifier());
		device_lookup_gap[thr_id] = 1;
	}

	if (opt_debug) {
		applog(LOG_INFO, "GPU #%d: interactive: %d", device_map[thr_id],
		   (device_interactive[thr_id]  != 0) ? 1 : 0);
	}

	// number of threads collaborating on one work unit (hash)
	unsigned int THREADS_PER_WU = kernel->threads_per_wu();
	unsigned int THREADS_PER_WARP = device_threads_per_warp[thr_id];
	unsigned int WU_PER_WARP = THREADS_PER_WARP / THREADS_PER_WU;
	unsigned int LOOKUP_GAP = device_lookup_gap[thr_id];
	unsigned int BACKOFF = device_backoff[thr_id];
	unsigned int N = (1 << (opt_nfactor+1));
	double szPerWarp = (double)(SCRATCH * WU_PER_WARP * sizeof(uint32_t));
	//applog(LOG_INFO, "WU_PER_WARP=%u, THREADS_PER_WU=%u, LOOKUP_GAP=%u, BACKOFF=%u, SCRATCH=%u", WU_PER_WARP, THREADS_PER_WU, LOOKUP_GAP, BACKOFF, SCRATCH);
	int dev_id = device_map[thr_id];
	const char* dev_name = device_name[dev_id] ? device_name[dev_id] : "Unknown";
	applog(LOG_INFO, "GPU #%d (%s): %d hashes / %.1f MB per warp (size=%d, lookup_gap=%d).", dev_id, dev_name, WU_PER_WARP, szPerWarp / (1024.0 * 1024.0), THREADS_PER_WARP, LOOKUP_GAP);

	uint32_t *d_V = NULL;
	// Determine MAXWARPS based on remaining available GPU memory and szPerWarp
	size_t free_mem = 0, total_mem = 0;
	cudaError_t mem_err = cudaMemGetInfo(&free_mem, &total_mem);
	if (mem_err == cudaSuccess) {
		size_t reserve_bytes = (opt_reserve_vram > 0) ? ((size_t)opt_reserve_vram * 1024ULL * 1024ULL) : 0ULL;
		size_t available_mem = (free_mem > reserve_bytes) ? (free_mem - reserve_bytes) : 0;
		MAXWARPS[thr_id] = min((int)(available_mem / szPerWarp), TOTAL_WARP_LIMIT);
		applog(LOG_INFO, "GPU #%d (%s): Total: %.2f MB, Available: %.2f MB, Calculated MAXWARPS: %d",
			dev_id, dev_name, (double)total_mem / (1024.0 * 1024.0), (double)free_mem / (1024.0 * 1024.0), MAXWARPS[thr_id]);
	} else {
		MAXWARPS[thr_id] = TOTAL_WARP_LIMIT;
		applog(LOG_WARNING, "GPU #%d: Cannot determine MAXWARPS from memory info, using TOTAL_WARP_LIMIT: %s",
			device_map[thr_id], cudaGetErrorString(mem_err));
	}

	// chunked memory allocation up to device limits
	int warp;
	for (warp = 0; warp < MAXWARPS[thr_id]; ++warp) {
		// work around partition camping problems by adding a random start address offset to each allocation
		h_V_extra[thr_id][warp] = (props.major == 1) ? (16 * (rand()%(16384/16))) : 0;
		cudaGetLastError(); // clear the error state
		cudaMalloc((void **) &h_V[thr_id][warp], (SCRATCH * WU_PER_WARP + h_V_extra[thr_id][warp])*sizeof(uint32_t));
		if (cudaGetLastError() == cudaSuccess) h_V[thr_id][warp] += h_V_extra[thr_id][warp];
		else {
			applog(LOG_WARNING, "GPU #%d: Failed to allocate memory for warp %d, back off by 1 warp", device_map[thr_id], warp);
			h_V_extra[thr_id][warp] = 0;
			// Just back off by 1 warp
			warp--;
			checkCudaErrors(cudaFree(h_V[thr_id][warp]-h_V_extra[thr_id][warp]));
			h_V[thr_id][warp] = NULL; h_V_extra[thr_id][warp] = 0;
			break;
			// // back off by several warp allocations to have some breathing room
			// int remove = (BACKOFF*warp+50)/100;
			// for (int i=0; warp > 0 && i < remove; ++i) {
			// 	warp--;
			// 	checkCudaErrors(cudaFree(h_V[thr_id][warp]-h_V_extra[thr_id][warp]));
			// 	h_V[thr_id][warp] = NULL; h_V_extra[thr_id][warp] = 0;
			// }

		}
	}
	MAXWARPS[thr_id] = warp;
	applog(LOG_INFO, "GPU #%d (%s): Actual MAXWARPS: %d", dev_id, dev_name, MAXWARPS[thr_id]);
	log_gpu_memory_info(thr_id, "After VRAM warp allocation: ");
	
	// Initialize MAXWARPS_RAM to 0
	MAXWARPS_RAM[thr_id] = 0;
	
	// Allocate additional h_V buffers in system RAM if enabled
	// These will be stored in h_V starting at index MAXWARPS[thr_id]
	if (opt_use_system_ram) {
		// Check if device supports mapped host memory
		if (!props.canMapHostMemory) {
			applog(LOG_WARNING, "GPU #%d (%s): Device does not support mapped host memory, system RAM buffers disabled", 
				dev_id, dev_name);
			MAXWARPS_RAM[thr_id] = 0;
		} else {
			size_t available_system_ram = get_available_system_ram_per_gpu(dev_id, dev_name);
			if (available_system_ram > 0) {
				// Calculate MAXWARPS_RAM based on available system RAM
				int max_warps_ram_calc = min((int)(available_system_ram / szPerWarp), TOTAL_WARP_LIMIT);
				applog(LOG_INFO, "GPU #%d (%s): Calculated MAXWARPS_RAM: %d (%.1f MB available system RAM)",
					dev_id, dev_name, max_warps_ram_calc, (double)available_system_ram / (1024.0 * 1024.0));
				
				// Allocate additional h_V buffers in system RAM using mapped pinned host memory
				// Store them in h_V starting at index MAXWARPS[thr_id]
				// Note: h_V will contain device pointers for mapped memory
				int warp_ram;
				for (warp_ram = 0; warp_ram < max_warps_ram_calc; ++warp_ram) {
					int h_V_index = MAXWARPS[thr_id] + warp_ram;
					h_V[thr_id][h_V_index] = NULL; // Initialize to NULL
					
					// Allocate mapped host memory (accessible from both host and device)
					void *host_ptr = NULL;
					cudaError_t err = cudaHostAlloc(&host_ptr, 
						(SCRATCH * WU_PER_WARP) * sizeof(uint32_t), 
						cudaHostAllocMapped);
					if (err != cudaSuccess) {
						applog(LOG_WARNING, "GPU #%d: Failed to allocate mapped system RAM for warp %d: %s", 
							device_map[thr_id], warp_ram, cudaGetErrorString(err));
						// Free any already allocated buffers
						for (int i = 0; i < warp_ram; ++i) {
							int idx = MAXWARPS[thr_id] + i;
							if (h_V[thr_id][idx]) {
								cudaFreeHost(h_V[thr_id][idx]);
								h_V[thr_id][idx] = NULL;
							}
						}
						break;
					}
					
					// Get device pointer for the mapped host memory
					uint32_t *device_ptr = NULL;
					err = cudaHostGetDevicePointer((void **)&device_ptr, host_ptr, 0);
					if (err != cudaSuccess) {
						applog(LOG_WARNING, "GPU #%d: Failed to get device pointer for mapped system RAM warp %d: %s", 
							device_map[thr_id], warp_ram, cudaGetErrorString(err));
						cudaFreeHost(host_ptr);
						// Free any already allocated buffers
						for (int i = 0; i < warp_ram; ++i) {
							int idx = MAXWARPS[thr_id] + i;
							if (h_V[thr_id][idx]) {
								cudaFreeHost(h_V[thr_id][idx]);
								h_V[thr_id][idx] = NULL;
							}
						}
						break;
					}
					
					// Store device pointer in h_V (kernels will use this)
					h_V[thr_id][h_V_index] = device_ptr;
				}
				MAXWARPS_RAM[thr_id] = warp_ram;
				applog(LOG_INFO, "GPU #%d (%s): Actual MAXWARPS_RAM: %d", dev_id, dev_name, MAXWARPS_RAM[thr_id]);
				
				if (MAXWARPS_RAM[thr_id] > 0) {
					applog(LOG_INFO, "GPU #%d (%s): Total warps (VRAM + System RAM): %d + %d = %d",
						dev_id, dev_name, MAXWARPS[thr_id], MAXWARPS_RAM[thr_id], 
						MAXWARPS[thr_id] + MAXWARPS_RAM[thr_id]);
				}
			} else {
				applog(LOG_WARNING, "GPU #%d: Failed to get available system RAM, system RAM buffers disabled", device_map[thr_id]);
				MAXWARPS_RAM[thr_id] = 0;
			}
		}
	} else {
		MAXWARPS_RAM[thr_id] = 0;
	}
	
	// Pass total number of warps (VRAM + System RAM) to kernel
	kernel->set_scratchbuf_constants(MAXWARPS[thr_id] + MAXWARPS_RAM[thr_id], h_V[thr_id]);

	if (validate_config(device_config[thr_id], optimal_blocks, WARPS_PER_BLOCK))
	{
		if (optimal_blocks * WARPS_PER_BLOCK > MAXWARPS[thr_id])
		{
			optimal_blocks = MAXWARPS[thr_id] / WARPS_PER_BLOCK;
			applog(LOG_WARNING, "GPU #%d: WARNING: Given launch config '%s' requires too much memory, adjust it to %dx%d.", device_map[thr_id], device_config[thr_id], optimal_blocks, WARPS_PER_BLOCK);
		}

		if (WARPS_PER_BLOCK > kernel->max_warps_per_block())
		{
			applog(LOG_ERR, "GPU #%d: FATAL: Given launch config '%s' exceeds warp limit for '%c' kernel.", device_map[thr_id], device_config[thr_id], kernel->get_identifier());
			return 0;
		}
	}
	else
	{
		if (device_config[thr_id] != NULL && strcasecmp("auto", device_config[thr_id]))
			applog(LOG_WARNING, "GPU #%d: Given launch config '%s' does not validate.", device_map[thr_id], device_config[thr_id]);

		// Heuristics to find a good kernel launch configuration
		// base the initial block estimate on the number of multiprocessors
		int device_cores = props.multiProcessorCount * _ConvertSMVer2Cores(props.major, props.minor);

		// defaults, in case nothing else is chosen below
		optimal_blocks = 4 * device_cores / WU_PER_WARP;

		if (optimal_blocks > MAXWARPS[thr_id] + MAXWARPS_RAM[thr_id])
		{
			WARPS_PER_BLOCK = 1;
			optimal_blocks = MAXWARPS[thr_id] + MAXWARPS_RAM[thr_id];
		} else {
			WARPS_PER_BLOCK = 2;
			optimal_blocks = optimal_blocks / 2;
		}

		if (WARPS_PER_BLOCK > kernel->max_warps_per_block()) {
			applog(LOG_ERR, "GPU #%d: FATAL: Given launch config '%s' exceeds warp limit for '%c' kernel.", device_map[thr_id], device_config[thr_id], kernel->get_identifier());
			return 0;
		}
	}

	applog(LOG_INFO, "GPU #%d (%s): using launch configuration %c%dx%d", dev_id, dev_name, kernel->get_identifier(), optimal_blocks, WARPS_PER_BLOCK);

	return optimal_blocks;
}

void cuda_scrypt_HtoD(int thr_id, uint32_t *X, int stream)
{
	unsigned int WU_PER_LAUNCH = context_wu_per_launch[thr_id];
	unsigned int mem_size = WU_PER_LAUNCH * sizeof(uint32_t) * 32;

	// copy host memory to device
	cudaMemcpyAsync(context_idata[stream][thr_id], X, mem_size, cudaMemcpyHostToDevice, context_streams[stream][thr_id]);
}

void cuda_scrypt_serialize(int thr_id, int stream)
{
	// if the device can concurrently execute multiple kernels, then we must
	// wait for the serialization event recorded by the other stream
	if (context_concurrent[thr_id] || device_interactive[thr_id])
		cudaStreamWaitEvent(context_streams[stream][thr_id], context_serialize[(stream+1)&1][thr_id], 0);
}

void cuda_scrypt_done(int thr_id, int stream)
{
	// record the serialization event in the current stream
	cudaEventRecord(context_serialize[stream][thr_id], context_streams[stream][thr_id]);
}

void cuda_scrypt_flush(int thr_id, int stream)
{
	// flush the work queue (required for WDDM drivers)
	cudaStreamSynchronize(context_streams[stream][thr_id]);
}

void cuda_scrypt_core(int thr_id, int stream, unsigned int N)
{
	unsigned int WARPS_PER_BLOCK = context_wpb[thr_id];
	unsigned int WU_PER_BLOCK = context_wu_per_block[thr_id];
	unsigned int WU_PER_LAUNCH = context_wu_per_launch[thr_id];
	unsigned int THREADS_PER_WU = context_kernel[thr_id]->threads_per_wu();
	unsigned int LOOKUP_GAP = device_lookup_gap[thr_id];

	// setup execution parameters
	// Use WU_PER_BLOCK for configurable warp size (16 or 32)
	dim3 grid(WU_PER_LAUNCH/WU_PER_BLOCK, 1, 1);
	dim3 threads(THREADS_PER_WU*WU_PER_BLOCK, 1, 1);

		context_kernel[thr_id]->run_kernel(grid, threads, WARPS_PER_BLOCK, thr_id,
		context_streams[stream][thr_id], context_idata[stream][thr_id], context_odata[stream][thr_id],
		N, LOOKUP_GAP, device_interactive[thr_id], opt_benchmark);
}

void cuda_scrypt_DtoH(int thr_id, uint32_t *X, int stream, bool postSHA)
{
	unsigned int WU_PER_LAUNCH = context_wu_per_launch[thr_id];
	unsigned int mem_size = WU_PER_LAUNCH * sizeof(uint32_t) * (postSHA ? 8 : 32);
	// copy result from device to host (asynchronously)
	checkCudaErrors(cudaMemcpyAsync(X, postSHA ? context_hash[stream][thr_id] : context_odata[stream][thr_id], mem_size, cudaMemcpyDeviceToHost, context_streams[stream][thr_id]));
}

bool cuda_scrypt_sync(int thr_id, int stream)
{
	cudaError_t err;
	uint32_t wait_us = 0;

	if (device_interactive[thr_id] && !opt_benchmark)
	{
		// For devices that also do desktop rendering or compositing, we want to free up some time slots.
		// That requires making a pause in work submission when there is no active task on the GPU,
		// and Device Synchronize ensures that.

		// this call was replaced by the loop below to workaround the high CPU usage issue
		//err = cudaDeviceSynchronize();

		while((err = cudaStreamQuery(context_streams[0][thr_id])) == cudaErrorNotReady ||
			  (err == cudaSuccess && (err = cudaStreamQuery(context_streams[1][thr_id])) == cudaErrorNotReady)) {
			usleep(50); wait_us+=50;
		}

		usleep(50); wait_us+=50;
	} else {
		// this call was replaced by the loop below to workaround the high CPU usage issue
		//err = cudaStreamSynchronize(context_streams[stream][thr_id]);

		while((err = cudaStreamQuery(context_streams[stream][thr_id])) == cudaErrorNotReady) {
			usleep(50); wait_us+=50;
		}
	}

	if (err != cudaSuccess) {
		if (!abort_flag)
			applog(LOG_ERR, "GPU #%d: CUDA error `%s` while waiting the kernel.", device_map[thr_id], cudaGetErrorString(err));
		return false;
	}

	//if (opt_debug) {
	//	applog(LOG_DEBUG, "GPU #%d: %s %u us", device_map[thr_id], __FUNCTION__, wait_us);
	//}

	return true;
}

uint32_t* cuda_transferbuffer(int thr_id, int stream)
{
	return context_X[stream][thr_id];
}

uint32_t* cuda_hashbuffer(int thr_id, int stream)
{
	return context_H[stream][thr_id];
}
