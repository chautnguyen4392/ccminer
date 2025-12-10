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
#include "cuda_helper.h"

#include "salsa_kernel.h"

#include "nv_kernel2.h"
#include "titan_kernel.h"
#include "nv_kernel.h"
#include "kepler_kernel.h"
#include "fermi_kernel.h"
#include "test_kernel.h"

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
uint32_t* h_V[MAX_GPUS][TOTAL_WARP_LIMIT*64];          // NOTE: the *64 prevents buffer overflow for --keccak
uint32_t  h_V_extra[MAX_GPUS][TOTAL_WARP_LIMIT*64];    //       with really large kernel launch configurations

KernelInterface *Best_Kernel_Heuristics(cudaDeviceProp *props)
{
	KernelInterface *kernel = NULL;
	uint64_t N = 1UL << (opt_nfactor+1);

	if (N <= 8192)
	{
		// low N-factor scrypt-jane = high register count kernels
		if (props->major > 3 || (props->major == 3 && props->minor >= 5))
			kernel = new NV2Kernel(); // we don't want this for Keccak though
		else if (props->major == 3 && props->minor == 0)
			kernel = new NVKernel();
		else
			kernel = new FermiKernel();
	}
	else
	{
		// high N-factor scrypt-jane = low register count kernels
		if (props->major > 3 || (props->major == 3 && props->minor >= 5))
			kernel = new TitanKernel();
		else if (props->major == 3 && props->minor == 0)
			kernel = new KeplerKernel();
		else
			kernel = new TestKernel();
	}
	return kernel;
}


bool validate_config(char *config, int &b, int &w, KernelInterface **kernel = NULL, cudaDeviceProp *props = NULL)
{
	bool success = false;
	char kernelid = ' ';
	if (config != NULL)
	{
		if (config[0] == 'T' || config[0] == 'K' || config[0] == 'F' || config[0] == 'L' ||
			config[0] == 't' || config[0] == 'k' || config[0] == 'f' ||
			config[0] == 'Z' || config[0] == 'Y' || config[0] == 'X') {
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
				case 'T': case 'Z': *kernel = new NV2Kernel(); break;
				case 't':           *kernel = new TitanKernel(); break;
				case 'K': case 'Y': *kernel = new NVKernel(); break;
				case 'k':           *kernel = new KeplerKernel(); break;
				case 'F': case 'L': *kernel = new FermiKernel(); break;
				case 'f': case 'X': *kernel = new TestKernel(); break;
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
		checkCudaErrors(cudaSetDeviceFlags(cudaDeviceScheduleYield));

		KernelInterface *kernel;
		bool concurrent;
		GRID_BLOCKS = find_optimal_blockcount(thr_id, kernel, concurrent, WARPS_PER_BLOCK);

		if(GRID_BLOCKS == 0)
			return 0;

		unsigned int THREADS_PER_WU = kernel->threads_per_wu();
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

		context_kernel[thr_id] = kernel;
		context_concurrent[thr_id] = concurrent;
		context_blocks[thr_id] = GRID_BLOCKS;
		context_wpb[thr_id] = WARPS_PER_BLOCK;
	}

	GRID_BLOCKS = context_blocks[thr_id];
	WARPS_PER_BLOCK = context_wpb[thr_id];
	unsigned int THREADS_PER_WU = context_kernel[thr_id]->threads_per_wu();
	return WU_PER_LAUNCH;
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
	if (mem_err == cudaSuccess) {
		double free_mb = (double)free_mem / (1024.0 * 1024.0);
		double total_mb = (double)total_mem / (1024.0 * 1024.0);
		double used_mb = total_mb - free_mb;
		applog(LOG_INFO, "GPU #%d: %sMemory - Total: %.2f MB, Available: %.2f MB, Used: %.2f MB (%.1f%%)",
			device_map[thr_id], prefix, total_mb, free_mb, used_mb, (used_mb / total_mb) * 100.0);
	} else {
		applog(LOG_WARNING, "GPU #%d: %sFailed to query memory info: %s", device_map[thr_id], prefix, cudaGetErrorString(mem_err));
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
	if (!validate_config(device_config[thr_id], optimal_blocks, WARPS_PER_BLOCK, &kernel, &props)) {
		kernel = NULL;
		if (device_config[thr_id] != NULL) {
				 if (device_config[thr_id][0] == 'T' || device_config[thr_id][0] == 'Z')
				kernel = new NV2Kernel();
			else if (device_config[thr_id][0] == 't')
				kernel = new TitanKernel();
			else if (device_config[thr_id][0] == 'K' || device_config[thr_id][0] == 'Y')
				kernel = new NVKernel();
			else if (device_config[thr_id][0] == 'k')
				kernel = new KeplerKernel();
			else if (device_config[thr_id][0] == 'F' || device_config[thr_id][0] == 'L')
				kernel = new FermiKernel();
			else if (device_config[thr_id][0] == 'f' || device_config[thr_id][0] == 'X')
				kernel = new TestKernel();
		}
		if (kernel == NULL) kernel = Best_Kernel_Heuristics(&props);
	}

	if (kernel->get_major_version() > props.major || kernel->get_major_version() == props.major && kernel->get_minor_version() > props.minor)
	{
		applog(LOG_ERR, "GPU #%d: FATAL: the '%c' kernel requires %d.%d capability!", device_map[thr_id], kernel->get_identifier(), kernel->get_major_version(), kernel->get_minor_version());
		return 0;
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
	unsigned int LOOKUP_GAP = device_lookup_gap[thr_id];
	unsigned int BACKOFF = device_backoff[thr_id];
	unsigned int N = (1 << (opt_nfactor+1));
	double szPerWarp = (double)(SCRATCH * WU_PER_WARP * sizeof(uint32_t));
	//applog(LOG_INFO, "WU_PER_WARP=%u, THREADS_PER_WU=%u, LOOKUP_GAP=%u, BACKOFF=%u, SCRATCH=%u", WU_PER_WARP, THREADS_PER_WU, LOOKUP_GAP, BACKOFF, SCRATCH);
	applog(LOG_INFO, "GPU #%d: %d hashes / %.1f MB per warp.", device_map[thr_id], WU_PER_WARP, szPerWarp / (1024.0 * 1024.0));

	// compute highest MAXWARPS numbers for kernels allowing cudaBindTexture to succeed
	int MW_1D_4 = 134217728 / (SCRATCH * WU_PER_WARP / 4); // for uint4_t textures
	int MW_1D_2 = 134217728 / (SCRATCH * WU_PER_WARP / 2); // for uint2_t textures
	int MW_1D = kernel->get_texel_width() == 2 ? MW_1D_2 : MW_1D_4;

	uint32_t *d_V = NULL;
	// Determine MAXWARPS based on remaining available GPU memory and szPerWarp (reserve 50 MB)
	size_t free_mem = 0, total_mem = 0;
	cudaError_t mem_err = cudaMemGetInfo(&free_mem, &total_mem);
	if (mem_err == cudaSuccess) {
		size_t available_mem = (free_mem > 52428800ULL) ? (free_mem - 52428800ULL) : 0;
		MAXWARPS[thr_id] = min((int)(available_mem / szPerWarp), TOTAL_WARP_LIMIT);
		applog(LOG_INFO, "GPU #%d: Total: %.2f MB, Available: %.2f MB, Calculated MAXWARPS: %d",
			device_map[thr_id], (double)total_mem / (1024.0 * 1024.0), (double)free_mem / (1024.0 * 1024.0), MAXWARPS[thr_id]);
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
	applog(LOG_INFO, "GPU #%d: Actual MAXWARPS: %d", device_map[thr_id], MAXWARPS[thr_id]);
	log_gpu_memory_info(thr_id, "After warp allocation: ");
	kernel->set_scratchbuf_constants(MAXWARPS[thr_id], h_V[thr_id]);

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
		WARPS_PER_BLOCK = 2;

		// Based on compute capability, pick a known good block x warp configuration.
		if (props.major >= 6 && props.minor >= 1)
		{
			optimal_blocks = MAXWARPS[thr_id];
			WARPS_PER_BLOCK = 1;
		}
		else if (props.major >= 3)
		{
			if (props.major == 3 && props.minor == 5) // GK110 (Tesla K20X, K20, GeForce GTX TITAN)
			{
				// TODO: what to do with Titan and Tesla K20(X)?
				// for now, do the same as for GTX 660Ti (2GB)
				optimal_blocks = (int)(optimal_blocks * 0.8809524);
				WARPS_PER_BLOCK = 2;
			}
			else // GK104, GK106, GK107 ...
			{
				if (MAXWARPS[thr_id] > (int)(optimal_blocks * 1.7261905) * 2)
				{
					// this results in 290x2 configuration on GTX 660Ti (3GB)
					// but it requires 3GB memory on the card!
					optimal_blocks = (int)(optimal_blocks * 1.7261905);
					WARPS_PER_BLOCK = 2;
				}
				else
				{
					// this results in 148x2 configuration on GTX 660Ti (2GB)
					optimal_blocks = (int)(optimal_blocks * 0.8809524);
					WARPS_PER_BLOCK = 2;
				}
			}
		}
		// 1st generation Fermi (compute 2.0) GF100, GF110
		else if (props.major == 2 && props.minor == 0)
		{
			// this results in a 60x4 configuration on GTX 570
			optimal_blocks = 4 * device_cores / WU_PER_WARP;
			WARPS_PER_BLOCK = 4;
		}
		// 2nd generation Fermi (compute 2.1) GF104,106,108,114,116
		else if (props.major == 2 && props.minor == 1)
		{
			// this results in a 56x2 configuration on GTX 460
			optimal_blocks = props.multiProcessorCount * 8;
			WARPS_PER_BLOCK = 2;
		}

		// in case we run out of memory with the automatically chosen configuration,
		// first back off with WARPS_PER_BLOCK, then reduce optimal_blocks.
		if (WARPS_PER_BLOCK==3 && optimal_blocks * WARPS_PER_BLOCK > MAXWARPS[thr_id])
			WARPS_PER_BLOCK = 2;
		while (optimal_blocks > 0 && optimal_blocks * WARPS_PER_BLOCK > MAXWARPS[thr_id])
			optimal_blocks--;
	}

	applog(LOG_INFO, "GPU #%d: using launch configuration %c%dx%d", device_map[thr_id], kernel->get_identifier(), optimal_blocks, WARPS_PER_BLOCK);

	// back off unnecessary memory allocations to have some breathing room
	while (MAXWARPS[thr_id] > 0 && MAXWARPS[thr_id] > optimal_blocks * WARPS_PER_BLOCK) {
		(MAXWARPS[thr_id])--;
		checkCudaErrors(cudaFree(h_V[thr_id][MAXWARPS[thr_id]]-h_V_extra[thr_id][MAXWARPS[thr_id]]));
		h_V[thr_id][MAXWARPS[thr_id]] = NULL; h_V_extra[thr_id][MAXWARPS[thr_id]] = 0;
	}

	return optimal_blocks;
}

void cuda_scrypt_HtoD(int thr_id, uint32_t *X, int stream)
{
	unsigned int GRID_BLOCKS = context_blocks[thr_id];
	unsigned int WARPS_PER_BLOCK = context_wpb[thr_id];
	unsigned int THREADS_PER_WU = context_kernel[thr_id]->threads_per_wu();
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
	unsigned int GRID_BLOCKS = context_blocks[thr_id];
	unsigned int WARPS_PER_BLOCK = context_wpb[thr_id];
	unsigned int THREADS_PER_WU = context_kernel[thr_id]->threads_per_wu();
	unsigned int LOOKUP_GAP = device_lookup_gap[thr_id];

	// setup execution parameters
	dim3 grid(WU_PER_LAUNCH/WU_PER_BLOCK, 1, 1);
	dim3 threads(THREADS_PER_WU*WU_PER_BLOCK, 1, 1);

		context_kernel[thr_id]->run_kernel(grid, threads, WARPS_PER_BLOCK, thr_id,
		context_streams[stream][thr_id], context_idata[stream][thr_id], context_odata[stream][thr_id],
		N, LOOKUP_GAP, device_interactive[thr_id], opt_benchmark);
}

void cuda_scrypt_DtoH(int thr_id, uint32_t *X, int stream, bool postSHA)
{
	unsigned int GRID_BLOCKS = context_blocks[thr_id];
	unsigned int WARPS_PER_BLOCK = context_wpb[thr_id];
	unsigned int THREADS_PER_WU = context_kernel[thr_id]->threads_per_wu();
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
