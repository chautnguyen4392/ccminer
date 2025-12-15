#ifndef SALSA_KERNEL_H
#define SALSA_KERNEL_H

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#ifndef __APPLE__
#include <malloc.h>
#endif
#include <string.h>
#include <cuda_runtime.h>

#include "miner.h"

// from ccminer.cpp
extern short device_map[MAX_GPUS];
extern int device_batchsize[MAX_GPUS]; // cudaminer -b
extern int device_interactive[MAX_GPUS]; // cudaminer -i
extern int device_lookup_gap[MAX_GPUS]; // -L
extern int device_backoff[MAX_GPUS]; // WIN32/LINUX var
extern char *device_config[MAX_GPUS]; // -l
extern char *device_name[MAX_GPUS];

extern int opt_nfactor;
extern char *jane_params;

typedef unsigned int uint32_t; // define this as 32 bit type derived from int

// scrypt variant (only scrypt-jane is supported)
#define A_SCRYPT_JANE 1

// CUDA externals
extern int cuda_throughput(int thr_id);
extern uint32_t *cuda_transferbuffer(int thr_id, int stream);
extern uint32_t *cuda_hashbuffer(int thr_id, int stream);

extern void cuda_scrypt_HtoD(int thr_id, uint32_t *X, int stream);
extern void cuda_scrypt_serialize(int thr_id, int stream);
extern void cuda_scrypt_core(int thr_id, int stream, unsigned int N);
extern void cuda_scrypt_done(int thr_id, int stream);
extern void cuda_scrypt_DtoH(int thr_id, uint32_t *X, int stream, bool postSHA);
extern bool cuda_scrypt_sync(int thr_id, int stream);
extern void cuda_scrypt_flush(int thr_id, int stream);

// If we're in C++ mode, we're either compiling .cu files or scrypt.cpp

#ifdef __NVCC__

/**
 * An pure virtual interface for a CUDA kernel implementation.
 * TODO: encapsulate the kernel launch parameters in some kind of wrapper.
 */
class KernelInterface
{
public:
	virtual void set_scratchbuf_constants(int MAXWARPS, uint32_t** h_V) = 0;
	virtual bool run_kernel(dim3 grid, dim3 threads, int WARPS_PER_BLOCK, int thr_id, cudaStream_t stream, uint32_t* d_idata, uint32_t* d_odata, unsigned int N, unsigned int LOOKUP_GAP, bool interactive, bool benchmark) = 0;

	virtual char get_identifier() = 0;
	virtual int get_major_version() { return 1; }
	virtual int get_minor_version() { return 0; }
	virtual int max_warps_per_block() = 0;
	virtual int get_texel_width() = 0;
	virtual bool single_memory() { return false; };
	virtual int threads_per_wu() { return 1; }
	virtual bool support_lookup_gap() { return false; }
	virtual cudaSharedMemConfig shared_mem_config() { return cudaSharedMemBankSizeDefault; }
	virtual cudaFuncCache cache_config() { return cudaFuncCachePreferNone; }
};

// Not performing error checking is actually bad, but...
#define checkCudaErrors(x) x
#define getLastCudaError(x)

#endif // #ifdef __NVCC__

// Define work unit size
// threads per warp for tunable granularity (can be 8, 16, 24, or 32)
// This allows tuning the work unit granularity for certain kernels (e.g., NV2Kernel, TitanKernel)
// Set via -DTHREADS_PER_WARP=N at compile time, defaults to 16
#ifndef THREADS_PER_WARP
#define THREADS_PER_WARP 16
#endif
#define TOTAL_WARP_LIMIT 4096
#define WU_PER_WARP (THREADS_PER_WARP / THREADS_PER_WU)
#define WU_PER_BLOCK (WU_PER_WARP*WARPS_PER_BLOCK)
#define WU_PER_LAUNCH (GRID_BLOCKS*WU_PER_BLOCK)

// make scratchpad size dependent on N, LOOKUP_GAP
#define SCRATCH   (((N+LOOKUP_GAP-1)/LOOKUP_GAP)*32)

#endif // #ifndef SALSA_KERNEL_H
