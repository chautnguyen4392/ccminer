/* Copyright (C) 2013 David G. Andersen. All rights reserved.
 * with modifications by Christian Buchner
 *
 * Use of this code is covered under the Apache 2.0 license, which
 * can be found in the file "LICENSE"
 */

//       attempt V.Volkov style ILP (factor 4)

#include <map>

#include <cuda_runtime.h>
#include <cuda_helper.h>
#include "miner.h"

#include "salsa_kernel.h"
#include "titan_kernel.h"

#define THREADS_PER_WU 4  // four threads per hash

#if __CUDA_ARCH__ < 320
	// Kepler (Compute 3.0)
	#define __ldg(x) (*(x))
#endif

#if CUDA_VERSION >= 9000 && __CUDA_ARCH__ >= 300
#define __shfl2(var, srcLane)  __shfl_sync(0xFFFFFFFFu, var, srcLane)
#else
#define __shfl2 __shfl
#endif

#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ >= 300

// scratchbuf constants (pointers to scratch buffer for each warp, i.e. 32 hashes)
__constant__ uint32_t* c_V[TOTAL_WARP_LIMIT];

// Shared memory copy of c_V for faster access (size is WARPS_PER_BLOCK per block)
extern __shared__ uint32_t* s_V[];

// iteration count N
__constant__ uint32_t c_N;
__constant__ uint32_t c_N_1;                   // N-1
// scratch buffer size SCRATCH
__constant__ uint32_t c_SCRATCH;
__constant__ uint32_t c_SCRATCH_WU_PER_WARP;   // (SCRATCH * WU_PER_WARP)
__constant__ uint32_t c_SCRATCH_WU_PER_WARP_1; // (SCRATCH * WU_PER_WARP)-1

static __device__ uint4& operator ^= (uint4& left, const uint4& right) {
	left.x ^= right.x;
	left.y ^= right.y;
	left.z ^= right.z;
	left.w ^= right.w;
	return left;
}

static __device__ uint4& operator += (uint4& left, const uint4& right) {
	left.x += right.x;
	left.y += right.y;
	left.z += right.z;
	left.w += right.w;
	return left;
}

/* write_keys writes the 8 keys being processed by a warp to the global
 * scratchpad. To effectively use memory bandwidth, it performs the writes
 * (and reads, for read_keys) 128 bytes at a time per memory location
 * by __shfl'ing the 4 entries in bx to the threads in the next-up
 * thread group. It then has eight threads together perform uint4
 * (128 bit) writes to the destination region. This seems to make
 * quite effective use of memory bandwidth. An approach that spread
 * uint32s across more threads was slower because of the increased
 * computation it required.
 *
 * "start" is the loop iteration producing the write - the offset within
 * the block's memory.
 *
 * Internally, this algorithm first __shfl's the 4 bx entries to
 * the next up thread group, and then uses a conditional move to
 * ensure that odd-numbered thread groups exchange the b/bx ordering
 * so that the right parts are written together.
 *
 * Thanks to Babu for helping design the 128-bit-per-write version.
 *
 * _direct lets the caller specify the absolute start location instead of
 * the relative start location, as an attempt to reduce some recomputation.
 */
 __device__ __forceinline__
 uint4 ldcg_uint4(const uint4* ptr)
 {
	 const uint32_t* p = reinterpret_cast<const uint32_t*>(ptr);
	 uint4 v;
 
	 v.x = __ldcg(p + 0);
	 v.y = __ldcg(p + 1);
	 v.z = __ldcg(p + 2);
	 v.w = __ldcg(p + 3);
	 return v;
 }

 __device__ __forceinline__
void stcg_uint4(uint4* ptr, const uint4 &v)
{
    uint32_t* p = reinterpret_cast<uint32_t*>(ptr);

    asm volatile ("st.cg.global.u32 [%0], %1;" :: "l"((unsigned long long)(p+0)), "r"(v.x));
    asm volatile ("st.cg.global.u32 [%0], %1;" :: "l"((unsigned long long)(p+1)), "r"(v.y));
    asm volatile ("st.cg.global.u32 [%0], %1;" :: "l"((unsigned long long)(p+2)), "r"(v.z));
    asm volatile ("st.cg.global.u32 [%0], %1;" :: "l"((unsigned long long)(p+3)), "r"(v.w));
}

__device__ __forceinline__
void write_keys_direct(const uint4 &b, const uint4 &bx, uint32_t start)
{
	uint32_t *scratch = s_V[threadIdx.x/THREADS_PER_WARP];
	stcg_uint4((uint4 *)(&scratch[start   ]), b);
	stcg_uint4((uint4 *)(&scratch[start+16]), bx);
}

__device__ __forceinline__
void read_keys_direct(uint4 &b, uint4 &bx, uint32_t start)
{
	uint32_t *scratch = s_V[threadIdx.x/THREADS_PER_WARP];
	// Use __ldg() for read-only cache optimization (Pascal+)
	b = ldcg_uint4((uint4 *)(&scratch[start]));
	bx = ldcg_uint4((uint4 *)(&scratch[start+16]));
}

/*
 * load_key loads a 32*32bit key from a contiguous region of memory in B.
 * The input keys are in external order (i.e., 0, 1, 2, 3, ...).
 * After loading, each thread has its four b and four bx keys stored
 * in internal processing order.
 */
__device__  __forceinline__
void load_key(const uint32_t *B, uint4 &b, uint4 &bx)
{
	uint32_t scrypt_block = (blockIdx.x*blockDim.x + threadIdx.x)/THREADS_PER_WU;
	uint32_t thread_in_block = threadIdx.x & 3U;
	uint32_t key_offset = scrypt_block * 32 + thread_in_block;

	// Read in permuted order. Key loads are not our bottleneck right now.
	b.x  = B[key_offset      ];
	b.y  = B[key_offset + 4*1];
	b.z  = B[key_offset + 4*2];
	b.w  = B[key_offset + 4*3];

	key_offset += 16;
	bx.x = B[key_offset      ];
	bx.y = B[key_offset + 4  ];
	bx.z = B[key_offset + 4*2];
	bx.w = B[key_offset + 4*3];
}

/*
 * store_key performs the opposite transform as load_key, taking
 * internally-ordered b and bx and storing them into a contiguous
 * region of B in external order.
 */
__device__  __forceinline__
void store_key(uint32_t *B, const uint4 &b, const uint4 &bx)
{
	uint32_t scrypt_block = (blockIdx.x*blockDim.x + threadIdx.x)/THREADS_PER_WU;
	uint32_t thread_in_block = threadIdx.x & 3U;
	uint32_t key_offset = scrypt_block * 32U + thread_in_block;

	B[key_offset      ] = b.x;
	B[key_offset + 4  ] = b.y;
	B[key_offset + 4*2] = b.z;
	B[key_offset + 4*3] = b.w;

	key_offset += 16;
	B[key_offset      ] = bx.x;
	B[key_offset + 4  ] = bx.y;
	B[key_offset + 4*2] = bx.z;
	B[key_offset + 4*3] = bx.w;
}

/*
 * chacha_xor_core (ChaCha20/8 cypher)
 * This version is unrolled to handle both of these loops in a single
 * call to avoid unnecessary data movement.
 *
 * load_key and store_key must not use primary order when
 * using ChaCha20/8, but rather the basic transposed order
 * (referred to as "column mode" below)
 */

#if __CUDA_ARCH__ < 320
	// Kepler (Compute 3.0)
	#define CHACHA_PRIMITIVE(pt, rt, ps, amt) { uint32_t tmp = rt ^ (pt += ps); rt = ((tmp<<amt)|(tmp>>(32-amt))); }
#else
	// Kepler (Compute 3.5)
	#define ROTL(a, b) __funnelshift_l( a, a, b );
	#define CHACHA_PRIMITIVE(pt, rt, ps, amt) { pt += ps; rt = ROTL(rt ^ pt,amt); }
#endif

__device__  __forceinline__
void chacha_xor_core(uint4 &b, uint4 &bx, const int x1, const int x2, const int x3)
{
	uint4 x = b ^= bx;

	//b ^= bx;
	//x = b;

	// Enter in "column" mode (t0 has 0, 4,  8, 12)
	//                        (t1 has 1, 5,  9, 13)
	//                        (t2 has 2, 6, 10, 14)
	//                        (t3 has 3, 7, 11, 15)

	#pragma unroll
	for (int j = 0; j < 4; j++) {

		// Column Mixing phase of chacha
		CHACHA_PRIMITIVE(x.x ,x.w, x.y, 16)
		CHACHA_PRIMITIVE(x.z ,x.y, x.w, 12)
		CHACHA_PRIMITIVE(x.x ,x.w, x.y,  8)
		CHACHA_PRIMITIVE(x.z ,x.y, x.w,  7)

		x.y = __shfl2((int)x.y, x1);
		x.z = __shfl2((int)x.z, x2);
		x.w = __shfl2((int)x.w, x3);

		// Diagonal Mixing phase of chacha
		CHACHA_PRIMITIVE(x.x ,x.w, x.y, 16)
		CHACHA_PRIMITIVE(x.z ,x.y, x.w, 12)
		CHACHA_PRIMITIVE(x.x ,x.w, x.y,  8)
		CHACHA_PRIMITIVE(x.z ,x.y, x.w,  7)

		x.y = __shfl2((int)x.y, x3);
		x.z = __shfl2((int)x.z, x2);
		x.w = __shfl2((int)x.w, x1);
	}

	b += x;
	// The next two lines are the beginning of the BX-centric loop iteration
	bx ^= b;
	x = bx;

	#pragma unroll
	for (int j = 0; j < 4; j++) 
	{

		// Column Mixing phase of chacha
		CHACHA_PRIMITIVE(x.x ,x.w, x.y, 16)
		CHACHA_PRIMITIVE(x.z ,x.y, x.w, 12)
		CHACHA_PRIMITIVE(x.x ,x.w, x.y,  8)
		CHACHA_PRIMITIVE(x.z ,x.y, x.w,  7)

		x.y = __shfl2((int)x.y, x1);
		x.z = __shfl2((int)x.z, x2);
		x.w = __shfl2((int)x.w, x3);

		// Diagonal Mixing phase of chacha
		CHACHA_PRIMITIVE(x.x ,x.w, x.y, 16)
		CHACHA_PRIMITIVE(x.z ,x.y, x.w, 12)
		CHACHA_PRIMITIVE(x.x ,x.w, x.y,  8)
		CHACHA_PRIMITIVE(x.z ,x.y, x.w,  7)

		x.y = __shfl2((int)x.y, x3);
		x.z = __shfl2((int)x.z, x2);
		x.w = __shfl2((int)x.w, x1);
	}

#undef CHACHA_PRIMITIVE

	bx += x;
}

/*
 * The hasher_gen_kernel operates on a group of 1024-bit input keys
 * in B, stored as:
 * B = { k1B k1Bx k2B k2Bx ... }
 * and fills up the scratchpad with the iterative hashes derived from
 * those keys:
 * scratch { k1h1B k1h1Bx K1h2B K1h2Bx ... K2h1B K2h1Bx K2h2B K2h2Bx ... }
 * scratch is 1024 times larger than the input keys B.
 * It is extremely important to stream writes effectively into scratch;
 * less important to coalesce the reads from B.
 *
 * Key ordering note: Keys are input from B in "original" order:
 * K = {k1, k2, k3, k4, k5, ..., kx15, kx16, kx17, ..., kx31 }
 * After inputting into kernel_gen, each component k and kx of the
 * key is transmuted into a permuted internal order to make processing faster:
 * K = k, kx with:
 * k = 0, 4, 8, 12, 5, 9, 13, 1, 10, 14, 2, 6, 15, 3, 7, 11
 * and similarly for kx.
 */

 __global__
 void titan_scrypt_core_kernelA_LG(const uint32_t *d_idata, int iterations, unsigned int LOOKUP_GAP)
 {
	 // Copy from constant memory c_V to shared memory s_V for this block's warps
	 int warp_id = threadIdx.x / THREADS_PER_WARP;
	 int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / THREADS_PER_WARP;
	 if (threadIdx.x % THREADS_PER_WARP == 0) {
		 s_V[warp_id] = c_V[global_warp_id];
	 }
	 __syncthreads();
 
	 uint4 b, bx;
 
	 int x1 = (threadIdx.x & 0x1c) + (((threadIdx.x & 0x03)+1)&0x3);
	 int x2 = (threadIdx.x & 0x1c) + (((threadIdx.x & 0x03)+2)&0x3);
	 int x3 = (threadIdx.x & 0x1c) + (((threadIdx.x & 0x03)+3)&0x3);
 
	 int scrypt_block = (blockIdx.x*blockDim.x + threadIdx.x)/THREADS_PER_WU;
	 int start = (scrypt_block*c_SCRATCH + 4*(threadIdx.x%4)) % c_SCRATCH_WU_PER_WARP;
 
	 if (iterations <= 0)
		 return;
 
	 load_key(d_idata, b, bx);
	 write_keys_direct(b, bx, start);
 
	 // Optimized loop: avoid checking i % LOOKUP_GAP on every iteration
	 int remaining = iterations - 1;
	 int full_blocks = remaining / LOOKUP_GAP;
	 int remainder = remaining % LOOKUP_GAP;
	 int write_idx = 1;  // First write will be at i=LOOKUP_GAP, so write_idx=1
 
	 // Process full LOOKUP_GAP blocks
	 for (int block = 0; block < full_blocks; block++) {
		 // Process LOOKUP_GAP iterations
		 #pragma unroll
		 for (int j = 0; j < LOOKUP_GAP; j++) {
			 chacha_xor_core(b, bx, x1, x2, x3);
		 }
		 // Write after processing this block
		 write_keys_direct(b, bx, start+32*write_idx);
		 write_idx++;
	 }
 
	 // Process remaining iterations
	 for (int j = 0; j < remainder; j++) {
		 chacha_xor_core(b, bx, x1, x2, x3);
	 }
 }


/*
 * hasher_hash_kernel runs the second phase of scrypt after the scratch
 * buffer is filled with the iterative hashes: It bounces through
 * the scratch buffer in pseudorandom order, mixing the key as it goes.
 */

__global__
void titan_scrypt_core_kernelB_LG(uint32_t *d_odata, int iterations, unsigned int LOOKUP_GAP)
{
	// Copy from constant memory c_V to shared memory s_V for this block's warps
	int warp_id = threadIdx.x / THREADS_PER_WARP;
	int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / THREADS_PER_WARP;
	if (threadIdx.x % THREADS_PER_WARP == 0) {
		s_V[warp_id] = c_V[global_warp_id];
	}
	__syncthreads();

	uint4 b, bx;

	int scrypt_block = (blockIdx.x*blockDim.x + threadIdx.x)/THREADS_PER_WU;
	int start = ((scrypt_block*c_SCRATCH) + 4*(threadIdx.x%4)) % c_SCRATCH_WU_PER_WARP;

	int x1 = (threadIdx.x & 0x1c) + (((threadIdx.x & 0x03)+1)&0x3);
	int x2 = (threadIdx.x & 0x1c) + (((threadIdx.x & 0x03)+2)&0x3);
	int x3 = (threadIdx.x & 0x1c) + (((threadIdx.x & 0x03)+3)&0x3);

	if (iterations <= 0)
		return;

	int resume_pos = c_N_1/LOOKUP_GAP;
	int resume_loop = 1 + (c_N_1 - resume_pos*LOOKUP_GAP);
	read_keys_direct(b, bx, start+32*resume_pos);
	while (resume_loop--)
		chacha_xor_core(b, bx, x1, x2, x3);

	int j = (__shfl2((int)bx.x, (threadIdx.x & 0x1c)) & (c_N_1));
	int scratch_pos = j/LOOKUP_GAP;
	int loop = -1;
	uint4 t, tx;

	int i = 0;
	while (i < iterations)
	{
		if (loop == -1) {
			j = (__shfl2((int)bx.x, (threadIdx.x & 0x1c)) & (c_N_1));
			scratch_pos = j/LOOKUP_GAP;
			loop = j - scratch_pos*LOOKUP_GAP;
			read_keys_direct(t, tx, start+32*scratch_pos);
		}
		if (loop == 0) {
			b ^= t; bx ^= tx;
			t = b; tx = bx;
		}

		chacha_xor_core(t, tx, x1, x2, x3);
		if (loop == 0) {
			b = t; bx = tx;
			i++;
		}
		loop--;
	}

	store_key(d_odata, b, bx);
}


TitanKernel::TitanKernel() : KernelInterface()
{
}

void TitanKernel::set_scratchbuf_constants(int MAXWARPS, uint32_t** h_V)
{
	checkCudaErrors(cudaMemcpyToSymbol(c_V, h_V, MAXWARPS*sizeof(uint32_t*), 0, cudaMemcpyHostToDevice));
}

bool TitanKernel::run_kernel(dim3 grid, dim3 threads, int WARPS_PER_BLOCK, int thr_id, cudaStream_t stream,
	uint32_t* d_idata, uint32_t* d_odata, unsigned int N, unsigned int LOOKUP_GAP, bool interactive, bool benchmark, int texture_cache)
{
	bool success = true;
	if (!IS_SCRYPT_JANE())
		return false;

	// make some constants available to kernel, update only initially and when changing
	static uint32_t prev_N[MAX_GPUS] = { 0 };

	if (N != prev_N[thr_id]) {
		uint32_t h_N = N;
		uint32_t h_N_1 = N-1;
		uint32_t h_SCRATCH = SCRATCH;
		uint32_t h_SCRATCH_WU_PER_WARP = (SCRATCH * WU_PER_WARP);
		uint32_t h_SCRATCH_WU_PER_WARP_1 = (SCRATCH * WU_PER_WARP) - 1;

		cudaMemcpyToSymbolAsync(c_N, &h_N, sizeof(uint32_t), 0, cudaMemcpyHostToDevice, stream);
		cudaMemcpyToSymbolAsync(c_N_1, &h_N_1, sizeof(uint32_t), 0, cudaMemcpyHostToDevice, stream);
		cudaMemcpyToSymbolAsync(c_SCRATCH, &h_SCRATCH, sizeof(uint32_t), 0, cudaMemcpyHostToDevice, stream);
		cudaMemcpyToSymbolAsync(c_SCRATCH_WU_PER_WARP, &h_SCRATCH_WU_PER_WARP, sizeof(uint32_t), 0, cudaMemcpyHostToDevice, stream);
		cudaMemcpyToSymbolAsync(c_SCRATCH_WU_PER_WARP_1, &h_SCRATCH_WU_PER_WARP_1, sizeof(uint32_t), 0, cudaMemcpyHostToDevice, stream);

		prev_N[thr_id] = N;
	}

	// Calculate shared memory size needed for pointer array (WARPS_PER_BLOCK pointers)
	size_t shared_mem_size = WARPS_PER_BLOCK * sizeof(uint32_t*);

	// First phase: Sequential writes to scratchpad.
	titan_scrypt_core_kernelA_LG <<< grid, threads, shared_mem_size, stream >>>(d_idata, N, LOOKUP_GAP);

	// Second phase: Random read access from scratchpad.
	titan_scrypt_core_kernelB_LG <<< grid, threads, shared_mem_size, stream >>>(d_odata, N, LOOKUP_GAP);

	return success;
}

#endif /* prevent SM 2 */
