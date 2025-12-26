//
// Kernel that runs best on Fermi devices
//
// - shared memory use reduced by nearly factor 2 over legacy kernel
//   by transferring only half work units (16 x uint32_t) at once.
// - uses ulong2/uint4 based memory transfers (each thread moves 16 bytes),
//   allowing for shorter unrolled loops. This relies on Fermi's better
//   memory controllers to get high memory troughput.
//
// NOTE: compile this .cu module for compute_20,sm_20 with --maxrregcount=63
//
// TODO: batch-size support for this kernel
//

#include <map>

#include <cuda_runtime.h>
#include "miner.h"

#include "salsa_kernel.h"
#include "fermi_kernel.h"

#define THREADS_PER_WU 1  // single thread per hash

// forward references
__global__ void fermi_scrypt_core_kernelA(uint32_t *g_idata, unsigned int N);
__global__ void fermi_scrypt_core_kernelB(uint32_t *g_odata, unsigned int N);
__global__ void fermi_scrypt_core_kernelA_LG(uint32_t *g_idata, unsigned int N, unsigned int LOOKUP_GAP);
__global__ void fermi_scrypt_core_kernelB_LG(uint32_t *g_odata, unsigned int N, unsigned int LOOKUP_GAP);

// scratchbuf constants (pointers to scratch buffer for each warp, i.e. 32 hashes)
__constant__ uint32_t* c_V[TOTAL_WARP_LIMIT];

FermiKernel::FermiKernel() : KernelInterface()
{
}

void FermiKernel::set_scratchbuf_constants(int MAXWARPS, uint32_t** h_V)
{
	checkCudaErrors(cudaMemcpyToSymbol(c_V, h_V, MAXWARPS*sizeof(uint32_t*), 0, cudaMemcpyHostToDevice));
}

bool FermiKernel::run_kernel(dim3 grid, dim3 threads, int WARPS_PER_BLOCK, int thr_id, cudaStream_t stream, uint32_t* d_idata, uint32_t* d_odata, unsigned int N, unsigned int LOOKUP_GAP, bool interactive, bool benchmark)
{
	bool success = true;

	int shared = WARPS_PER_BLOCK * WU_PER_WARP * (16+4) * sizeof(uint32_t);

	// First phase: Sequential writes to scratchpad.

	if (LOOKUP_GAP == 1) {
		fermi_scrypt_core_kernelA<<< grid, threads, shared, stream >>>(d_idata, N);
	} else {
		fermi_scrypt_core_kernelA_LG<<< grid, threads, shared, stream >>>(d_idata, N, LOOKUP_GAP);
	}

	// Second phase: Random read access from scratchpad.

	if (LOOKUP_GAP == 1) {
		fermi_scrypt_core_kernelB<<< grid, threads, shared, stream >>>(d_odata, N);
	} else {
		fermi_scrypt_core_kernelB_LG<<< grid, threads, shared, stream >>>(d_odata, N, LOOKUP_GAP);
	}

	return success;
}

#if 0

#define ROTL(a, b) (((a) << (b)) | ((a) >> (32 - (b))))

#define QUARTER(a,b,c,d) \
	a += b; d ^= a; d = ROTL(d,16); \
	c += d; b ^= c; b = ROTL(b,12); \
	a += b; d ^= a; d = ROTL(d,8); \
	c += d; b ^= c; b = ROTL(b,7);

static __device__ void xor_chacha8(uint4 *B, uint4 *C)
{
	uint32_t x[16];
	x[0]=(B[0].x ^= C[0].x);
	x[1]=(B[0].y ^= C[0].y);
	x[2]=(B[0].z ^= C[0].z);
	x[3]=(B[0].w ^= C[0].w);
	x[4]=(B[1].x ^= C[1].x);
	x[5]=(B[1].y ^= C[1].y);
	x[6]=(B[1].z ^= C[1].z);
	x[7]=(B[1].w ^= C[1].w);
	x[8]=(B[2].x ^= C[2].x);
	x[9]=(B[2].y ^= C[2].y);
	x[10]=(B[2].z ^= C[2].z);
	x[11]=(B[2].w ^= C[2].w);
	x[12]=(B[3].x ^= C[3].x);
	x[13]=(B[3].y ^= C[3].y);
	x[14]=(B[3].z ^= C[3].z);
	x[15]=(B[3].w ^= C[3].w);

	/* Operate on columns. */
	QUARTER( x[0], x[4], x[ 8], x[12] )
	QUARTER( x[1], x[5], x[ 9], x[13] )
	QUARTER( x[2], x[6], x[10], x[14] )
	QUARTER( x[3], x[7], x[11], x[15] )

	/* Operate on diagonals */
	QUARTER( x[0], x[5], x[10], x[15] )
	QUARTER( x[1], x[6], x[11], x[12] )
	QUARTER( x[2], x[7], x[ 8], x[13] )
	QUARTER( x[3], x[4], x[ 9], x[14] )

	/* Operate on columns. */
	QUARTER( x[0], x[4], x[ 8], x[12] )
	QUARTER( x[1], x[5], x[ 9], x[13] )
	QUARTER( x[2], x[6], x[10], x[14] )
	QUARTER( x[3], x[7], x[11], x[15] )

	/* Operate on diagonals */
	QUARTER( x[0], x[5], x[10], x[15] )
	QUARTER( x[1], x[6], x[11], x[12] )
	QUARTER( x[2], x[7], x[ 8], x[13] )
	QUARTER( x[3], x[4], x[ 9], x[14] )

	/* Operate on columns. */
	QUARTER( x[0], x[4], x[ 8], x[12] )
	QUARTER( x[1], x[5], x[ 9], x[13] )
	QUARTER( x[2], x[6], x[10], x[14] )
	QUARTER( x[3], x[7], x[11], x[15] )

	/* Operate on diagonals */
	QUARTER( x[0], x[5], x[10], x[15] )
	QUARTER( x[1], x[6], x[11], x[12] )
	QUARTER( x[2], x[7], x[ 8], x[13] )
	QUARTER( x[3], x[4], x[ 9], x[14] )

	/* Operate on columns. */
	QUARTER( x[0], x[4], x[ 8], x[12] )
	QUARTER( x[1], x[5], x[ 9], x[13] )
	QUARTER( x[2], x[6], x[10], x[14] )
	QUARTER( x[3], x[7], x[11], x[15] )

	/* Operate on diagonals */
	QUARTER( x[0], x[5], x[10], x[15] )
	QUARTER( x[1], x[6], x[11], x[12] )
	QUARTER( x[2], x[7], x[ 8], x[13] )
	QUARTER( x[3], x[4], x[ 9], x[14] )

	B[0].x += x[0]; B[0].y += x[1]; B[0].z += x[2];  B[0].w += x[3];  B[1].x += x[4];  B[1].y += x[5];  B[1].z += x[6];  B[1].w += x[7];
	B[2].x += x[8]; B[2].y += x[9]; B[2].z += x[10]; B[2].w += x[11]; B[3].x += x[12]; B[3].y += x[13]; B[3].z += x[14]; B[3].w += x[15];
}

#else

#define ROTL(a, b) (((a) << (b)) | ((a) >> (32 - (b))))

#define ADD4(d1,d2,d3,d4,s1,s2,s3,s4) \
	d1 += s1; d2 += s2; d3 += s3; d4 += s4;

#define XOR4(d1,d2,d3,d4,s1,s2,s3,s4) \
	d1 ^= s1; d2 ^= s2; d3 ^= s3; d4 ^= s4;

#define ROTL4(d1,d2,d3,d4,amt) \
	d1 = ROTL(d1, amt); d2 = ROTL(d2, amt); d3 = ROTL(d3, amt); d4 = ROTL(d4, amt);

#define QROUND(a1,a2,a3,a4, b1,b2,b3,b4, c1,c2,c3,c4, amt) \
	ADD4 (a1,a2,a3,a4, c1,c2,c3,c4) \
	XOR4 (b1,b2,b3,b4, a1,a2,a3,a4) \
	ROTL4(b1,b2,b3,b4, amt)

static __device__ void xor_chacha8(uint4 *B, uint4 *C)
{
	uint32_t x[16];
	x[0]=(B[0].x ^= C[0].x);
	x[1]=(B[0].y ^= C[0].y);
	x[2]=(B[0].z ^= C[0].z);
	x[3]=(B[0].w ^= C[0].w);
	x[4]=(B[1].x ^= C[1].x);
	x[5]=(B[1].y ^= C[1].y);
	x[6]=(B[1].z ^= C[1].z);
	x[7]=(B[1].w ^= C[1].w);
	x[8]=(B[2].x ^= C[2].x);
	x[9]=(B[2].y ^= C[2].y);
	x[10]=(B[2].z ^= C[2].z);
	x[11]=(B[2].w ^= C[2].w);
	x[12]=(B[3].x ^= C[3].x);
	x[13]=(B[3].y ^= C[3].y);
	x[14]=(B[3].z ^= C[3].z);
	x[15]=(B[3].w ^= C[3].w);

	/* Operate on columns. */
	QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[12],x[13],x[14],x[15], x[ 4],x[ 5],x[ 6],x[ 7], 16);
	QROUND(x[ 8],x[ 9],x[10],x[11], x[ 4],x[ 5],x[ 6],x[ 7], x[12],x[13],x[14],x[15], 12);
	QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[12],x[13],x[14],x[15], x[ 4],x[ 5],x[ 6],x[ 7],  8);
	QROUND(x[ 8],x[ 9],x[10],x[11], x[ 4],x[ 5],x[ 6],x[ 7], x[12],x[13],x[14],x[15],  7);

	/* Operate on diagonals */
	QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[15],x[12],x[13],x[14], x[ 5],x[ 6],x[ 7],x[ 4], 16);
	QROUND(x[10],x[11],x[ 8],x[ 9], x[ 5],x[ 6],x[ 7],x[ 4], x[15],x[12],x[13],x[14], 12);
	QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[15],x[12],x[13],x[14], x[ 5],x[ 6],x[ 7],x[ 4],  8);
	QROUND(x[10],x[11],x[ 8],x[ 9], x[ 5],x[ 6],x[ 7],x[ 4], x[15],x[12],x[13],x[14],  7);

	/* Operate on columns. */
	QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[12],x[13],x[14],x[15], x[ 4],x[ 5],x[ 6],x[ 7], 16);
	QROUND(x[ 8],x[ 9],x[10],x[11], x[ 4],x[ 5],x[ 6],x[ 7], x[12],x[13],x[14],x[15], 12);
	QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[12],x[13],x[14],x[15], x[ 4],x[ 5],x[ 6],x[ 7],  8);
	QROUND(x[ 8],x[ 9],x[10],x[11], x[ 4],x[ 5],x[ 6],x[ 7], x[12],x[13],x[14],x[15],  7);

	/* Operate on diagonals */
	QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[15],x[12],x[13],x[14], x[ 5],x[ 6],x[ 7],x[ 4], 16);
	QROUND(x[10],x[11],x[ 8],x[ 9], x[ 5],x[ 6],x[ 7],x[ 4], x[15],x[12],x[13],x[14], 12);
	QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[15],x[12],x[13],x[14], x[ 5],x[ 6],x[ 7],x[ 4],  8);
	QROUND(x[10],x[11],x[ 8],x[ 9], x[ 5],x[ 6],x[ 7],x[ 4], x[15],x[12],x[13],x[14],  7);

	/* Operate on columns. */
	QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[12],x[13],x[14],x[15], x[ 4],x[ 5],x[ 6],x[ 7], 16);
	QROUND(x[ 8],x[ 9],x[10],x[11], x[ 4],x[ 5],x[ 6],x[ 7], x[12],x[13],x[14],x[15], 12);
	QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[12],x[13],x[14],x[15], x[ 4],x[ 5],x[ 6],x[ 7],  8);
	QROUND(x[ 8],x[ 9],x[10],x[11], x[ 4],x[ 5],x[ 6],x[ 7], x[12],x[13],x[14],x[15],  7);

	/* Operate on diagonals */
	QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[15],x[12],x[13],x[14], x[ 5],x[ 6],x[ 7],x[ 4], 16);
	QROUND(x[10],x[11],x[ 8],x[ 9], x[ 5],x[ 6],x[ 7],x[ 4], x[15],x[12],x[13],x[14], 12);
	QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[15],x[12],x[13],x[14], x[ 5],x[ 6],x[ 7],x[ 4],  8);
	QROUND(x[10],x[11],x[ 8],x[ 9], x[ 5],x[ 6],x[ 7],x[ 4], x[15],x[12],x[13],x[14],  7);

	/* Operate on columns. */
	QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[12],x[13],x[14],x[15], x[ 4],x[ 5],x[ 6],x[ 7], 16);
	QROUND(x[ 8],x[ 9],x[10],x[11], x[ 4],x[ 5],x[ 6],x[ 7], x[12],x[13],x[14],x[15], 12);
	QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[12],x[13],x[14],x[15], x[ 4],x[ 5],x[ 6],x[ 7],  8);
	QROUND(x[ 8],x[ 9],x[10],x[11], x[ 4],x[ 5],x[ 6],x[ 7], x[12],x[13],x[14],x[15],  7);

	/* Operate on diagonals */
	QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[15],x[12],x[13],x[14], x[ 5],x[ 6],x[ 7],x[ 4], 16);
	QROUND(x[10],x[11],x[ 8],x[ 9], x[ 5],x[ 6],x[ 7],x[ 4], x[15],x[12],x[13],x[14], 12);
	QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[15],x[12],x[13],x[14], x[ 5],x[ 6],x[ 7],x[ 4],  8);
	QROUND(x[10],x[11],x[ 8],x[ 9], x[ 5],x[ 6],x[ 7],x[ 4], x[15],x[12],x[13],x[14],  7);

	B[0].x += x[0]; B[0].y += x[1]; B[0].z += x[2];  B[0].w += x[3];  B[1].x += x[4];  B[1].y += x[5];  B[1].z += x[6];  B[1].w += x[7];
	B[2].x += x[8]; B[2].y += x[9]; B[2].z += x[10]; B[2].w += x[11]; B[3].x += x[12]; B[3].y += x[13]; B[3].z += x[14]; B[3].w += x[15];
}

#endif

#define ROTL7(a0,a1,a2,a3,a00,a10,a20,a30){\
a0^=(((a00)<<7) | ((a00)>>25) );\
a1^=(((a10)<<7) | ((a10)>>25) );\
a2^=(((a20)<<7) | ((a20)>>25) );\
a3^=(((a30)<<7) | ((a30)>>25) );\
};\

#define ROTL9(a0,a1,a2,a3,a00,a10,a20,a30){\
a0^=(((a00)<<9) | ((a00)>>23) );\
a1^=(((a10)<<9) | ((a10)>>23) );\
a2^=(((a20)<<9) | ((a20)>>23) );\
a3^=(((a30)<<9) | ((a30)>>23) );\
};\

#define ROTL13(a0,a1,a2,a3,a00,a10,a20,a30){\
a0^=(((a00)<<13) | ((a00)>>19) );\
a1^=(((a10)<<13) | ((a10)>>19) );\
a2^=(((a20)<<13) | ((a20)>>19) );\
a3^=(((a30)<<13) | ((a30)>>19) );\
};\

#define ROTL18(a0,a1,a2,a3,a00,a10,a20,a30){\
a0^=(((a00)<<18) | ((a00)>>14) );\
a1^=(((a10)<<18) | ((a10)>>14) );\
a2^=(((a20)<<18) | ((a20)>>14) );\
a3^=(((a30)<<18) | ((a30)>>14) );\
};\

static __device__ __forceinline__ uint4& operator^=(uint4& left, const uint4& right)
{
	left.x ^= right.x;
	left.y ^= right.y;
	left.z ^= right.z;
	left.w ^= right.w;
	return left;
}

////////////////////////////////////////////////////////////////////////////////
//! Scrypt core kernel for Fermi class devices.
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__
void fermi_scrypt_core_kernelA(uint32_t *g_idata, unsigned int N)
{
	extern __shared__ unsigned char x[];
	uint32_t ((*X)[WU_PER_WARP][16+4]) = (uint32_t (*)[WU_PER_WARP][16+4]) x;

	int warpIdx        = threadIdx.x / warpSize;
	int warpThread     = threadIdx.x % warpSize;
	const unsigned int LOOKUP_GAP = 1;

	// variables supporting the large memory transaction magic
	unsigned int Y = warpThread/4;
	unsigned int Z = 4*(warpThread%4);

	// add block specific offsets
	int WARPS_PER_BLOCK = blockDim.x / 32;
	int offset = blockIdx.x * WU_PER_BLOCK + warpIdx * WU_PER_WARP;
	g_idata += 32 * offset;
	uint32_t * V = c_V[offset / WU_PER_WARP]  + SCRATCH*Y + Z;

	// registers to store an entire work unit
	uint4 B[4], C[4];

	uint32_t ((*XB)[16+4]) = (uint32_t (*)[16+4])&X[warpIdx][Y][Z];
	uint32_t *XX = X[warpIdx][warpThread];

#pragma unroll 4
	for (int wu=0; wu < 32; wu+=8)
		*((ulonglong2*)(&V[SCRATCH*wu])) = *((ulonglong2*)XB[wu]) = *((ulonglong2*)(&g_idata[32*(wu+Y)+Z]));
#pragma unroll 4
	for (int idx=0; idx < 4; idx++) B[idx] = *((uint4*)&XX[4*idx]);

#pragma unroll 4
	for (int wu=0; wu < 32; wu+=8)
		*((ulonglong2*)(&V[SCRATCH*wu+16])) = *((ulonglong2*)XB[wu]) = *((ulonglong2*)(&g_idata[32*(wu+Y)+16+Z]));
#pragma unroll 4
	for (int idx=0; idx < 4; idx++) C[idx] = *((uint4*)&XX[4*idx]);

	for (int i = 1; i < N; i++) {

		xor_chacha8(B, C); xor_chacha8(C, B);

#pragma unroll 4
		for (int idx=0; idx < 4; idx++) *((uint4*)&XX[4*idx]) = B[idx];
#pragma unroll 4
		for (int wu=0; wu < 32; wu+=8)
			*((ulonglong2*)(&V[SCRATCH*wu + i*32])) = *((ulonglong2*)XB[wu]);

#pragma unroll 4
		for (int idx=0; idx < 4; idx++) *((uint4*)&XX[4*idx]) = C[idx];
#pragma unroll 4
		for (int wu=0; wu < 32; wu+=8)
			*((ulonglong2*)(&V[SCRATCH*wu + i*32 + 16])) = *((ulonglong2*)XB[wu]);
	}
}

__global__
void fermi_scrypt_core_kernelB(uint32_t *g_odata, unsigned int N)
{
	extern __shared__ unsigned char x[];
	uint32_t ((*X)[WU_PER_WARP][16+4]) = (uint32_t (*)[WU_PER_WARP][16+4]) x;

	int warpIdx        = threadIdx.x / warpSize;
	int warpThread     = threadIdx.x % warpSize;
	const unsigned int LOOKUP_GAP = 1;

	// variables supporting the large memory transaction magic
	unsigned int Y = warpThread/4;
	unsigned int Z = 4*(warpThread%4);

	// add block specific offsets
	int WARPS_PER_BLOCK = blockDim.x / 32;
	int offset = blockIdx.x * WU_PER_BLOCK + warpIdx * WU_PER_WARP;
	g_odata += 32 * offset;
	uint32_t * V = c_V[offset / WU_PER_WARP] + SCRATCH*Y + Z;

	// registers to store an entire work unit
	uint4 B[4], C[4];

	uint32_t ((*XB)[16+4]) = (uint32_t (*)[16+4])&X[warpIdx][Y][Z];
	uint32_t *XX = X[warpIdx][warpThread];

#pragma unroll 4
	for (int wu=0; wu < 32; wu+=8)
		*((ulonglong2*)XB[wu]) = *((ulonglong2*)(&V[SCRATCH*wu + (N-1)*32]));
#pragma unroll 4
	for (int idx=0; idx < 4; idx++) B[idx] = *((uint4*)&XX[4*idx]);

#pragma unroll 4
	for (int wu=0; wu < 32; wu+=8)
		*((ulonglong2*)XB[wu]) = *((ulonglong2*)(&V[SCRATCH*wu + (N-1)*32 + 16]));
#pragma unroll 4
	for (int idx=0; idx < 4; idx++) C[idx] = *((uint4*)&XX[4*idx]);

	xor_chacha8(B, C); xor_chacha8(C, B);

	for (int i = 0; i < N; i++) {

		XX[16] = 32 * (C[0].x & (N-1));

#pragma unroll 4
		for (int wu=0; wu < 32; wu+=8)
			*((ulonglong2*)XB[wu]) = *((ulonglong2*)(&V[SCRATCH*wu + XB[wu][16-Z]]));
#pragma unroll 4
		for (int idx=0; idx < 4; idx++) B[idx] ^= *((uint4*)&XX[4*idx]);

#pragma unroll 4
		for (int wu=0; wu < 32; wu+=8)
			*((ulonglong2*)XB[wu]) = *((ulonglong2*)(&V[SCRATCH*wu + XB[wu][16-Z] + 16]));
#pragma unroll 4
		for (int idx=0; idx < 4; idx++) C[idx] ^= *((uint4*)&XX[4*idx]);

		xor_chacha8(B, C); xor_chacha8(C, B);
	}

#pragma unroll 4
	for (int idx=0; idx < 4; idx++) *((uint4*)&XX[4*idx]) = B[idx];
#pragma unroll 4
	for (int wu=0; wu < 32; wu+=8)
		*((ulonglong2*)(&g_odata[32*(wu+Y)+Z])) = *((ulonglong2*)XB[wu]);

#pragma unroll 4
	for (int idx=0; idx < 4; idx++) *((uint4*)&XX[4*idx]) = C[idx];
#pragma unroll 4
	for (int wu=0; wu < 32; wu+=8)
		*((ulonglong2*)(&g_odata[32*(wu+Y)+16+Z])) = *((ulonglong2*)XB[wu]);

}

//
// Lookup-Gap variations of the above functions
//

__global__ void
fermi_scrypt_core_kernelA_LG(uint32_t *g_idata, unsigned int N, unsigned int LOOKUP_GAP)
{
	extern __shared__ unsigned char x[];
	uint32_t ((*X)[WU_PER_WARP][16+4]) = (uint32_t (*)[WU_PER_WARP][16+4]) x;

	int warpIdx        = threadIdx.x / warpSize;
	int warpThread     = threadIdx.x % warpSize;

	// variables supporting the large memory transaction magic
	unsigned int Y = warpThread/4;
	unsigned int Z = 4*(warpThread%4);

	// add block specific offsets
	int WARPS_PER_BLOCK = blockDim.x / 32;
	int offset = blockIdx.x * WU_PER_BLOCK + warpIdx * WU_PER_WARP;
	g_idata += 32 * offset;
	uint32_t * V = c_V[offset / WU_PER_WARP]  + SCRATCH*Y + Z;

	// registers to store an entire work unit
	uint4 B[4], C[4];

	uint32_t ((*XB)[16+4]) = (uint32_t (*)[16+4])&X[warpIdx][Y][Z];
	uint32_t *XX = X[warpIdx][warpThread];

#pragma unroll 4
	for (int wu=0; wu < 32; wu+=8)
		*((ulonglong2*)(&V[SCRATCH*wu])) = *((ulonglong2*)XB[wu]) = *((ulonglong2*)(&g_idata[32*(wu+Y)+Z]));
#pragma unroll 4
	for (int idx=0; idx < 4; idx++) B[idx] = *((uint4*)&XX[4*idx]);

#pragma unroll 4
	for (int wu=0; wu < 32; wu+=8)
		*((ulonglong2*)(&V[SCRATCH*wu+16])) = *((ulonglong2*)XB[wu]) = *((ulonglong2*)(&g_idata[32*(wu+Y)+16+Z]));
#pragma unroll 4
	for (int idx=0; idx < 4; idx++) C[idx] = *((uint4*)&XX[4*idx]);

	for (int i = 1; i < N; i++) {

		xor_chacha8(B, C); xor_chacha8(C, B);

		if (i % LOOKUP_GAP == 0) {
#pragma unroll 4
			for (int idx=0; idx < 4; idx++) *((uint4*)&XX[4*idx]) = B[idx];
#pragma unroll 4
			for (int wu=0; wu < 32; wu+=8)
				*((ulonglong2*)(&V[SCRATCH*wu + (i/LOOKUP_GAP)*32])) = *((ulonglong2*)XB[wu]);

#pragma unroll 4
			for (int idx=0; idx < 4; idx++) *((uint4*)&XX[4*idx]) = C[idx];
#pragma unroll 4
			for (int wu=0; wu < 32; wu+=8)
				*((ulonglong2*)(&V[SCRATCH*wu + (i/LOOKUP_GAP)*32 + 16])) = *((ulonglong2*)XB[wu]);
		}
	}
}

__global__ void
fermi_scrypt_core_kernelB_LG(uint32_t *g_odata, unsigned int N, unsigned int LOOKUP_GAP)
{
	extern __shared__ unsigned char x[];
	uint32_t ((*X)[WU_PER_WARP][16+4]) = (uint32_t (*)[WU_PER_WARP][16+4]) x;

	int warpIdx        = threadIdx.x / warpSize;
	int warpThread     = threadIdx.x % warpSize;

	// variables supporting the large memory transaction magic
	unsigned int Y = warpThread/4;
	unsigned int Z = 4*(warpThread%4);

	// add block specific offsets
	int WARPS_PER_BLOCK = blockDim.x / 32;
	int offset = blockIdx.x * WU_PER_BLOCK + warpIdx * WU_PER_WARP;
	g_odata += 32 * offset;
	uint32_t * V = c_V[offset / WU_PER_WARP] + SCRATCH*Y + Z;

	// registers to store an entire work unit
	uint4 B[4], C[4];

	uint32_t ((*XB)[16+4]) = (uint32_t (*)[16+4])&X[warpIdx][Y][Z];
	uint32_t *XX = X[warpIdx][warpThread];

	uint32_t pos = (N-1)/LOOKUP_GAP; uint32_t loop = 1 + (N-1)-pos*LOOKUP_GAP;
#pragma unroll 4
	for (int wu=0; wu < 32; wu+=8)
		*((ulonglong2*)XB[wu]) = *((ulonglong2*)(&V[SCRATCH*wu + pos*32]));
#pragma unroll 4
	for (int idx=0; idx < 4; idx++) B[idx] = *((uint4*)&XX[4*idx]);

#pragma unroll 4
	for (int wu=0; wu < 32; wu+=8)
		*((ulonglong2*)XB[wu]) = *((ulonglong2*)(&V[SCRATCH*wu + pos*32 + 16]));
#pragma unroll 4
	for (int idx=0; idx < 4; idx++) C[idx] = *((uint4*)&XX[4*idx]);

	while (loop--) {
		xor_chacha8(B, C); xor_chacha8(C, B);
	}

	for (int i = 0; i < N; i++) {

		uint32_t j = C[0].x & (N-1);
		uint32_t pos = j / LOOKUP_GAP; uint32_t loop = j - pos*LOOKUP_GAP;
		XX[16] = 32 * pos;

		uint4 b[4], c[4];
#pragma unroll 4
		for (int wu=0; wu < 32; wu+=8)
			*((ulonglong2*)XB[wu]) = *((ulonglong2*)(&V[SCRATCH*wu + XB[wu][16-Z]]));
#pragma unroll 4
		for (int idx=0; idx < 4; idx++) b[idx] = *((uint4*)&XX[4*idx]);

#pragma unroll 4
		for (int wu=0; wu < 32; wu+=8)
			*((ulonglong2*)XB[wu]) = *((ulonglong2*)(&V[SCRATCH*wu + XB[wu][16-Z] + 16]));
#pragma unroll 4
		for (int idx=0; idx < 4; idx++) c[idx] = *((uint4*)&XX[4*idx]);

		while (loop--) {
			xor_chacha8(b, c); xor_chacha8(c, b);
		}

#pragma unroll 4
		for (int idx=0; idx < 4; idx++) B[idx] ^= b[idx];
#pragma unroll 4
		for (int idx=0; idx < 4; idx++) C[idx] ^= c[idx];

		xor_chacha8(B, C); xor_chacha8(C, B);
	}

#pragma unroll 4
	for (int idx=0; idx < 4; idx++) *((uint4*)&XX[4*idx]) = B[idx];
#pragma unroll 4
	for (int wu=0; wu < 32; wu+=8)
		*((ulonglong2*)(&g_odata[32*(wu+Y)+Z])) = *((ulonglong2*)XB[wu]);

#pragma unroll 4
	for (int idx=0; idx < 4; idx++) *((uint4*)&XX[4*idx]) = C[idx];
#pragma unroll 4
	for (int wu=0; wu < 32; wu+=8)
		*((ulonglong2*)(&g_odata[32*(wu+Y)+16+Z])) = *((ulonglong2*)XB[wu]);

}

