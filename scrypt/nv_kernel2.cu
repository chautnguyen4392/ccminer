//
// Experimental Kernel for Kepler (Compute 3.5) devices
// code submitted by nVidia performance engineer Alexey Panteleev
// with modifications by Christian Buchner
//
// for Compute 3.5
// NOTE: compile this .cu module for compute_35,sm_35 with --maxrregcount=80
// for Compute 3.0
// NOTE: compile this .cu module for compute_30,sm_30 with --maxrregcount=63
//

#include <map>

#include <cuda_runtime.h>
#include <cuda_helper.h>
#include "miner.h"

#include "salsa_kernel.h"
#include "nv_kernel2.h"

#define THREADS_PER_WU 1  // single thread per hash

// THREADS_PER_WARP is defined in salsa_kernel.h (can be 8, 16, 24, or 32)
// This allows tuning the granularity of work units
// Set via -DTHREADS_PER_WARP=N at compile time

// Validate threads per warp at compile time
#if (THREADS_PER_WARP != 8) && (THREADS_PER_WARP != 16) && (THREADS_PER_WARP != 24) && (THREADS_PER_WARP != 32)
#error "THREADS_PER_WARP must be 8, 16, 24, or 32"
#endif

// Number of 8-thread tiles per virtual warp
#define TILES_PER_VWARP (THREADS_PER_WARP / 8)

// Fixed tile stride for memory layout - must be 32 to avoid overlap between tiles
// Each tile needs 8 positions (lanes) * 4 rows = 32 positions in the strided layout
// Using THREADS_PER_WARP here would cause overlap when THREADS_PER_WARP < 32
#define TILE_STRIDE 32

// Use synchronized shuffle for CUDA 9.0+ (required for Volta/Ampere and later)
#if CUDA_VERSION >= 9000 && __CUDA_ARCH__ >= 300
#define SHFL(var, srcLane, width) __shfl_sync(0xFFFFFFFFu, var, srcLane, width)
#else
#define SHFL(var, srcLane, width) __shfl(var, srcLane, width)
#endif

#if __CUDA_ARCH__ < 350
	// Kepler (Compute 3.0)
	#define __ldg(x) (*(x))
#endif

#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ >= 300

// grab hardware lane ID (0-31 within actual warp)
static __device__ __inline__ unsigned int __laneId() { unsigned int laneId; asm( "mov.u32 %0, %%laneid;" : "=r"( laneId ) ); return laneId; }

// virtual lane ID within virtual warp (0 to THREADS_PER_WARP-1)
static __device__ __inline__ unsigned int __vLaneId() { return threadIdx.x % THREADS_PER_WARP; }

// forward references - N_1 and spacing passed as register parameters for better performance
__global__ void nv2_scrypt_core_kernelA_LG(uint32_t *g_idata, int iterations, unsigned int LOOKUP_GAP, uint32_t N_1, uint32_t spacing);
__global__ void nv2_scrypt_core_kernelB_LG(uint32_t *g_odata, int iterations, unsigned int LOOKUP_GAP, uint32_t N_1, uint32_t spacing);

// scratchbuf constants (pointers to scratch buffer for each work unit)
// Note: c_V stays in constant memory because it's an array of pointers that varies per virtual warp
__constant__ uint32_t* c_V[TOTAL_WARP_LIMIT];


NV2Kernel::NV2Kernel() : KernelInterface()
{
}

void NV2Kernel::set_scratchbuf_constants(int MAXWARPS, uint32_t** h_V)
{
	checkCudaErrors(cudaMemcpyToSymbol(c_V, h_V, MAXWARPS*sizeof(uint32_t*), 0, cudaMemcpyHostToDevice));
}

bool NV2Kernel::run_kernel(dim3 grid, dim3 threads, int WARPS_PER_BLOCK, int thr_id, cudaStream_t stream, uint32_t* d_idata, uint32_t* d_odata, unsigned int N, unsigned int LOOKUP_GAP, bool interactive, bool benchmark)
{
	// Compute N_1 and spacing as register parameters (faster than constant memory)
	uint32_t N_1 = N - 1;
	uint32_t spacing = (N + LOOKUP_GAP - 1) / LOOKUP_GAP;

	// First phase: Sequential writes to scratchpad.
	nv2_scrypt_core_kernelA_LG<<< grid, threads, 0, stream >>>(d_idata, N, LOOKUP_GAP, N_1, spacing);

	// Second phase: Random read access from scratchpad.
	nv2_scrypt_core_kernelB_LG<<< grid, threads, 0, stream >>>(d_odata, N, LOOKUP_GAP, N_1, spacing);

	return true;
}

static __device__ uint4& operator^=(uint4& left, const uint4& right)
{
	left.x ^= right.x;
	left.y ^= right.y;
	left.z ^= right.z;
	left.w ^= right.w;
	return left;
}

__device__ __forceinline__ uint4 shfl4(const uint4 val, unsigned int lane, unsigned int width)
{
	return make_uint4(
		(unsigned int)SHFL((int)val.x, lane, width),
		(unsigned int)SHFL((int)val.y, lane, width),
		(unsigned int)SHFL((int)val.z, lane, width),
		(unsigned int)SHFL((int)val.w, lane, width)
	);
}

__device__ __forceinline__ void __transposed_write_BC(uint4 (&B)[4], uint4 (&C)[4], uint4 *D, int spacing)
{
	unsigned int vLaneId = __vLaneId();

	unsigned int lane8 = vLaneId % 8;
	unsigned int tile  = vLaneId / 8;  // 0 to (TILES_PER_VWARP-1)

	uint4 T1[8], T2[8];

	/* Source matrix, A-H are threads, 0-7 are data items, thread A is marked with `*`:

	   *A0  B0  C0  D0  E0  F0  G0  H0
	   *A1  B1  C1  D1  E1  F1  G1  H1
	   *A2  B2  C2  D2  E2  F2  G2  H2
	   *A3  B3  C3  D3  E3  F3  G3  H3
	   *A4  B4  C4  D4  E4  F4  G4  H4
	   *A5  B5  C5  D5  E5  F5  G5  H5
	   *A6  B6  C6  D6  E6  F6  G6  H6
	   *A7  B7  C7  D7  E7  F7  G7  H7
	*/

	// rotate rows
	T1[0] = B[0];
	T1[1] = shfl4(B[1], lane8 + 7, 8);
	T1[2] = shfl4(B[2], lane8 + 6, 8);
	T1[3] = shfl4(B[3], lane8 + 5, 8);
	T1[4] = shfl4(C[0], lane8 + 4, 8);
	T1[5] = shfl4(C[1], lane8 + 3, 8);
	T1[6] = shfl4(C[2], lane8 + 2, 8);
	T1[7] = shfl4(C[3], lane8 + 1, 8);

	/* Matrix after row rotates:

	   *A0  B0  C0  D0  E0  F0  G0  H0
		H1 *A1  B1  C1  D1  E1  F1  G1
		G2  H2 *A2  B2  C2  D2  E2  F2
		F3  G3  H3 *A3  B3  C3  D3  E3
		E4  F4  G4  H4 *A4  B4  C4  D4
		D5  E5  F5  G5  H5 *A5  B5  C5
		C6  D6  E6  F6  G6  H6 *A6  B6
		B7  C7  D7  E7  F7  G7  H7 *A7
	*/

	// rotate columns up using a barrel shifter simulation
	// column X is rotated up by (X+1) items
#pragma unroll 8
	for(int n = 0; n < 8; n++) T2[n] = ((lane8+1) & 1) ? T1[(n+1) % 8] : T1[n];
#pragma unroll 8
	for(int n = 0; n < 8; n++) T1[n] = ((lane8+1) & 2) ? T2[(n+2) % 8] : T2[n];
#pragma unroll 8
	for(int n = 0; n < 8; n++) T2[n] = ((lane8+1) & 4) ? T1[(n+4) % 8] : T1[n];

	/* Matrix after column rotates:

		H1  H2  H3  H4  H5  H6  H7  H0
		G2  G3  G4  G5  G6  G7  G0  G1
		F3  F4  F5  F6  F7  F0  F1  F2
		E4  E5  E6  E7  E0  E1  E2  E3
		D5  D6  D7  D0  D1  D2  D3  D4
		C6  C7  C0  C1  C2  C3  C4  C5
		B7  B0  B1  B2  B3  B4  B5  B6
	   *A0 *A1 *A2 *A3 *A4 *A5 *A6 *A7
	*/

	// rotate rows again using address math and write to D, in reverse row order
	// Use TILE_STRIDE (fixed 32) to avoid overlap between tiles when THREADS_PER_WARP < 32
	// Each tile needs 32 positions: 8 lanes * 4 strided rows (offsets 0,4,8,12,16,20,24,28)
	D[spacing*2*(TILE_STRIDE*tile   )+ lane8     ] = T2[7];
	D[spacing*2*(TILE_STRIDE*tile+4 )+(lane8+7)%8] = T2[6];
	D[spacing*2*(TILE_STRIDE*tile+8 )+(lane8+6)%8] = T2[5];
	D[spacing*2*(TILE_STRIDE*tile+12)+(lane8+5)%8] = T2[4];
	D[spacing*2*(TILE_STRIDE*tile+16)+(lane8+4)%8] = T2[3];
	D[spacing*2*(TILE_STRIDE*tile+20)+(lane8+3)%8] = T2[2];
	D[spacing*2*(TILE_STRIDE*tile+24)+(lane8+2)%8] = T2[1];
	D[spacing*2*(TILE_STRIDE*tile+28)+(lane8+1)%8] = T2[0];
}

__device__ __forceinline__ void __transposed_read_BC(const uint4 *S, uint4 (&B)[4], uint4 (&C)[4], int spacing, int row)
{
	unsigned int vLaneId = __vLaneId();

	unsigned int lane8 = vLaneId % 8;
	unsigned int tile  = vLaneId / 8;  // 0 to (TILES_PER_VWARP-1)

	// Perform the same transposition as in __transposed_write_BC, but in reverse order.
	// See the illustrations in comments for __transposed_write_BC.

	// read and rotate rows, in reverse row order
	// Use TILE_STRIDE (fixed 32) to match the write pattern and avoid overlap
	uint4 T1[8], T2[8];
	const uint4 *loc;
	loc = &S[(spacing*2*(TILE_STRIDE*tile   ) +  lane8      + 8*SHFL(row, 0, 8))];
	T1[7] = __ldg(loc);
	loc = &S[(spacing*2*(TILE_STRIDE*tile+4 ) + (lane8+7)%8 + 8*SHFL(row, 1, 8))];
	T1[6] = __ldg(loc);
	loc = &S[(spacing*2*(TILE_STRIDE*tile+8 ) + (lane8+6)%8 + 8*SHFL(row, 2, 8))];
	T1[5] = __ldg(loc);
	loc = &S[(spacing*2*(TILE_STRIDE*tile+12) + (lane8+5)%8 + 8*SHFL(row, 3, 8))];
	T1[4] = __ldg(loc);
	loc = &S[(spacing*2*(TILE_STRIDE*tile+16) + (lane8+4)%8 + 8*SHFL(row, 4, 8))];
	T1[3] = __ldg(loc);
	loc = &S[(spacing*2*(TILE_STRIDE*tile+20) + (lane8+3)%8 + 8*SHFL(row, 5, 8))];
	T1[2] = __ldg(loc);
	loc = &S[(spacing*2*(TILE_STRIDE*tile+24) + (lane8+2)%8 + 8*SHFL(row, 6, 8))];
	T1[1] = __ldg(loc);
	loc = &S[(spacing*2*(TILE_STRIDE*tile+28) + (lane8+1)%8 + 8*SHFL(row, 7, 8))];
	T1[0] = __ldg(loc);

	// rotate columns down using a barrel shifter simulation
	// column X is rotated down by (X+1) items, or up by (8-(X+1)) = (7-X) items
#pragma unroll 8
	for(int n = 0; n < 8; n++) T2[n] = ((7-lane8) & 1) ? T1[(n+1) % 8] : T1[n];
#pragma unroll 8
	for(int n = 0; n < 8; n++) T1[n] = ((7-lane8) & 2) ? T2[(n+2) % 8] : T2[n];
#pragma unroll 8
	for(int n = 0; n < 8; n++) T2[n] = ((7-lane8) & 4) ? T1[(n+4) % 8] : T1[n];

	// rotate rows
	B[0] = T2[0];
	B[1] = shfl4(T2[1], lane8 + 1, 8);
	B[2] = shfl4(T2[2], lane8 + 2, 8);
	B[3] = shfl4(T2[3], lane8 + 3, 8);
	C[0] = shfl4(T2[4], lane8 + 4, 8);
	C[1] = shfl4(T2[5], lane8 + 5, 8);
	C[2] = shfl4(T2[6], lane8 + 6, 8);
	C[3] = shfl4(T2[7], lane8 + 7, 8);

}

__device__ __forceinline__ void __transposed_xor_BC(const uint4 *S, uint4 (&B)[4], uint4 (&C)[4], int spacing, int row)
{
	uint4 BT[4], CT[4];
	__transposed_read_BC(S, BT, CT, spacing, row);

#pragma unroll 4
	for(int n = 0; n < 4; n++)
	{
		B[n] ^= BT[n];
		C[n] ^= CT[n];
	}
}

// Optimized rotate using funnel shift on SM 3.5+
#if __CUDA_ARCH__ >= 350
	#define ROTL(a, b) __funnelshift_l(a, a, b)
#else
	#define ROTL(a, b) (((a) << (b)) | ((a) >> (32 - (b))))
#endif

// ChaCha quarter round: operates on 4 words (a, b, c, d)
#define QUARTERROUND(a, b, c, d) { \
	a += b; d ^= a; d = ROTL(d, 16); \
	c += d; b ^= c; b = ROTL(b, 12); \
	a += b; d ^= a; d = ROTL(d, 8); \
	c += d; b ^= c; b = ROTL(b, 7); \
}

static __device__ __forceinline__ void xor_chacha8(uint4 *B, uint4 *C)
{
	// Use named registers instead of array for better register allocation
	register uint32_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;
	
	// XOR B with C and load into registers
	x0  = (B[0].x ^= C[0].x);
	x1  = (B[0].y ^= C[0].y);
	x2  = (B[0].z ^= C[0].z);
	x3  = (B[0].w ^= C[0].w);
	x4  = (B[1].x ^= C[1].x);
	x5  = (B[1].y ^= C[1].y);
	x6  = (B[1].z ^= C[1].z);
	x7  = (B[1].w ^= C[1].w);
	x8  = (B[2].x ^= C[2].x);
	x9  = (B[2].y ^= C[2].y);
	x10 = (B[2].z ^= C[2].z);
	x11 = (B[2].w ^= C[2].w);
	x12 = (B[3].x ^= C[3].x);
	x13 = (B[3].y ^= C[3].y);
	x14 = (B[3].z ^= C[3].z);
	x15 = (B[3].w ^= C[3].w);

	// 8 rounds (4 double-rounds)
	#pragma unroll 4
	for (int i = 0; i < 4; i++) {
		// Column round
		QUARTERROUND(x0, x4,  x8, x12);
		QUARTERROUND(x1, x5,  x9, x13);
		QUARTERROUND(x2, x6, x10, x14);
		QUARTERROUND(x3, x7, x11, x15);
		// Diagonal round
		QUARTERROUND(x0, x5, x10, x15);
		QUARTERROUND(x1, x6, x11, x12);
		QUARTERROUND(x2, x7,  x8, x13);
		QUARTERROUND(x3, x4,  x9, x14);
	}

	// Add back to B
	B[0].x += x0;  B[0].y += x1;  B[0].z += x2;  B[0].w += x3;
	B[1].x += x4;  B[1].y += x5;  B[1].z += x6;  B[1].w += x7;
	B[2].x += x8;  B[2].y += x9;  B[2].z += x10; B[2].w += x11;
	B[3].x += x12; B[3].y += x13; B[3].z += x14; B[3].w += x15;
}


////////////////////////////////////////////////////////////////////////////////
//! Experimental Scrypt-Jane core kernel for Titan devices.
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
//! 
//! Modified to support configurable threads per warp (8, 16, 24, 32)
//! Use THREADS_PER_WARP to tune the granularity of work units.
////////////////////////////////////////////////////////////////////////////////
__global__ void nv2_scrypt_core_kernelA_LG(uint32_t *g_idata, int iterations, unsigned int LOOKUP_GAP, uint32_t N_1, uint32_t spacing)
{
	// Calculate warp ID (global index of this warp)
	int vwarp_id = (blockIdx.x * blockDim.x + threadIdx.x) / THREADS_PER_WARP;
	
	// Host allocates THREADS_PER_WARP work units per warp (with THREADS_PER_WU=1)
	// Each "work unit" has 32 uint32_t, so total input per warp = 32 * THREADS_PER_WARP
	// The transposed read accesses 8 * THREADS_PER_WARP uint4 = 32 * THREADS_PER_WARP uint32_t
	g_idata += 32 * THREADS_PER_WARP * vwarp_id;
	uint32_t * V = c_V[vwarp_id];
	uint4 B[4], C[4];

	__transposed_read_BC((uint4*)g_idata, B, C, 1, 0);
	__transposed_write_BC(B, C, (uint4*)V, spacing);

	for (int i = 1; i < iterations; i++) {
		xor_chacha8(B, C); xor_chacha8(C, B);
		if (i % LOOKUP_GAP == 0)
		  // Stride between scratchpad rows: 8 uint4 = 32 uint32_t per row
		  __transposed_write_BC(B, C, (uint4*)(V + (i/LOOKUP_GAP)*32), spacing);
	}
}

__global__ void nv2_scrypt_core_kernelB_LG(uint32_t *g_odata, int iterations, unsigned int LOOKUP_GAP, uint32_t N_1, uint32_t spacing)
{
	// Calculate warp ID (global index of this warp)
	int vwarp_id = (blockIdx.x * blockDim.x + threadIdx.x) / THREADS_PER_WARP;
	
	// Host expects THREADS_PER_WARP work units per warp (with THREADS_PER_WU=1)
	// Each "work unit" has 32 uint32_t, so total output per warp = 32 * THREADS_PER_WARP
	// The transposed write produces 8 * THREADS_PER_WARP uint4 = 32 * THREADS_PER_WARP uint32_t
	g_odata += 32 * THREADS_PER_WARP * vwarp_id;
	uint32_t * V = c_V[vwarp_id];
	uint4 B[4], C[4];

	int pos = N_1/LOOKUP_GAP, loop = 1 + (N_1-pos*LOOKUP_GAP);
	__transposed_read_BC((uint4*)V, B, C, spacing, pos);
	while(loop--) { xor_chacha8(B, C); xor_chacha8(C, B); }

	for (int i = 0; i < iterations; i++)  {
		// Each thread calculates its own slot from its own C[0].x
		// The transposed_read_BC uses SHFL(row, k, 8) to gather from different rows
		// based on each thread's slot value - this is a vectorized gather pattern
		int slot = C[0].x & N_1;
		int pos = slot/LOOKUP_GAP, loop = slot-pos*LOOKUP_GAP;
		uint4 b[4], c[4]; __transposed_read_BC((uint4*)(V), b, c, spacing, pos);
		while(loop--) { xor_chacha8(b, c); xor_chacha8(c, b); }
#pragma unroll 4
		for(int n = 0; n < 4; n++) { B[n] ^= b[n]; C[n] ^= c[n]; }
		xor_chacha8(B, C); xor_chacha8(C, B);
	}

	__transposed_write_BC(B, C, (uint4*)(g_odata), 1);
}

#endif /* prevent SM 2 */

