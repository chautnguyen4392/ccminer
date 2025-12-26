#ifndef VOLTA_KERNEL_H
#define VOLTA_KERNEL_H

#include "miner.h"
#include <cuda_runtime.h>

#include "salsa_kernel.h"

class VoltaKernel : public KernelInterface
{
public:
	VoltaKernel();

	virtual void set_scratchbuf_constants(int MAXWARPS, uint32_t** h_V);
	virtual bool run_kernel(dim3 grid, dim3 threads, int WARPS_PER_BLOCK, int thr_id, cudaStream_t stream, uint32_t* d_idata, uint32_t* d_odata, unsigned int N, unsigned int LOOKUP_GAP, bool interactive, bool benchmark);

	virtual char get_identifier() { return 'V'; };
	virtual int get_major_version() { return 7; };
	virtual int get_minor_version() { return 0; };

	virtual int max_warps_per_block() { return 64; };
	virtual int get_texel_width() { return 4; };
	virtual int threads_per_wu() { return 1; }
	virtual bool support_lookup_gap() { return true; }

	virtual cudaSharedMemConfig shared_mem_config() { return cudaSharedMemBankSizeFourByte; }
	virtual cudaFuncCache cache_config() { return cudaFuncCachePreferL1; }

};

#endif // #ifndef VOLTA_KERNEL_H
