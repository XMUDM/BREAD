#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

const int TOTAL_THREADS_GPU = 2048 * 13;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

static int calc_num_blocks(int total_items, int items_per_block) {
	int num_blocks = (total_items + items_per_block - 1) / items_per_block;
	int max_num_blocks = TOTAL_THREADS_GPU / items_per_block;
	if (max_num_blocks < num_blocks) num_blocks = max_num_blocks;
	return num_blocks;
}

