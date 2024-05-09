#ifndef DEBUG_CUH
#define DEBUG_CUH

#include <cuda_runtime.h>
#define PRINT_GPU(...) fastlanes::gpu::debug::print_gpu(__VA_ARGS__)

namespace fastlanes::gpu::debug {
template <typename T>
__device__ void print_gpu(T* arr, const char* str) {
	__syncthreads();
	if (threadIdx.x == 0) {
		printf("\n ==================   %s   ================= \n ", str);

		for (int ITEM = 0; ITEM < 1024; ++ITEM) {
			if (ITEM % 128 == 0) { printf("\n"); }
			printf(" %d | ", arr[ITEM]);
		}

		printf("\n");
	}
	__syncthreads();
}

} // namespace fastlanes::gpu::debug

#endif // DEBUG_CUH
