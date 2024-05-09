#pragma once

#include "common.cuh"
#include "error.cuh"
#include <cub/util_allocator.cuh>

#define SETUP_TIMING()                                                                                                 \
	cudaEvent_t start, stop;                                                                                           \
	cudaEventCreate(&start);                                                                                           \
	cudaEventCreate(&stop);

#define TIME_FUNC(f, t)                                                                                                \
	{                                                                                                                  \
		cudaEventRecord(start, 0);                                                                                     \
		f;                                                                                                             \
		cudaEventRecord(stop, 0);                                                                                      \
		cudaEventSynchronize(stop);                                                                                    \
		cudaEventElapsedTime(&t, start, stop);                                                                         \
	}

#define CLEANUP(vec)                                                                                                   \
	if (vec) CubDebugExit(fastlanes::gpu::g_allocator.DeviceFree(vec))

#define ALLOCATE(vec, size) CubDebugExit(g_allocator.DeviceAllocate((void**)&vec, size))

template <typename T>
T* loadToGPU(const T* src, int numEntries, cub::CachingDeviceAllocator& g_allocator) {
	T* dest;
	CHECK_CUDA_ERROR(g_allocator.DeviceAllocate((void**)&dest, sizeof(T) * numEntries));
	CHECK_CUDA_ERROR(cudaMemcpy(dest, src, sizeof(T) * numEntries, cudaMemcpyHostToDevice));
	return dest;
}

inline void* gpu_load(const void* src, bsz_t size, cub::CachingDeviceAllocator& g_allocator) {
	void* dest;
	CHECK_CUDA_ERROR(g_allocator.DeviceAllocate((void**)&dest, size));
	CHECK_CUDA_ERROR(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice));
	return dest;
}

#define CHECK_ERROR()                                                                                                  \
	{                                                                                                                  \
		cudaDeviceSynchronize();                                                                                       \
		cudaError_t error = cudaGetLastError();                                                                        \
		if (error != cudaSuccess) {                                                                                    \
			printf("CUDA error: %s\n", cudaGetErrorString(error));                                                     \
			exit(-1);                                                                                                  \
		}                                                                                                              \
	}

#define CUDA_SAFE_CALL(call)                                                                                           \
	do {                                                                                                               \
		cudaError_t err = call;                                                                                        \
		if (cudaSuccess != err) {                                                                                      \
			fprintf(stderr, "Cuda error in file '%s' in line %i : %s.", __FILE__, __LINE__, cudaGetErrorString(err));  \
			exit(EXIT_FAILURE);                                                                                        \
		}                                                                                                              \
	} while (0)

namespace fastlanes::gpu {
inline void* load_arr(void* src, uint64_t bsz) {
	void* dest = nullptr;
	cudaMalloc((void**)&dest, bsz);
	cudaMemcpy(dest, src, bsz, cudaMemcpyHostToDevice);
	return dest;
}

template <typename T>
inline T* load_arr(T* src, uint64_t bsz) {
	T* dest = nullptr;
	cudaMalloc((void**)&dest, bsz);
	cudaMemcpy(dest, src, bsz, cudaMemcpyHostToDevice);
	return dest;
}

template <typename T>
inline T* load_to_gpu(const T* src, bsz_t size, cub::CachingDeviceAllocator& g_allocator) {
	T* dest;
	CHECK_CUDA_ERROR(g_allocator.DeviceAllocate((void**)&dest, size));
	CHECK_CUDA_ERROR(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice));
	return dest;
}

inline void* load_to_gpu(const void* src, bsz_t size, cub::CachingDeviceAllocator& g_allocator) {
	void* dest;
	CHECK_CUDA_ERROR(g_allocator.DeviceAllocate((void**)&dest, size));
	CHECK_CUDA_ERROR(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice));
	return dest;
}

inline cub::CachingDeviceAllocator g_allocator(true); // Caching allocator for device memory

} // namespace fastlanes::gpu