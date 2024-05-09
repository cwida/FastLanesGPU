#ifndef FLS_GPU_UTIL_CUH
#define FLS_GPU_UTIL_CUH

namespace fasltanes::gpu{

/**/
template <typename T>
T* loadColumnToGPU(T* src, int len) {
	T* dest = nullptr;
	cudaMalloc((void**)&dest, sizeof(T) * len);
	cudaMemcpy(dest, src, sizeof(T) * len, cudaMemcpyHostToDevice);
	return dest;
}
/**/
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

#define CUDA_SAFE_CALL(call)                                                                                           \
	do {                                                                                                               \
		cudaError_t err = call;                                                                                        \
		if (cudaSuccess != err) {                                                                                      \
			fprintf(stderr, "Cuda error in file '%s' in line %i : %s.", __FILE__, __LINE__, cudaGetErrorString(err));  \
			exit(EXIT_FAILURE);                                                                                        \
		}                                                                                                              \
	} while (0)

#define SETUP_TIMING()                                                                                                 \
	cudaEvent_t start, stop;                                                                                           \
	cudaEventCreate(&start);                                                                                           \
	cudaEventCreate(&stop);

#define PERF(f, t)                                                                                                     \
	{                                                                                                                  \
		cudaEventRecord(start, 0);                                                                                     \
		f;                                                                                                             \
		cudaEventRecord(stop, 0);                                                                                      \
		cudaEventSynchronize(stop);                                                                                    \
		cudaEventElapsedTime(&t, start, stop);                                                                         \
	}

/*
  RAM with a memory clock rate of 1,546 MHz and a 384-bit wide memory interface. Using these data items,
  the peak theoretical memory bandwidth of the NVIDIA Tesla M2050 is 148 GB/sec, as computed in the following.

    BWTheoretical = 1546 * 10^6 * (384/8) * 2 / 10^9 = 148 GB/s
 */
inline double BWTheoretical(int memoryClockRate, int memoryBusWidth) {
	return ((memoryClockRate * (1e3)) * (static_cast<double>(memoryBusWidth) / 8) * 2) / static_cast<double>((1e9));
}

/*

 BWEffective = (RB + WB) / (t * 109)

 Here, BWEffective is the effective bandwidth in units of GB/s, RB is the number of bytes read per kernel,
 WB is the number of bytes written per kernel, and t is the elapsed time given in seconds.

 */
inline double BWEffective(uint64_t read_bsz, uint64_t write_bsz, uint64_t milliseconds) {
	return static_cast<double>(read_bsz + write_bsz) / milliseconds / 1e6;
}

template <typename T>
class DevicePtr {
public:
	__device__ __host__ __inline__ explicit DevicePtr(T* a_p)
	    : p(a_p) {}

public:
	T* p;
};

template <typename T>
__host__ inline auto make_device_ptr(T* p) {
	return DevicePtr<T>(p);
}

}

#endif //FLS_GPU_UTIL_CUH
