#include "crystal/crystal.cuh"
#include "cub/test/test_util.h"
#include "fls_gen/unpack/unpack_fused.cuh"
#include "gpu_utils.h"
#include "ssb_utils.h"
#include <fls_gen/pack/pack.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace fastlanes::gpu;
using namespace fastlanes;

struct QueryMtd {
	n_t      n_vec;
	uint8_t  bw;
	n_t      n_tup;
	uint64_t result;
};

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void QueryKernel(const uint32_t* encoded_col, QueryMtd query_mtd, unsigned long long* revenue) {
	// int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
	// Load a segment of consecutive items that are blocked across threads
	uint32_t items[ITEMS_PER_THREAD];

	long long sum = 0;

	// int tile_offset    = blockIdx.x * TILE_SIZE;
	// int num_tiles      = (query_mtd.n_tup + TILE_SIZE - 1) / TILE_SIZE;


	int extendedprice_tile_offset = blockIdx.x * query_mtd.bw * 32;
	unpack_device(encoded_col + extendedprice_tile_offset, items, query_mtd.bw);

#pragma unroll
	for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
		sum += items[ITEM];
	}

	__syncthreads();

	static __shared__ long long buffer[32];
	unsigned long long aggregate = BlockSum<long long, BLOCK_THREADS, ITEMS_PER_THREAD>(sum, (long long*)buffer);
	__syncthreads();

	if (threadIdx.x == 0) { atomicAdd(revenue, aggregate); }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
float query_aggregate(const uint32_t* enc_arr, QueryMtd hardcoded, cub::CachingDeviceAllocator& g_allocator) {
	// int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

	SETUP_TIMING();
	float                                     time_query;
	chrono::high_resolution_clock::time_point st, finish;
	st = chrono::high_resolution_clock::now();
	cudaEventRecord(start, 0);
	unsigned long long* d_sum = NULL;
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_sum, sizeof(long long)));

	cudaMemset(d_sum, 0, sizeof(long long));

	// Run
	QueryKernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<hardcoded.n_vec, BLOCK_THREADS>>>(enc_arr, hardcoded, d_sum);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_query, start, stop);

	unsigned long long revenue;
	CubDebugExit(cudaMemcpy(&revenue, d_sum, sizeof(uint64_t), cudaMemcpyDeviceToHost));

	finish                             = chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = finish - st;

	double total_time_taken {diff.count() * 1000};
	FLS_SHOW(total_time_taken)

	/*Check the result*/
	FLS_SHOW(revenue)
	if (revenue != hardcoded.result) { throw std::runtime_error("RESULT INCOREECT!"); }

	CLEANUP(d_sum);

	return time_query;
}

n_t bitpacked_vec_n_tup(uint8_t bitdwith) {
	/**/
	return bitdwith * 32;
}

void shared_memory_bitpacking_with_aggregation() {

	constexpr uint64_t n_vec          = 256 * 1024;
	constexpr uint64_t vec_sz         = 1024;
	constexpr uint64_t n_tup          = vec_sz * n_vec;
	auto*              h_org_arr      = new uint32_t[n_tup];
	auto*              h_encoded_data = new uint32_t[n_tup];
	size_t             repeat         = 3;

	for (uint8_t bitwidth {0}; bitwidth < 33; bitwidth++) {

		uint32_t bw = bitwidth;

		uint32_t mask            = (1 << bitwidth) - 1;
		uint64_t encoded_arr_bsz = n_tup * sizeof(int);

		FLS_SHOW(bw)
		uint64_t sum {0};
		/* generate random numbers. */
		for (int i = 0; i < n_tup; i++) {
			h_org_arr[i] = 5 & mask;
			sum += h_org_arr[i];
		}
		FLS_SHOW(sum)

		auto in  = h_org_arr;
		auto out = h_encoded_data;
		for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
			generated::pack::fallback::scalar::pack(in, out, bitwidth);
			in  = in + vec_sz;
			out = out + (bitwidth * 32);
		}

		auto* d_encoded_arr = gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, g_allocator);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		QueryMtd query_mtd {n_vec, bitwidth, n_tup, sum};
		for (int i {0}; i < repeat; ++i) {
			auto time = query_aggregate<32, 32>(d_encoded_arr, query_mtd, g_allocator);
			FLS_SHOW(time)
		}


		CLEANUP(d_encoded_arr)
	}
}

int main() {
	/**/
	shared_memory_bitpacking_with_aggregation();
}