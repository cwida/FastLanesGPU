#include "crystal/crystal.cuh"
#include "cub/test/test_util.h"
#include "fls_gen/unpack/unpack.cuh"
#include "gpu_utils.h"
#include "ssb_utils.h"
#include <fls_gen/pack/pack.hpp>
#include <fls_gen/rsum/rsum.cuh>
#include <fls_gen/transpose/transpose.hpp>
#include <fls_gen/unrsum/unrsum.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace fastlanes;
using namespace fastlanes::gpu;

struct QueryMtd {
	n_t      n_vec;
	uint8_t  bw;
	n_t      n_tup;
	uint64_t result;
};

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void
QueryKernel(const uint32_t* base_col, const uint32_t* bitpacked_col, QueryMtd query_mtd, unsigned long long* revenue) {
	int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
	// Load a segment of consecutive items that are blocked across threads
	uint32_t items[ITEMS_PER_THREAD];

	static __shared__ uint32_t unpacked[1024];
	static __shared__ uint32_t rsumed[1024];

	long long sum = 0;

	int tile_offset    = blockIdx.x * TILE_SIZE;
	int num_tiles      = (query_mtd.n_tup + TILE_SIZE - 1) / TILE_SIZE;
	int num_tile_items = TILE_SIZE;
	if (blockIdx.x == num_tiles - 1) { num_tile_items = query_mtd.n_tup - tile_offset; }

	int bitpacked_col_tile_offset = blockIdx.x * query_mtd.bw * 32;
	unpack_device(bitpacked_col + bitpacked_col_tile_offset, unpacked, query_mtd.bw);

	int based_col_tile_offset = blockIdx.x * 32;
	d_rsum_32(unpacked, rsumed, base_col + based_col_tile_offset);

	BlockLoad<uint32_t, BLOCK_THREADS, ITEMS_PER_THREAD>(rsumed, items, num_tile_items);

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
float query_aggregate(const uint32_t*              base_col,
                      const uint32_t*              bitpacked_col,
                      QueryMtd                     hardcoded,
                      cub::CachingDeviceAllocator& g_allocator) {
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
	QueryKernel<BLOCK_THREADS, ITEMS_PER_THREAD>
	    <<<hardcoded.n_vec, BLOCK_THREADS>>>(base_col, bitpacked_col, hardcoded, d_sum);

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

void shared_memory_delta_with_aggregation() {
	size_t         repeat           = 1;
	const uint64_t n_vec            = 256 * 1024;
	const uint64_t vec_sz           = 1024;
	const uint64_t n_tup            = vec_sz * n_vec;
	const uint64_t n_base           = 32 * n_vec;
	auto*          h_org_arr        = new uint32_t[n_tup];
	auto*          h_encoded_data   = new uint32_t[n_tup];
	auto*          h_transposed_arr = new uint32_t[vec_sz];
	auto*          h_unrsummed_arr  = new uint32_t[vec_sz];
	auto*          h_base_arr       = new uint32_t[n_base];
	uint64_t       encoded_arr_bsz  = n_tup * sizeof(int);
	uint32_t*      d_base_arr       = nullptr;
	uint32_t*      d_encoded_arr    = nullptr;

	for (uint8_t bitwidth {0}; bitwidth < 33; bitwidth++) {
		uint32_t bw = bitwidth;
		uint64_t sum {0};

		/* generate random numbers. */
		for (int i = 0; i < n_tup; i++) {
			if (bitwidth < 10) {
				h_org_arr[i] = bitwidth;
			} else {
				h_org_arr[i] = (i % 1024);
			}
			sum += h_org_arr[i];
		}

		FLS_SHOW(sum)
		FLS_SHOW(bw)

		auto in_als   = h_org_arr;
		auto out_als  = h_encoded_data;
		auto base_als = h_base_arr;

		for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
			generated::transpose::fallback::scalar::transpose_i(in_als, h_transposed_arr);

			generated::unrsum::fallback::scalar::unrsum(h_transposed_arr, h_unrsummed_arr);

			std::memcpy(base_als, h_transposed_arr, sizeof(uint32_t) * 32);

			generated::pack::fallback::scalar::pack(h_unrsummed_arr, out_als, bitwidth);

			in_als   = in_als + vec_sz;
			out_als  = out_als + (bitwidth * 32);
			base_als = base_als + 32;
		}

		d_encoded_arr = load_to_gpu(h_encoded_data, encoded_arr_bsz, g_allocator);
		d_base_arr    = load_to_gpu(h_base_arr, 32 * n_vec * sizeof(uint32_t), g_allocator);

		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		QueryMtd query_mtd {n_vec, bitwidth, n_tup, sum};
		for (int i {0}; i < repeat; ++i) {
			auto time = query_aggregate<32, 32>(d_base_arr, d_encoded_arr, query_mtd, g_allocator);
			FLS_SHOW(time)
		}

		CLEANUP(d_encoded_arr)
		CLEANUP(d_base_arr)
	}
}

int main() {
	/**/
	shared_memory_delta_with_aggregation();
}