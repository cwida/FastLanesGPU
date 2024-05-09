#include "config.hpp"
#include "crystal/crystal.cuh"
#include "cub/test/test_util.h"
#include "data/footer/ssb/ssb.hpp"
#include "fls_gen/unpack/unpack.cuh"
#include "gpu_utils.h"
#include "ssb_utils.h"
#include "tile_based/kernel.cuh"
#include <fls_gen/pack/pack.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace fastlanes::gpu;
using namespace fastlanes;

uint32_t bin_pack(uint32_t*& in, uint32_t*& out, uint32_t*& block_offsets, uint32_t tup_c) {
	uint32_t out_ofs = 0;

	uint32_t block_size      = 128;
	uint32_t miniblock_count = 4;
	uint32_t miniblock_size  = block_size / miniblock_count;
	uint32_t total_count     = tup_c;
	uint32_t first_val       = in[0];

	out[0] = block_size;
	out[1] = miniblock_count;
	out[2] = total_count;
	out[3] = first_val;

	out_ofs += 4;

	for (uint32_t idx = 0; idx < tup_c; idx += block_size) {
		uint32_t blk_idx       = idx / block_size;
		block_offsets[blk_idx] = out_ofs;

		// Find min val
		uint32_t min_val = in[0];
		for (int i = 1; i < block_size; i++) {
			if (in[i] < min_val) { min_val = in[i]; }
		}

		for (int i = 0; i < block_size; i++) {
			in[i] = in[i] - min_val;
		}

		uint32_t* miniblock_bitwidths = new uint32_t[miniblock_count];
		for (int i = 0; i < miniblock_count; i++) {
			miniblock_bitwidths[i] = 0;
		}

		for (uint32_t miniblock = 0; miniblock < miniblock_count; miniblock++) {
			for (uint32_t i = 0; i < miniblock_size; i++) {
				uint32_t bitwidth = uint32_t(ceil(log2(in[miniblock * miniblock_size + i] + 1)));
				if (bitwidth > miniblock_bitwidths[miniblock]) { miniblock_bitwidths[miniblock] = bitwidth; }
			}
		}

		// Extra for Simple BinPack
		uint32_t max_bitwidth = miniblock_bitwidths[0];
		for (int i = 1; i < miniblock_count; i++) {
			max_bitwidth = std::max(max_bitwidth, miniblock_bitwidths[i]);
		}
		for (int i = 0; i < miniblock_count; i++) {
			miniblock_bitwidths[i] = max_bitwidth;
		}

		out[out_ofs] = min_val;
		out_ofs++;

		out[out_ofs] = miniblock_bitwidths[0] + (miniblock_bitwidths[1] << 8) + (miniblock_bitwidths[2] << 16) +
		               (miniblock_bitwidths[3] << 24);
		out_ofs++;

		for (int miniblock = 0; miniblock < miniblock_count; miniblock++) {
			uint32_t bitwidth = miniblock_bitwidths[miniblock];
			uint32_t shift    = 0;
			for (int i = 0; i < miniblock_size; i++) {
				if (shift + bitwidth > 32) {
					if (shift != 32) { out[out_ofs] += in[miniblock * miniblock_size + i] << shift; }
					out_ofs++;
					shift        = (shift + bitwidth) & (32 - 1);
					out[out_ofs] = in[miniblock * miniblock_size + i] >> (bitwidth - shift);
				} else {
					out[out_ofs] += in[miniblock * miniblock_size + i] << shift;
					shift += bitwidth;
				}
			}
			out_ofs++;
		}

		// Increment the input pointer by block size
		in += block_size;
	}

	block_offsets[tup_c / block_size] = out_ofs;

	return out_ofs;
}

struct QueryMtd {
	n_t      n_vec;
	uint     bw;
	n_t      n_tup;
	uint64_t result;
};

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void
QueryKernel(const uint* col_block_start, const uint* col_data, QueryMtd query_mtd, unsigned long long* revenue) {
	int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
	// Load a segment of consecutive items that are blocked across threads
	uint32_t items[ITEMS_PER_THREAD];

	static __shared__ uint32_t unpacked[512];

	long long sum = 0;

	int tile_offset    = blockIdx.x * TILE_SIZE;
	int num_tiles      = (query_mtd.n_tup + TILE_SIZE - 1) / TILE_SIZE;
	int num_tile_items = TILE_SIZE;
	if (blockIdx.x == num_tiles - 1) { num_tile_items = query_mtd.n_tup - tile_offset; }

	LoadBinPack<BLOCK_THREADS, ITEMS_PER_THREAD>(col_block_start, col_data, unpacked, items);

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
float query_aggregate(const uint*                  col_block_start,
                      const uint*                  col_data,
                      QueryMtd                     hardcoded,
                      cub::CachingDeviceAllocator& g_allocator) {
	int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

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
	    <<<hardcoded.n_vec, BLOCK_THREADS>>>(col_block_start, col_data, hardcoded, d_sum);

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

n_t bitpacked_vec_n_tup(uint bitdwith) {
	/**/
	return bitdwith * 32;
}
namespace tile_based {
template <typename T>
T* loadColumnToGPU(const T* src, int len) {
	T* dest = nullptr;
	cudaMalloc((void**)&dest, sizeof(T) * len);
	CubDebugExit(cudaMemcpy(dest, src, sizeof(T) * len, cudaMemcpyHostToDevice));
	return dest;
}

} // namespace tile_based

void shared_memory_bitpacking_with_aggregation() {

	constexpr uint64_t n_vec           = 2 * 256 * 1024;
	constexpr uint64_t vec_sz          = 512;
	constexpr uint64_t n_tup           = vec_sz * n_vec;
	auto*              h_org_arr       = new uint32_t[n_tup];
	auto*              h_encoded_data  = new uint32_t[n_tup];
	size_t             repeat          = 3;
	int                block_size      = 128;
	int                elem_per_thread = 4;
	int                tile_size       = block_size * elem_per_thread;
	int                num_blocks      = n_tup / block_size;
	auto*              encoded_data    = new uint32_t[n_tup]();
	uint64_t           ofs_c           = num_blocks + 1;
	auto*              ofs_arr         = new uint32_t[ofs_c]();
	auto*              copy_data       = new uint32_t[n_tup];

	for (uint bitwidth {0}; bitwidth < 33; bitwidth++) {
		uint32_t bw              = bitwidth;
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

		/* Data needs to be copied. the encoding change the original data. */
		memcpy(copy_data, h_org_arr, n_tup * sizeof(int));

		uint32_t encoded_data_bsz = bin_pack(copy_data, encoded_data, ofs_arr, n_tup);

		tile_based::encoded_column h_col {reinterpret_cast<uint*>(ofs_arr), //
		                                  reinterpret_cast<uint*>(encoded_data),
		                                  n_tup * 4};

		uint* d_col_block_start = tile_based::loadColumnToGPU<uint>(h_col.block_start, num_blocks + 1);
		uint* d_col_data        = tile_based::loadColumnToGPU<uint>(h_col.data, h_col.data_size / 4);
		tile_based::encoded_column d_col {d_col_block_start, d_col_data};

		QueryMtd query_mtd {n_vec, bitwidth, n_tup, sum};
		for (int i {0}; i < repeat; ++i) {
			auto time = query_aggregate<128, 4>(d_col_block_start, d_col_data, query_mtd, g_allocator);
			FLS_SHOW(time)
		}

		CLEANUP(d_col_block_start)
		CLEANUP(d_col_data)
	}
}

int main() {
	/**/
	shared_memory_bitpacking_with_aggregation();
}