// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include "crystal.cuh"
#include "cub/test/test_util.h"
#include "data/footer/ssb/ssb.hpp"
#include "fls_gen/unpack/unpack.cuh"
#include "gpu_utils.h"
#include "ssb_utils.h"
#include <fls_gen/pack/pack.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace fastlanes::gpu;
using namespace fastlanes;

/*
select sum(lo_extendedprice * lo_discount) as revenue
from lineorder
where lo_orderdate >= 19930000 and lo_orderdate <= 19940000 and lo_discount>=1
and lo_discount<=3
and lo_quantity<25;
*/

// Query kernel for crystal opt
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void QueryKernelOpt(const int*          enc_lo_orderdate,
                               const int*          enc_lo_discount,
                               const int*          enc_lo_quantity,
                               const int*          enc_lo_extendedprice,
                               ssb::SSB            query_mtd,
                               unsigned long long* revenue) {

	int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

	// Load a segment of consecutive items that are blocked across threads
	int items[ITEMS_PER_THREAD];
	int selection_flags[ITEMS_PER_THREAD];
	int items2[ITEMS_PER_THREAD];

	static __shared__ int unpacked[1024];

	long long sum = 0;

	int tile_offset    = blockIdx.x * TILE_SIZE;
	int num_tiles      = (query_mtd.n_tup_line_order + TILE_SIZE - 1) / TILE_SIZE;
	int num_tile_items = TILE_SIZE;
	if (blockIdx.x == num_tiles - 1) { num_tile_items = query_mtd.n_tup_line_order - tile_offset; }

	int orderdate_tile_offset = blockIdx.x * query_mtd.lo_orderdate_bw * 32;
	unpack_device(enc_lo_orderdate + orderdate_tile_offset, unpacked, query_mtd.lo_orderdate_bw);

	BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(unpacked, items, num_tile_items);
	BlockPredGT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, 19930000 - query_mtd.lo_orderdate_min, selection_flags, num_tile_items);
	BlockPredAndLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, 19940000 - query_mtd.lo_orderdate_min, selection_flags, num_tile_items);

	int quantity_tile_offset = blockIdx.x * query_mtd.lo_quantity_bw * 32;
	unpack_device(enc_lo_quantity + quantity_tile_offset, unpacked, query_mtd.lo_quantity_bw);

	BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(unpacked, items, num_tile_items, selection_flags);
	BlockPredAndLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, 25, selection_flags, num_tile_items);

	int discount_tile_offset = blockIdx.x * query_mtd.lo_discount_bw * 32;
	unpack_device(enc_lo_discount + discount_tile_offset, unpacked, query_mtd.lo_discount_bw);

	BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(unpacked, items, num_tile_items, selection_flags);
	BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, 1, selection_flags, num_tile_items);
	BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, 3, selection_flags, num_tile_items);

	int extendedprice_tile_offset = blockIdx.x * query_mtd.lo_extendedprice_bw * 32;
	unpack_device(enc_lo_extendedprice + extendedprice_tile_offset, unpacked, query_mtd.lo_extendedprice_bw);

	BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(unpacked, items2, num_tile_items, selection_flags);

#pragma unroll
	for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
		if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
			if (selection_flags[ITEM])
				sum += items[ITEM] * items2[ITEM];
	}

	__syncthreads();

	static __shared__ long long buffer[32];
	unsigned long long aggregate =
	    BlockSum<long long, BLOCK_THREADS, ITEMS_PER_THREAD>(sum,
	                                                         (long long *)buffer);
	__syncthreads();

	if (threadIdx.x == 0) {
		atomicAdd(revenue, aggregate);
	}
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
float query(int*                         lo_orderdate,
            int*                         lo_discount,
            int*                         lo_quantity,
            int*                         lo_extendedprice,
            ssb::SsbQueryMtd             query_mtd,
            cub::CachingDeviceAllocator& g_allocator) {
	SETUP_TIMING();

	float                                     time_query;
	chrono::high_resolution_clock::time_point st, finish;
	st = chrono::high_resolution_clock::now();

	cudaEventRecord(start, 0);

	unsigned long long* d_sum = NULL;
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_sum, sizeof(long long)));

	cudaMemset(d_sum, 0, sizeof(long long));

	// Run
	int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
	int num_blocks = (query_mtd.ssb.n_tup_line_order + tile_items - 1) / tile_items;
	QueryKernelOpt<BLOCK_THREADS, ITEMS_PER_THREAD>
	    <<<num_blocks, BLOCK_THREADS>>>(lo_orderdate, lo_discount, lo_quantity, lo_extendedprice, query_mtd.ssb, d_sum);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_query, start, stop);

	unsigned long long revenue;
	CubDebugExit(cudaMemcpy(&revenue, d_sum, sizeof(long long), cudaMemcpyDeviceToHost));

	finish                             = chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = finish - st;

	double total_time_taken {diff.count() * 1000};
	FLS_SHOW(total_time_taken)

	/*Check the result*/
	FLS_SHOW(revenue)
	if (revenue != query_mtd.result) { throw std::runtime_error("RESULT INCOREECT!"); }
	FLS_SUCCESS(query_mtd.ssb.name)

	CLEANUP(d_sum);

	return time_query;
}

int main() {
	int  num_trials  = 1;
	auto queries_mtd = {
	    //
	    //	    ssb::ssb_q11_1,
	    //	    ssb::ssb_q11_0_1
	    ssb::ssb_q11_10,
	    //
	};
	for (const auto query_mtd : queries_mtd) {
		auto       hard_coded         = query_mtd.ssb;
		const auto lineorder          = Reader::ReadCsv(hard_coded.dir);
		const auto rowgroup           = lineorder->GetFirstRowGroup();
		const int* h_lo_orderdate     = rowgroup->GetChunk("LO_ORDERDATE").GetData<int32_t>();
		const int* h_lo_discount      = rowgroup->GetChunk("LO_DISCOUNT").GetData<int32_t>();
		const int* h_lo_quantity      = rowgroup->GetChunk("LO_QUANTITY").GetData<int32_t>();
		const int* h_lo_extendedprice = rowgroup->GetChunk("LO_EXTENDEDPRICE").GetData<int32_t>();

		int32_t lo_orderdate_min = h_lo_orderdate[0];
		int32_t lo_orderdate_max = h_lo_orderdate[0];
		for (size_t i {0}; i < hard_coded.n_tup_line_order; ++i) {
			lo_orderdate_min = std::min(lo_orderdate_min, h_lo_orderdate[i]);
			lo_orderdate_max = std::max(lo_orderdate_max, h_lo_orderdate[i]);
		}
		FLS_SHOW(lo_orderdate_min)
		FLS_SHOW(lo_orderdate_max)

		int32_t lo_discount_min = h_lo_discount[0];
		int32_t lo_discount_max = h_lo_discount[0];
		for (size_t i {0}; i < hard_coded.n_tup_line_order; ++i) {
			lo_discount_min = std::min(lo_discount_min, h_lo_discount[i]);
			lo_discount_max = std::max(lo_discount_max, h_lo_discount[i]);
		}
		FLS_SHOW(lo_discount_min)
		FLS_SHOW(lo_discount_max)

		int32_t lo_quantity_min = h_lo_quantity[0];
		int32_t lo_quantity_max = h_lo_quantity[0];
		for (size_t i {0}; i < hard_coded.n_tup_line_order; ++i) {
			lo_quantity_min = std::min(lo_quantity_min, h_lo_quantity[i]);
			lo_quantity_max = std::max(lo_quantity_max, h_lo_quantity[i]);
		}
		FLS_SHOW(lo_quantity_min)
		FLS_SHOW(lo_quantity_max)

		int32_t lo_extendedprice_min = h_lo_extendedprice[0];
		int32_t lo_extendedprice_max = h_lo_extendedprice[0];
		for (size_t i {0}; i < hard_coded.n_tup_line_order; ++i) {
			lo_extendedprice_min = std::min(lo_extendedprice_min, h_lo_extendedprice[i]);
			lo_extendedprice_max = std::max(lo_extendedprice_max, h_lo_extendedprice[i]);
		}
		FLS_SHOW(lo_extendedprice_min)
		FLS_SHOW(lo_extendedprice_max)

		uint16_t x = RANGE_BIT(lo_orderdate_max - lo_orderdate_min);
		FLS_SHOW(x)

		auto n_vec = hard_coded.n_vec;

		int* tmp = new int[n_vec * 1024];
		for (size_t i {0}; i < rowgroup->RowCount(); ++i) {
			tmp[i] = h_lo_orderdate[i] - hard_coded.lo_orderdate_min;
		}

		const int* h_enc_lo_orderdate     = new int[n_vec * 1024];
		const int* h_enc_lo_discount      = new int[n_vec * 1024];
		const int* h_enc_lo_quantity      = new int[n_vec * 1024];
		const int* h_enc_lo_extendedprice = new int[n_vec * 1024];

		auto* orderdate_in     = const_cast<const int32_t*>(tmp);
		auto* discount_in      = const_cast<int32_t*>(h_lo_discount);
		auto* quantity_in      = const_cast<int32_t*>(h_lo_quantity);
		auto* extendedprice_in = const_cast<int32_t*>(h_lo_extendedprice);

		auto* orderdate_out     = const_cast<int32_t*>(h_enc_lo_orderdate);
		auto* discount_out      = const_cast<int32_t*>(h_enc_lo_discount);
		auto* quantity_out      = const_cast<int32_t*>(h_enc_lo_quantity);
		auto* extendedprice_out = const_cast<int32_t*>(h_enc_lo_extendedprice);

		for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
			generated::pack::fallback::scalar::pack(orderdate_in, orderdate_out, hard_coded.lo_orderdate_bw);
			orderdate_in  = orderdate_in + 1024;
			orderdate_out = orderdate_out + (hard_coded.lo_orderdate_bw * 32);

			generated::pack::fallback::scalar::pack(discount_in, discount_out, hard_coded.lo_discount_bw);
			discount_in  = discount_in + 1024;
			discount_out = discount_out + (hard_coded.lo_discount_bw * 32);

			generated::pack::fallback::scalar::pack(quantity_in, quantity_out, hard_coded.lo_quantity_bw);
			quantity_in  = quantity_in + 1024;
			quantity_out = quantity_out + (hard_coded.lo_quantity_bw * 32);

			generated::pack::fallback::scalar::pack(
			    extendedprice_in, extendedprice_out, hard_coded.lo_extendedprice_bw);
			extendedprice_in  = extendedprice_in + 1024;
			extendedprice_out = extendedprice_out + (hard_coded.lo_extendedprice_bw * 32);
		}

		FLS_LOG("LOADED DATA")

		int* d_lo_orderdate     = loadToGPU<int32_t>(h_enc_lo_orderdate, hard_coded.n_tup_line_order, g_allocator);
		int* d_lo_discount      = loadToGPU<int32_t>(h_enc_lo_discount, hard_coded.n_tup_line_order, g_allocator);
		int* d_lo_quantity      = loadToGPU<int32_t>(h_enc_lo_quantity, hard_coded.n_tup_line_order, g_allocator);
		int* d_lo_extendedprice = loadToGPU<int32_t>(h_enc_lo_extendedprice, hard_coded.n_tup_line_order, g_allocator);

		FLS_LOG("LOADED DATA TO GPU")

		for (int n = 0; n < num_trials; n++) {
			auto t =
			    query<32, 32>(d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice, query_mtd, g_allocator);
			FLS_RESULT(t)
		}
	}
	return 0;
}