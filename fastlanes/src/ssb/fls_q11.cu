// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include "crystal-opt/crystal.cuh"
#include "crystal_ssb_utils.h"
#include "cub/test/test_util.h"
#include "fls_gen/unpack/unpack_fused.cuh"
#include "gpu_utils.h"
#include "ssb_utils.h"
#include "gtest/gtest.h"
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

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void QueryKernel_v1(const int*          enc_lo_orderdate,
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

	long long sum = 0;

	int tile_offset    = blockIdx.x * TILE_SIZE;
	int num_tiles      = (query_mtd.n_tup_line_order + TILE_SIZE - 1) / TILE_SIZE;
	int num_tile_items = TILE_SIZE;
	if (blockIdx.x == num_tiles - 1) { num_tile_items = query_mtd.n_tup_line_order - tile_offset; }

	int orderdate_tile_offset = blockIdx.x * query_mtd.lo_orderdate_bw * 32;
	unpack_device(enc_lo_orderdate + orderdate_tile_offset, items, query_mtd.lo_orderdate_bw);
	BlockPredGT<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, 19930000 - query_mtd.lo_orderdate_min, selection_flags, num_tile_items);
	BlockPredAndLT<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, 19940000 - query_mtd.lo_orderdate_min, selection_flags, num_tile_items);

	int quantity_tile_offset = blockIdx.x * query_mtd.lo_quantity_bw * 32;
	unpack_device(enc_lo_quantity + quantity_tile_offset, items, query_mtd.lo_quantity_bw);
	BlockPredAndLT<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 25, selection_flags, num_tile_items);

	int discount_tile_offset = blockIdx.x * query_mtd.lo_discount_bw * 32;
	unpack_device(enc_lo_discount + discount_tile_offset, items, query_mtd.lo_discount_bw);
	BlockPredAndGTE<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1, selection_flags, num_tile_items);
	BlockPredAndLTE<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 3, selection_flags, num_tile_items);

	int extendedprice_tile_offset = blockIdx.x * query_mtd.lo_extendedprice_bw * 32;
	unpack_device(enc_lo_extendedprice + extendedprice_tile_offset, items2, query_mtd.lo_extendedprice_bw);

#pragma unroll
	for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
		if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
			if (selection_flags[ITEM]) sum += items[ITEM] * items2[ITEM];
	}

	__syncthreads();

	static __shared__ long long buffer[32];
	unsigned long long aggregate = BlockSum<long long, BLOCK_THREADS, ITEMS_PER_THREAD>(sum, (long long*)buffer);
	__syncthreads();

	if (threadIdx.x == 0) { atomicAdd(revenue, aggregate); }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void QueryKernel_v2(const int*          enc_lo_orderdate,
                               const int*          enc_lo_discount,
                               const int*          enc_lo_quantity,
                               int*                lo_extendedprice,
                               ssb::SSB            query_mtd,
                               unsigned long long* revenue) {
	int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
	// Load a segment of consecutive items that are blocked across threads
	int items[ITEMS_PER_THREAD];
	int selection_flags[ITEMS_PER_THREAD];
	int items2[ITEMS_PER_THREAD];

	long long sum = 0;

	int tile_offset    = blockIdx.x * TILE_SIZE;
	int num_tiles      = (query_mtd.n_tup_line_order + TILE_SIZE - 1) / TILE_SIZE;
	int num_tile_items = TILE_SIZE;
	if (blockIdx.x == num_tiles - 1) { num_tile_items = query_mtd.n_tup_line_order - tile_offset; }

	int orderdate_tile_offset = blockIdx.x * query_mtd.lo_orderdate_bw * 32;
	unpack_device(enc_lo_orderdate + orderdate_tile_offset, items, query_mtd.lo_orderdate_bw);
	BlockPredGT<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, 19930000 - query_mtd.lo_orderdate_min, selection_flags, num_tile_items);
	BlockPredAndLT<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, 19940000 - query_mtd.lo_orderdate_min, selection_flags, num_tile_items);

	int quantity_tile_offset = blockIdx.x * query_mtd.lo_quantity_bw * 32;
	unpack_device(enc_lo_quantity + quantity_tile_offset, items, query_mtd.lo_quantity_bw);
	BlockPredAndLT<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 25, selection_flags, num_tile_items);

	int discount_tile_offset = blockIdx.x * query_mtd.lo_discount_bw * 32;
	unpack_device(enc_lo_discount + discount_tile_offset, items, query_mtd.lo_discount_bw);
	BlockPredAndGTE<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1, selection_flags, num_tile_items);
	BlockPredAndLTE<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 3, selection_flags, num_tile_items);

	BlockPredLoad<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    lo_extendedprice + tile_offset, items2, num_tile_items, selection_flags);

#pragma unroll
	for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
		if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
			if (selection_flags[ITEM]) sum += items[ITEM] * items2[ITEM];
	}

	__syncthreads();

	static __shared__ long long buffer[32];
	unsigned long long aggregate = BlockSum<long long, BLOCK_THREADS, ITEMS_PER_THREAD>(sum, (long long*)buffer);
	__syncthreads();

	if (threadIdx.x == 0) { atomicAdd(revenue, aggregate); }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
void query(int*                         lo_orderdate,
           int*                         lo_discount,
           int*                         lo_quantity,
           int*                         lo_extendedprice,
           ssb::SSBQuery1               query_mtd,
           cub::CachingDeviceAllocator& g_allocator,
           int                          version) {

	unsigned long long* d_sum = NULL;
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_sum, sizeof(long long)));
	cudaMemset(d_sum, 0, sizeof(long long));

	// Run
	int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
	int num_blocks = (query_mtd.ssb.n_tup_line_order + tile_items - 1) / tile_items;

	if (version == 1) {
		QueryKernel_v1<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(
		    lo_orderdate, lo_discount, lo_quantity, lo_extendedprice, query_mtd.ssb, d_sum);
	} else if (version == 2) {
		QueryKernel_v2<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(
		    lo_orderdate, lo_discount, lo_quantity, lo_extendedprice, query_mtd.ssb, d_sum);
	} else {
		throw std::runtime_error("this version does not exist");
	}

	unsigned long long revenue;
	CubDebugExit(cudaMemcpy(&revenue, d_sum, sizeof(long long), cudaMemcpyDeviceToHost));

	/*Check the result*/
	ASSERT_EQ(revenue, query_mtd.result);

	CLEANUP(d_sum);
}

int main(int argc, char* argv[]) {
	int version = 0;
	version     = std::stoi(argv[1]);

	const auto query_mtd  = ssb::ssb_q11_10;
	auto       hard_coded = query_mtd.ssb;

	int* h_lo_orderdate     = loadColumn<int>("lo_orderdate", LO_LEN);
	int* h_lo_discount      = loadColumn<int>("lo_discount", LO_LEN);
	int* h_lo_quantity      = loadColumn<int>("lo_quantity", LO_LEN);
	int* h_lo_extendedprice = loadColumn<int>("lo_extendedprice", LO_LEN);

	auto n_vec = hard_coded.n_vec;

	int* tmp = new int[n_vec * 1024];
	for (size_t i {0}; i < LO_LEN; ++i) {
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

		generated::pack::fallback::scalar::pack(extendedprice_in, extendedprice_out, hard_coded.lo_extendedprice_bw);
		extendedprice_in  = extendedprice_in + 1024;
		extendedprice_out = extendedprice_out + (hard_coded.lo_extendedprice_bw * 32);
	}

	int* d_lo_orderdate = loadToGPU<int32_t>(h_enc_lo_orderdate, hard_coded.n_tup_line_order, g_allocator);
	int* d_lo_discount  = loadToGPU<int32_t>(h_enc_lo_discount, hard_coded.n_tup_line_order, g_allocator);
	int* d_lo_quantity  = loadToGPU<int32_t>(h_enc_lo_quantity, hard_coded.n_tup_line_order, g_allocator);
	int* d_lo_extendedprice;

	if (version == 1) {
		d_lo_extendedprice = loadToGPU<int32_t>(h_enc_lo_extendedprice, hard_coded.n_tup_line_order, g_allocator);
	} else if (version == 2) {
		d_lo_extendedprice = loadToGPU<int32_t>(h_lo_extendedprice, hard_coded.n_tup_line_order, g_allocator);
	} else {
		throw std::runtime_error("this version does not exist");
	}

	query<32, 32>(d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice, query_mtd, g_allocator, version);

	return 0;
}