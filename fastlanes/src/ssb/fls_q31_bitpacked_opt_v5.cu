// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR
#define SORTED

#include "crystal/crystal.cuh"
#include "crystal_ssb_utils.h"
#include "cub/test/test_util.h"
#include "fastlanes/join.cuh"
#include "fls_gen/unpack/unpack.cuh"
#include "gpu_utils.h"
#include "ssb_utils.h"
#include "gtest/gtest.h"
#include <cub/util_allocator.cuh>
#include <fls_gen/pack/pack.hpp>
#include <fls_gen/unpack/hardcoded_16.cuh>
#include <iostream>
#include <query/query_31.hpp>
#include <stdio.h>

using namespace std;
using namespace fastlanes::gpu;
using namespace fastlanes;

using namespace std;

auto query_mtd = ssb::ssb_q31_10;

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_v4(int* lo_orderdate,
                         int* lo_orderdate_bw,
                         int* lo_orderdate_base,
                         int* lo_orderdate_offset,
                         int* lo_custkey,
                         int* lo_suppkey,
                         int* lo_revenue,
                         int  lo_len,
                         int* ht_s,
                         int  s_len,
                         int* ht_c,
                         int  c_len,
                         int* ht_d,
                         int  d_len,
                         int* res) {

	int mtd_offset = blockIdx.x / 4;

	constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
	// Load a segment of consecutive items that are blocked across threads
	int items[ITEMS_PER_THREAD];
	int selection_flags[ITEMS_PER_THREAD];
	int c_nation[ITEMS_PER_THREAD];
	int s_nation[ITEMS_PER_THREAD];
	int year[ITEMS_PER_THREAD];
	int revenue[ITEMS_PER_THREAD];

	int tile_offset    = blockIdx.x * TILE_SIZE;
	int num_tiles      = (lo_len + TILE_SIZE - 1) / TILE_SIZE;
	int num_tile_items = TILE_SIZE;

	if (blockIdx.x == num_tiles - 1) { num_tile_items = lo_len - tile_offset; }

	InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

	int suppkey_tile_offset = blockIdx.x * query_mtd.ssb.lo_chosen_suppkey_bw * 8;
	unpack_8_at_a_time::unpack_device(lo_suppkey + suppkey_tile_offset, items, query_mtd.ssb.lo_chosen_suppkey_bw);
	BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, s_nation, selection_flags, ht_s, s_len, num_tile_items);

	int custkey_tile_offset = blockIdx.x * query_mtd.ssb.lo_chosen_custkey_bw * 8;
	unpack_8_at_a_time::unpack_device(lo_custkey + custkey_tile_offset, items, query_mtd.ssb.lo_chosen_custkey_bw);
	BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, c_nation, selection_flags, ht_c, c_len, num_tile_items);

	int bw                    = lo_orderdate_bw[mtd_offset];
	int base                  = lo_orderdate_base[mtd_offset];
	int orderdate_tile_offset = lo_orderdate_offset[mtd_offset] + (blockIdx.x % 4) * bw * 8;

	unpack_8_at_a_time::unpack_device(lo_orderdate + orderdate_tile_offset, items, bw);
#pragma unroll
	for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
		items[ITEM] = items[ITEM] + base;
	}

	BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, year, selection_flags, ht_d, d_len, 0, num_tile_items);

	// BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_revenue + tile_offset, revenue, num_tile_items);
	BlockPredLoad<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    lo_revenue + tile_offset, revenue, num_tile_items, selection_flags);

#pragma unroll
	for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
		if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_tile_items) {
			if (selection_flags[ITEM]) {
				int hash = (s_nation[ITEM] * 25 * 7 + c_nation[ITEM] * 7 + (year[ITEM] - 1992)) %
				           ((1998 - 1992 + 1) * 25 * 25);
				res[hash * 6]     = year[ITEM];
				res[hash * 6 + 1] = c_nation[ITEM];
				res[hash * 6 + 2] = s_nation[ITEM];
				/*atomicAdd(&res[hash * 6 + 4], revenue[ITEM]);*/
				atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(revenue[ITEM]));
			}
		}
	}
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void
build_hashtable_s(int* filter_col, int* dim_key, int* dim_val, int num_tuples, int* hash_table, int num_slots) {
	int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

	int items[ITEMS_PER_THREAD];
	int items2[ITEMS_PER_THREAD];
	int selection_flags[ITEMS_PER_THREAD];

	int tile_offset    = blockIdx.x * TILE_SIZE;
	int num_tiles      = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
	int num_tile_items = TILE_SIZE;

	if (blockIdx.x == num_tiles - 1) { num_tile_items = num_tuples - tile_offset; }

	BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset, items, num_tile_items);
	BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 2, selection_flags, num_tile_items);

	BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items, num_tile_items);
	BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2, num_tile_items);
	BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, items2, selection_flags, hash_table, num_slots, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void
build_hashtable_c(int* filter_col, int* dim_key, int* dim_val, int num_tuples, int* hash_table, int num_slots) {
	int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

	int items[ITEMS_PER_THREAD];
	int items2[ITEMS_PER_THREAD];
	int selection_flags[ITEMS_PER_THREAD];

	int tile_offset    = blockIdx.x * TILE_SIZE;
	int num_tiles      = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
	int num_tile_items = TILE_SIZE;

	if (blockIdx.x == num_tiles - 1) { num_tile_items = num_tuples - tile_offset; }

	BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset, items, num_tile_items);
	BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 2, selection_flags, num_tile_items);

	BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items, num_tile_items);
	BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2, num_tile_items);
	BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, items2, selection_flags, hash_table, num_slots, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void
build_hashtable_d(int* dim_key, int* dim_val, int num_tuples, int* hash_table, int num_slots, int val_min) {
	int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

	int items[ITEMS_PER_THREAD];
	int items2[ITEMS_PER_THREAD];
	int selection_flags[ITEMS_PER_THREAD];

	int tile_offset    = blockIdx.x * TILE_SIZE;
	int num_tiles      = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
	int num_tile_items = TILE_SIZE;

	if (blockIdx.x == num_tiles - 1) { num_tile_items = num_tuples - tile_offset; }

	BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items, num_tile_items);
	BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1992, selection_flags, num_tile_items);
	BlockPredLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1997, selection_flags, num_tile_items);

	BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items2, num_tile_items);
	BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items2, items, selection_flags, hash_table, num_slots, 19920101, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
void runQuery(int*                         lo_orderdate,
              int*                         d_lo_orderdate_bw,
              int*                         d_lo_orderdate_base,
              int*                         d_lo_orderdate_offset,
              int*                         lo_custkey,
              int*                         lo_suppkey,
              int*                         lo_revenue,
              int                          lo_len,
              int*                         d_datekey,
              int*                         d_year,
              int                          d_len,
              int*                         s_suppkey,
              int*                         s_region,
              int*                         s_nation,
              int                          s_len,
              int*                         c_custkey,
              int*                         c_region,
              int*                         c_nation,
              int                          c_len,
              cub::CachingDeviceAllocator& g_allocator) {

	int *ht_d, *ht_c, *ht_s;
	int  d_val_len = 19981230 - 19920101 + 1;
	CubDebugExit(g_allocator.DeviceAllocate((void**)&ht_d, 2 * d_val_len * sizeof(int)));
	CubDebugExit(g_allocator.DeviceAllocate((void**)&ht_s, 2 * s_len * sizeof(int)));
	CubDebugExit(g_allocator.DeviceAllocate((void**)&ht_c, 2 * c_len * sizeof(int)));

	CubDebugExit(cudaMemset(ht_d, 0, 2 * d_val_len * sizeof(int)));
	CubDebugExit(cudaMemset(ht_s, 0, 2 * s_len * sizeof(int)));

	int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
	build_hashtable_s<BLOCK_THREADS, ITEMS_PER_THREAD>
	    <<<(s_len + tile_items - 1) / tile_items, BLOCK_THREADS>>>(s_region, s_suppkey, s_nation, s_len, ht_s, s_len);
	/*CHECK_ERROR();*/

	build_hashtable_c<BLOCK_THREADS, ITEMS_PER_THREAD>
	    <<<(c_len + tile_items - 1) / tile_items, BLOCK_THREADS>>>(c_region, c_custkey, c_nation, c_len, ht_c, c_len);
	/*CHECK_ERROR();*/

	int d_val_min = 19920101;
	build_hashtable_d<BLOCK_THREADS, ITEMS_PER_THREAD><<<(d_len + tile_items - 1) / tile_items, BLOCK_THREADS>>>(
	    d_datekey, d_year, d_len, ht_d, d_val_len, d_val_min);
	/*CHECK_ERROR();*/

	int* res;
	int  res_size       = ((1998 - 1992 + 1) * 25 * 25);
	int  res_array_size = res_size * 6;
	CubDebugExit(g_allocator.DeviceAllocate((void**)&res, res_array_size * sizeof(int)));

	CubDebugExit(cudaMemset(res, 0, res_array_size * sizeof(int)));

	// Run
	probe_v4<BLOCK_THREADS, ITEMS_PER_THREAD>
	    <<<(lo_len + tile_items - 1) / tile_items, BLOCK_THREADS>>>(lo_orderdate,
	                                                                d_lo_orderdate_bw,
	                                                                d_lo_orderdate_base,
	                                                                d_lo_orderdate_offset,
	                                                                lo_custkey,
	                                                                lo_suppkey,
	                                                                lo_revenue,
	                                                                lo_len,
	                                                                ht_s,
	                                                                s_len,
	                                                                ht_c,
	                                                                c_len,
	                                                                ht_d,
	                                                                d_val_len,
	                                                                res);

	int* h_res = new int[res_array_size];
	CubDebugExit(cudaMemcpy(h_res, res, res_array_size * sizeof(int), cudaMemcpyDeviceToHost));

	ssb::SSBQuery3ResultTable result_of_query;
	for (int i = 0; i < res_size; i++) {
		if (h_res[6 * i] != 0) {
			result_of_query.emplace_back(h_res[6 * i],
			                             h_res[6 * i + 1],
			                             h_res[6 * i + 2],
			                             reinterpret_cast<unsigned long long*>(&h_res[6 * i + 4])[0]);
		}
	}

	ASSERT_EQ(result_of_query.size(), ssb::ssb_q31_10.reuslt.size());
	ASSERT_EQ(result_of_query, ssb::ssb_q31_10.reuslt);

	delete[] h_res;
}

/**
 * Main
 */
int main(int argc, char* argv[]) {
	auto hard_coded     = query_mtd.ssb;
	int* h_lo_orderdate = loadColumn<int>("lo_orderdate", LO_LEN);
	int* h_lo_custkey   = loadColumn<int>("lo_custkey", LO_LEN);
	int* h_lo_suppkey   = loadColumn<int>("lo_suppkey", LO_LEN);
	int* h_lo_revenue   = loadColumn<int>("lo_revenue", LO_LEN);

	auto n_vec = hard_coded.n_vec;

	int* tmp = new int[n_vec * 1024];
	for (size_t i {0}; i < LO_LEN; ++i) {
		tmp[i] = h_lo_orderdate[i] - hard_coded.lo_orderdate_min;
	}

	const int* h_enc_lo_orderdate = new int[n_vec * 1024];
	const int* h_enc_lo_custkey   = new int[n_vec * 1024];
	const int* h_enc_lo_suppkey   = new int[n_vec * 1024];
	const int* h_enc_lo_revenue   = new int[n_vec * 1024];

	int* h_lo_orderdate_base   = new int[n_vec];
	int* h_lo_orderdate_bw     = new int[n_vec];
	int* h_lo_orderdate_offset = new int[n_vec];

	auto* orderdate_in = const_cast<int32_t*>(tmp);
	auto* custkey_in   = const_cast<int32_t*>(h_lo_custkey);
	auto* suppkey_in   = const_cast<int32_t*>(h_lo_suppkey);
	auto* revenue_in   = const_cast<int32_t*>(h_lo_revenue);

	auto* orderdate_out = const_cast<int32_t*>(h_enc_lo_orderdate);
	auto* custkey_out   = const_cast<int32_t*>(h_enc_lo_custkey);
	auto* suppkey_out   = const_cast<int32_t*>(h_enc_lo_suppkey);
	auto* revenue_out   = const_cast<int32_t*>(h_enc_lo_revenue);

	constexpr int SF10_LAST_VECTOR_IDX = 58580;
	constexpr int LAST_VECTOR_SIZE     = 294;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		if (vec_idx == SF10_LAST_VECTOR_IDX) { set_zero_after<1024>(orderdate_in, LAST_VECTOR_SIZE); }

		h_lo_orderdate_base[vec_idx] = find_base<1024>(orderdate_in);
		subtract_base<1024>(orderdate_in, h_lo_orderdate_base[vec_idx]);
		h_lo_orderdate_bw[vec_idx] = find_bw<1024>(orderdate_in);

		if (vec_idx + 1 < n_vec) {
			h_lo_orderdate_offset[vec_idx + 1] = h_lo_orderdate_offset[vec_idx] + (h_lo_orderdate_bw[vec_idx] * 32);
		}

		if (h_lo_orderdate_bw[vec_idx] > 16) {
			std::cout << h_lo_orderdate_bw[vec_idx] << " bigger than 16 is not possible in orderdate! \n";
			exit(-2);
		}

		generated::pack::fallback::scalar::pack(orderdate_in, orderdate_out, h_lo_orderdate_bw[vec_idx]);
		orderdate_in  = orderdate_in + 1024;
		orderdate_out = orderdate_out + (h_lo_orderdate_bw[vec_idx] * 32);

		generated::pack::fallback::scalar::pack(custkey_in, custkey_out, hard_coded.lo_chosen_custkey_bw);
		custkey_in  = custkey_in + 1024;
		custkey_out = custkey_out + (hard_coded.lo_chosen_custkey_bw * 32);

		generated::pack::fallback::scalar::pack(suppkey_in, suppkey_out, hard_coded.lo_chosen_suppkey_bw);
		suppkey_in  = suppkey_in + 1024;
		suppkey_out = suppkey_out + (hard_coded.lo_chosen_suppkey_bw * 32);

		generated::pack::fallback::scalar::pack(revenue_in, revenue_out, hard_coded.lo_revenue_bw);
		revenue_in  = revenue_in + 1024;
		revenue_out = revenue_out + (hard_coded.lo_revenue_bw * 32);
	}

	int* d_lo_orderdate = loadToGPU<int32_t>(h_enc_lo_orderdate, hard_coded.n_tup_line_order, g_allocator);
	int* d_lo_custkey   = loadToGPU<int32_t>(h_enc_lo_custkey, hard_coded.n_tup_line_order, g_allocator);
	int* d_lo_suppkey   = loadToGPU<int32_t>(h_enc_lo_suppkey, hard_coded.n_tup_line_order, g_allocator);
	int* d_lo_revenue;

	d_lo_revenue = loadToGPU<int32_t>(h_lo_revenue, hard_coded.n_tup_line_order, g_allocator);

	int* h_d_datekey = loadColumn<int>("d_datekey", D_LEN);
	int* h_d_year    = loadColumn<int>("d_year", D_LEN);

	int* h_s_suppkey = loadColumn<int>("s_suppkey", S_LEN);
	int* h_s_nation  = loadColumn<int>("s_nation", S_LEN);
	int* h_s_region  = loadColumn<int>("s_region", S_LEN);

	int* h_c_custkey = loadColumn<int>("c_custkey", C_LEN);
	int* h_c_nation  = loadColumn<int>("c_nation", C_LEN);
	int* h_c_region  = loadColumn<int>("c_region", C_LEN);

	int* d_d_datekey = loadToGPU<int>(h_d_datekey, D_LEN, g_allocator);
	int* d_d_year    = loadToGPU<int>(h_d_year, D_LEN, g_allocator);

	int* d_s_suppkey = loadToGPU<int>(h_s_suppkey, S_LEN, g_allocator);
	int* d_s_region  = loadToGPU<int>(h_s_region, S_LEN, g_allocator);
	int* d_s_nation  = loadToGPU<int>(h_s_nation, S_LEN, g_allocator);

	int* d_c_custkey = loadToGPU<int>(h_c_custkey, C_LEN, g_allocator);
	int* d_c_region  = loadToGPU<int>(h_c_region, C_LEN, g_allocator);
	int* d_c_nation  = loadToGPU<int>(h_c_nation, C_LEN, g_allocator);

	int* d_lo_orderdate_base   = loadToGPU<int32_t>(h_lo_orderdate_base, n_vec, g_allocator);
	int* d_lo_orderdate_bw     = loadToGPU<int32_t>(h_lo_orderdate_bw, n_vec, g_allocator);
	int* d_lo_orderdate_offset = loadToGPU<int32_t>(h_lo_orderdate_offset, n_vec, g_allocator);

	// Run
	runQuery<32, 8>(d_lo_orderdate,
	                d_lo_orderdate_bw,
	                d_lo_orderdate_base,
	                d_lo_orderdate_offset,
	                d_lo_custkey,
	                d_lo_suppkey,
	                d_lo_revenue,
	                LO_LEN,
	                d_d_datekey,
	                d_d_year,
	                D_LEN,
	                d_s_suppkey,
	                d_s_region,
	                d_s_nation,
	                S_LEN,
	                d_c_custkey,
	                d_c_region,
	                d_c_nation,
	                C_LEN,
	                g_allocator);
}