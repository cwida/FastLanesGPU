// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

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
__global__ void probe_v1(int* lo_orderdate,
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
	int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
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
	static __shared__ int unpacked[1024];

	int suppkey_tile_offset = blockIdx.x * query_mtd.ssb.lo_chosen_suppkey_bw * 32;
	unpack_device(lo_suppkey + suppkey_tile_offset, unpacked, query_mtd.ssb.lo_chosen_suppkey_bw);
	BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(unpacked, items, num_tile_items);
	BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, s_nation, selection_flags, ht_s, s_len, num_tile_items);

	int custkey_tile_offset = blockIdx.x * query_mtd.ssb.lo_chosen_custkey_bw * 32;
	unpack_device(lo_custkey + custkey_tile_offset, unpacked, query_mtd.ssb.lo_chosen_custkey_bw);
	BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(unpacked, items, num_tile_items);
	BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, c_nation, selection_flags, ht_c, c_len, num_tile_items);

	int orderdate_tile_offset = blockIdx.x * query_mtd.ssb.lo_orderdate_bw * 32;
	unpack_device(lo_orderdate + orderdate_tile_offset, unpacked, query_mtd.ssb.lo_orderdate_bw);
	BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(unpacked, items, num_tile_items);
	BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, year, selection_flags, ht_d, d_len, 0, num_tile_items);

	int revenue_tile_offset = blockIdx.x * query_mtd.ssb.lo_revenue_bw * 32;
	unpack_device(lo_revenue + revenue_tile_offset, unpacked, query_mtd.ssb.lo_revenue_bw);
	BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(unpacked, revenue, num_tile_items);

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
__global__ void probe_v2(int* lo_orderdate,
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
	int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
	// Load a segment of consecutive items that are blocked across threads
	int items[ITEMS_PER_THREAD];
	int selection_flags[ITEMS_PER_THREAD];
	// int c_nation[ITEMS_PER_THREAD];
	static __shared__ int shared_4_c_nation[1024];
	// int s_nation[ITEMS_PER_THREAD];
	static __shared__ int shared_3_s_nation[1024];
	// int year[ITEMS_PER_THREAD];
	static __shared__ int shared_2_year[1024];
	// int revenue[ITEMS_PER_THREAD];
	static __shared__ int shared_1_revenue[1024];

	int tile_offset    = blockIdx.x * TILE_SIZE;
	int num_tiles      = (lo_len + TILE_SIZE - 1) / TILE_SIZE;
	int num_tile_items = TILE_SIZE;

	if (blockIdx.x == num_tiles - 1) { num_tile_items = lo_len - tile_offset; }

	InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

	int suppkey_tile_offset = blockIdx.x * query_mtd.ssb.lo_chosen_suppkey_bw * 32;
	unpack_device(lo_suppkey + suppkey_tile_offset, shared_1_revenue, query_mtd.ssb.lo_chosen_suppkey_bw);
	BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(shared_1_revenue, items, num_tile_items);
	BlockProbeAndPHT_2_R_S<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, shared_3_s_nation, selection_flags, ht_s, s_len, num_tile_items);

	int custkey_tile_offset = blockIdx.x * query_mtd.ssb.lo_chosen_custkey_bw * 32;
	unpack_device(lo_custkey + custkey_tile_offset, shared_1_revenue, query_mtd.ssb.lo_chosen_custkey_bw);
	BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(shared_1_revenue, items, num_tile_items);
	BlockProbeAndPHT_2_R_S<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, shared_4_c_nation, selection_flags, ht_c, c_len, num_tile_items);

	int orderdate_tile_offset = blockIdx.x * query_mtd.ssb.lo_orderdate_bw * 32;
	unpack_device(lo_orderdate + orderdate_tile_offset, shared_1_revenue, query_mtd.ssb.lo_orderdate_bw);
	BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(shared_1_revenue, items, num_tile_items);
	BlockProbeAndPHT_2_R_S<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, shared_2_year, selection_flags, ht_d, d_len, int(0), num_tile_items);

	int revenue_tile_offset = blockIdx.x * query_mtd.ssb.lo_revenue_bw * 32;
	unpack_device(lo_revenue + revenue_tile_offset, shared_1_revenue, query_mtd.ssb.lo_revenue_bw);
	// BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(unpacked, revenue, num_tile_items);

#pragma unroll
	for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
		auto shared_idx = BLOCK_THREADS * ITEM + threadIdx.x;
		if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_tile_items) {
			if (selection_flags[ITEM]) {
				int hash = (shared_3_s_nation[shared_idx] * 25 * 7 + shared_4_c_nation[shared_idx] * 7 +
				            (shared_2_year[shared_idx] - 1992)) %
				           ((1998 - 1992 + 1) * 25 * 25);
				res[hash * 6]     = shared_2_year[shared_idx];
				res[hash * 6 + 1] = shared_4_c_nation[shared_idx];
				res[hash * 6 + 2] = shared_3_s_nation[shared_idx];
				/*atomicAdd(&res[hash * 6 + 4], revenue[ITEM]);*/
				atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]),
				          (long long)(shared_1_revenue[shared_idx]));
			}
		}
	}
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_v3(int* lo_orderdate,
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

	int orderdate_tile_offset = blockIdx.x * query_mtd.ssb.lo_orderdate_bw * 8;
	unpack_8_at_a_time::unpack_device(lo_orderdate + orderdate_tile_offset, items, query_mtd.ssb.lo_orderdate_bw);
	BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, year, selection_flags, ht_d, d_len, 0, num_tile_items);

	BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_revenue + tile_offset, revenue, num_tile_items);

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
__global__ void probe_v4(int* lo_orderdate,
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

	int orderdate_tile_offset = blockIdx.x * query_mtd.ssb.lo_orderdate_bw * 8;
	unpack_8_at_a_time::unpack_device(lo_orderdate + orderdate_tile_offset, items, query_mtd.ssb.lo_orderdate_bw);
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
              cub::CachingDeviceAllocator& g_allocator,
              int                          version) {

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
	if (version == 1) {
		probe_v1<BLOCK_THREADS, ITEMS_PER_THREAD><<<(lo_len + tile_items - 1) / tile_items, BLOCK_THREADS>>>(
		    lo_orderdate, lo_custkey, lo_suppkey, lo_revenue, lo_len, ht_s, s_len, ht_c, c_len, ht_d, d_val_len, res);
	} else if (version == 2) {
		if constexpr (ITEMS_PER_THREAD == 32) {
			probe_v2<BLOCK_THREADS, ITEMS_PER_THREAD>
			    <<<(lo_len + tile_items - 1) / tile_items, BLOCK_THREADS>>>(lo_orderdate,
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
		}
	} else if (version == 3) {
		if constexpr (ITEMS_PER_THREAD == 8) {
			probe_v3<BLOCK_THREADS, ITEMS_PER_THREAD>
			    <<<(lo_len + tile_items - 1) / tile_items, BLOCK_THREADS>>>(lo_orderdate,
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
		}
	} else if (version == 4) {
		if constexpr (ITEMS_PER_THREAD == 8) {
			probe_v4<BLOCK_THREADS, ITEMS_PER_THREAD>
			    <<<(lo_len + tile_items - 1) / tile_items, BLOCK_THREADS>>>(lo_orderdate,
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
		}
	} else {
		throw std::runtime_error("this version does not exist");
	}

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
	int version = 0;
	version     = std::stoi(argv[1]);

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

	auto* orderdate_in = const_cast<const int32_t*>(tmp);
	auto* custkey_in   = const_cast<int32_t*>(h_lo_custkey);
	auto* suppkey_in   = const_cast<int32_t*>(h_lo_suppkey);
	auto* revenue_in   = const_cast<int32_t*>(h_lo_revenue);

	auto* orderdate_out = const_cast<int32_t*>(h_enc_lo_orderdate);
	auto* custkey_out   = const_cast<int32_t*>(h_enc_lo_custkey);
	auto* suppkey_out   = const_cast<int32_t*>(h_enc_lo_suppkey);
	auto* revenue_out   = const_cast<int32_t*>(h_enc_lo_revenue);

	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(orderdate_in, orderdate_out, hard_coded.lo_orderdate_bw);
		orderdate_in  = orderdate_in + 1024;
		orderdate_out = orderdate_out + (hard_coded.lo_orderdate_bw * 32);

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

	if (version == 1 || version == 2) {
		d_lo_revenue = loadToGPU<int32_t>(h_enc_lo_revenue, hard_coded.n_tup_line_order, g_allocator);
	} else if (version == 3 || version == 4) {
		d_lo_revenue = loadToGPU<int32_t>(h_lo_revenue, hard_coded.n_tup_line_order, g_allocator);
	} else {
		throw std::runtime_error("this version does not exist");
	}

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

	// Run
	if (version == 1 || version == 2) {
		runQuery<32, 32>(d_lo_orderdate,
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
		                 g_allocator,
		                 version);
	} else if (version == 3 || version == 4) {
		runQuery<32, 8>(d_lo_orderdate,
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
		                g_allocator,
		                version);
	} else {
		throw std::runtime_error("this version does not exist");
	}
}