// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include "crystal.cuh"
#include "crystal_ssb_utils.h"
#include "cub/test/test_util.h"
#include "data/footer/ssb/ssb.hpp"
#include "fls_gen/unpack/unpack.cuh"
#include "gpu_utils.h"
#include "ssb_utils.h"
#include <cub/util_allocator.cuh>
#include <cuda.h>
#include <curand.h>
#include <fls_gen/pack/pack.hpp>
#include <fls_gen/unpack/hardcoded_16.cuh>
#include <iostream>
#include <stdio.h>


using namespace std;
using namespace fastlanes::gpu;
using namespace fastlanes;

using namespace std;

auto query_mtd = ssb::ssb_q13_1;

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe(int* lo_orderdate,
                      int* lo_partkey,
                      int* lo_custkey,
                      int* lo_suppkey,
                      int* lo_revenue,
                      int* lo_supplycost,
                      int  lo_len,
                      int* ht_p,
                      int  p_len,
                      int* ht_s,
                      int  s_len,
                      int* ht_c,
                      int  c_len,
                      int* ht_d,
                      int  d_len,
                      int* res) {
	// Load a segment of consecutive items that are blocked across threads
	int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

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
	BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, selection_flags, ht_s, s_len, num_tile_items);
	if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) { return; }

	int custkey_tile_offset = blockIdx.x * query_mtd.ssb.lo_chosen_custkey_bw * 8;
	unpack_8_at_a_time::unpack_device(lo_custkey + custkey_tile_offset, items, query_mtd.ssb.lo_chosen_custkey_bw);
	BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, c_nation, selection_flags, ht_c, c_len, num_tile_items);
	if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) { return; }

	int partkey_tile_offset = blockIdx.x * query_mtd.ssb.lo_partkey_bw * 8;
	unpack_8_at_a_time::unpack_device(lo_partkey + partkey_tile_offset, items, query_mtd.ssb.lo_partkey_bw);
	BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, selection_flags, ht_p, p_len, num_tile_items);
	if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) { return; }

	int orderdate_tile_offset = blockIdx.x * query_mtd.ssb.lo_orderdate_bw * 8;
	unpack_8_at_a_time::unpack_device(lo_orderdate + orderdate_tile_offset, items, query_mtd.ssb.lo_orderdate_bw);
	BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, year, selection_flags, ht_d, d_len, 0, num_tile_items);
	if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) { return; }

	BlockPredLoad<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    lo_revenue + tile_offset, revenue, num_tile_items, selection_flags);

	int supplycost_tile_offset = blockIdx.x * query_mtd.ssb.lo_chosen_supplycost_bw * 8;
	unpack_8_at_a_time::unpack_device(lo_supplycost + supplycost_tile_offset, items, query_mtd.ssb.lo_chosen_supplycost_bw);

#pragma unroll
	for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
		if (threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items) {
			if (selection_flags[ITEM]) {
				int hash          = (c_nation[ITEM] * 7 + (year[ITEM] - 1992)) % ((1998 - 1992 + 1) * 25);
				res[hash * 4]     = year[ITEM];
				res[hash * 4 + 1] = c_nation[ITEM];
				/*atomicAdd(&res[hash * 4 + 2], (1));*/
				/*atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 4 + 2]), (long long)(1));*/
				atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 4 + 2]),
				          (long long)(revenue[ITEM] - items[ITEM]));
			}
		}
	}
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_s(int* filter_col, int* dim_key, int num_tuples, int* hash_table, int num_slots) {
	int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

	int items[ITEMS_PER_THREAD];
	int selection_flags[ITEMS_PER_THREAD];

	int tile_offset    = blockIdx.x * TILE_SIZE;
	int num_tiles      = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
	int num_tile_items = TILE_SIZE;

	if (blockIdx.x == num_tiles - 1) { num_tile_items = num_tuples - tile_offset; }

	BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset, items, num_tile_items);
	BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1, selection_flags, num_tile_items);

	BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items, num_tile_items);
	BlockBuildSelectivePHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, selection_flags, hash_table, num_slots, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_p(int* filter_col, int* dim_key, int num_tuples, int* hash_table, int num_slots) {
	int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

	int items[ITEMS_PER_THREAD];
	int selection_flags[ITEMS_PER_THREAD];

	int tile_offset    = blockIdx.x * TILE_SIZE;
	int num_tiles      = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
	int num_tile_items = TILE_SIZE;

	if (blockIdx.x == num_tiles - 1) { num_tile_items = num_tuples - tile_offset; }

	BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset, items, num_tile_items);
	BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 0, selection_flags, num_tile_items);
	BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1, selection_flags, num_tile_items);

	BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items, num_tile_items);
	BlockBuildSelectivePHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, selection_flags, hash_table, num_slots, num_tile_items);
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
	BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1, selection_flags, num_tile_items);

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

	InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);
	BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items, num_tile_items);
	BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2, num_tile_items);
	BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, items2, selection_flags, hash_table, num_slots, val_min, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
float runQuery(int*                         lo_orderdate,
               int*                         lo_custkey,
               int*                         lo_partkey,
               int*                         lo_suppkey,
               int*                         lo_revenue,
               int*                         lo_supplycost,
               int                          lo_len,
               int*                         d_datekey,
               int*                         d_year,
               int                          d_len,
               int*                         p_partkey,
               int*                         p_mfgr,
               int                          p_len,
               int*                         s_suppkey,
               int*                         s_region,
               int                          s_len,
               int*                         c_custkey,
               int*                         c_region,
               int*                         c_nation,
               int                          c_len,
               cub::CachingDeviceAllocator& g_allocator) {
	SETUP_TIMING();

	float                                     time_query;
	chrono::high_resolution_clock::time_point st, finish;
	st = chrono::high_resolution_clock::now();

	cudaEventRecord(start, 0);

	int *ht_d, *ht_c, *ht_s, *ht_p;
	int  d_val_len = 19981230 - 19920101 + 1;
	CubDebugExit(g_allocator.DeviceAllocate((void**)&ht_d, 2 * d_val_len * sizeof(int)));
	CubDebugExit(g_allocator.DeviceAllocate((void**)&ht_s, 2 * s_len * sizeof(int)));
	CubDebugExit(g_allocator.DeviceAllocate((void**)&ht_c, 2 * c_len * sizeof(int)));
	CubDebugExit(g_allocator.DeviceAllocate((void**)&ht_p, 2 * p_len * sizeof(int)));

	CubDebugExit(cudaMemset(ht_d, 0, 2 * d_val_len * sizeof(int)));
	CubDebugExit(cudaMemset(ht_s, 0, 2 * s_len * sizeof(int)));
	CubDebugExit(cudaMemset(ht_c, 0, 2 * c_len * sizeof(int)));
	CubDebugExit(cudaMemset(ht_p, 0, 2 * p_len * sizeof(int)));

	int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
	build_hashtable_s<BLOCK_THREADS, ITEMS_PER_THREAD>
	    <<<(s_len + tile_items - 1) / tile_items, BLOCK_THREADS>>>(s_region, s_suppkey, s_len, ht_s, s_len);
	/*CHECK_ERROR();*/

	int* s_res = new int[s_len * 2];
	CubDebugExit(cudaMemcpy(s_res, ht_s, s_len * 2 * sizeof(int), cudaMemcpyDeviceToHost));

	build_hashtable_c<BLOCK_THREADS, ITEMS_PER_THREAD>
	    <<<(c_len + tile_items - 1) / tile_items, BLOCK_THREADS>>>(c_region, c_custkey, c_nation, c_len, ht_c, c_len);
	/*CHECK_ERROR();*/

	int* c_res = new int[c_len * 2];
	CubDebugExit(cudaMemcpy(c_res, ht_c, c_len * 2 * sizeof(int), cudaMemcpyDeviceToHost));

	build_hashtable_p<BLOCK_THREADS, ITEMS_PER_THREAD>
	    <<<(p_len + tile_items - 1) / tile_items, BLOCK_THREADS>>>(p_mfgr, p_partkey, p_len, ht_p, p_len);
	/*CHECK_ERROR();*/

	int* p_res = new int[p_len * 2];
	CubDebugExit(cudaMemcpy(p_res, ht_p, p_len * 2 * sizeof(int), cudaMemcpyDeviceToHost));

	int d_val_min = 19920101;
	build_hashtable_d<BLOCK_THREADS, ITEMS_PER_THREAD><<<(d_len + tile_items - 1) / tile_items, BLOCK_THREADS>>>(
	    d_datekey, d_year, d_len, ht_d, d_val_len, d_val_min);
	/*CHECK_ERROR();*/

#if 0
  int *h_ht_s = new int[s_len * 2];
  int *h_ht_c = new int[c_len * 2];
  int *h_ht_p = new int[p_len * 2];
  int *h_ht_d = new int[d_val_len * 2];

  int num_s = 0 , num_c = 0, num_d = 0, num_p = 0;

  CubDebugExit(cudaMemcpy(h_ht_s, ht_s, 2 * s_len * sizeof(int), cudaMemcpyDeviceToHost));
  for (int i=0; i<s_len; i++) if (h_ht_s[i*2] != 0) num_s += 1;

  cout << "Num Matched" << " " << num_s << " " << s_len << endl;

  CubDebugExit(cudaMemcpy(h_ht_d, ht_d, 2 * d_val_len * sizeof(int), cudaMemcpyDeviceToHost));
  for (int i=0; i<d_val_len; i++) if (h_ht_d[i*2] != 0) num_d += 1;

  cout << "Num Matched" << " " << num_d << " " << d_len << endl;

  CubDebugExit(cudaMemcpy(h_ht_c, ht_c, 2 * c_len * sizeof(int), cudaMemcpyDeviceToHost));
  for (int i=0; i<c_len; i++) if (h_ht_c[i*2] != 0) num_c += 1;

  cout << "Num Matched" << " " << num_c << " " << c_len << endl;

  CubDebugExit(cudaMemcpy(h_ht_p, ht_p, 2 * p_len * sizeof(int), cudaMemcpyDeviceToHost));
  for (int i=0; i<p_len; i++) if (h_ht_p[i*2] != 0) num_p += 1;

  cout << "Num Matched" << " " << num_p << " " << p_len << endl;
#endif

	int* res;
	int  res_size       = ((1998 - 1992 + 1) * 25);
	int  ht_entries     = 4; // int,int,long long
	int  res_array_size = res_size * ht_entries;
	CubDebugExit(g_allocator.DeviceAllocate((void**)&res, res_array_size * sizeof(int)));

	CubDebugExit(cudaMemset(res, 0, res_array_size * sizeof(int)));

	// Run
	probe<BLOCK_THREADS, ITEMS_PER_THREAD><<<(lo_len + tile_items - 1) / tile_items, BLOCK_THREADS>>>(lo_orderdate,
	                                                                                                  lo_partkey,
	                                                                                                  lo_custkey,
	                                                                                                  lo_suppkey,
	                                                                                                  lo_revenue,
	                                                                                                  lo_supplycost,
	                                                                                                  lo_len,
	                                                                                                  ht_p,
	                                                                                                  p_len,
	                                                                                                  ht_s,
	                                                                                                  s_len,
	                                                                                                  ht_c,
	                                                                                                  c_len,
	                                                                                                  ht_d,
	                                                                                                  d_val_len,
	                                                                                                  res);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_query, start, stop);

	int* h_res = new int[res_array_size];
	CubDebugExit(cudaMemcpy(h_res, res, res_array_size * sizeof(int), cudaMemcpyDeviceToHost));
	finish                             = chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = finish - st;

	cout << "Result:" << endl;
	int res_count = 0;
	for (int i = 0; i < res_size; i++) {
		if (h_res[4 * i] != 0) {
			cout << h_res[4 * i] << " " << h_res[4 * i + 1] << " "
			     << reinterpret_cast<unsigned long long*>(&h_res[4 * i + 2])[0] << endl;
			res_count += 1;
		}
	}

	cout << "Res Count: " << res_count << endl;
	cout << "Time Taken Total: " << diff.count() * 1000 << endl;

	delete[] h_res;

	return time_query;
}

/**
 * Main
 */
int main() {
	int num_trials = 3;

	auto query_mtd = ssb::ssb_q11_10;

	auto hard_coded      = query_mtd.ssb;
	int* h_lo_orderdate  = loadColumn<int>("lo_orderdate", LO_LEN);
	int* h_lo_suppkey    = loadColumn<int>("lo_suppkey", LO_LEN);
	int* h_lo_custkey    = loadColumn<int>("lo_custkey", LO_LEN);
	int* h_lo_partkey    = loadColumn<int>("lo_partkey", LO_LEN);
	int* h_lo_revenue    = loadColumn<int>("lo_revenue", LO_LEN);
	int* h_lo_supplycost = loadColumn<int>("lo_supplycost", LO_LEN);

	auto n_vec = hard_coded.n_vec;

	int* tmp = new int[n_vec * 1024];
	for (size_t i {0}; i < LO_LEN; ++i) {
		tmp[i] = h_lo_orderdate[i] - hard_coded.lo_orderdate_min;
	}

	const int* h_enc_lo_orderdate  = new int[n_vec * 1024];
	const int* h_enc_lo_custkey    = new int[n_vec * 1024];
	const int* h_enc_lo_suppkey    = new int[n_vec * 1024];
	const int* h_enc_lo_revenue    = new int[n_vec * 1024];
	const int* h_enc_lo_partkey    = new int[n_vec * 1024];
	const int* h_enc_lo_supplycost = new int[n_vec * 1024];

	auto* orderdate_in  = const_cast<const int32_t*>(tmp);
	auto* custkey_in    = const_cast<int32_t*>(h_lo_custkey);
	auto* suppkey_in    = const_cast<int32_t*>(h_lo_suppkey);
	auto* revenue_in    = const_cast<int32_t*>(h_lo_revenue);
	auto* partkey_in    = const_cast<int32_t*>(h_lo_partkey);
	auto* supplycost_in = const_cast<int32_t*>(h_lo_supplycost);

	auto* orderdate_out  = const_cast<int32_t*>(h_enc_lo_orderdate);
	auto* custkey_out    = const_cast<int32_t*>(h_enc_lo_custkey);
	auto* suppkey_out    = const_cast<int32_t*>(h_enc_lo_suppkey);
	auto* revenue_out    = const_cast<int32_t*>(h_enc_lo_revenue);
	auto* partkey_out    = const_cast<int32_t*>(h_enc_lo_partkey);
	auto* supplycost_out = const_cast<int32_t*>(h_enc_lo_supplycost);

	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(orderdate_in, orderdate_out, hard_coded.lo_orderdate_bw);
		orderdate_in  = orderdate_in + 1024;
		orderdate_out = orderdate_out + (hard_coded.lo_orderdate_bw * 32);

		generated::pack::fallback::scalar::pack(partkey_in, partkey_out, hard_coded.lo_partkey_bw);
		partkey_in  = partkey_in + 1024;
		partkey_out = partkey_out + (hard_coded.lo_partkey_bw * 32);

		generated::pack::fallback::scalar::pack(supplycost_in, supplycost_out, hard_coded.lo_chosen_supplycost_bw);
		supplycost_in  = supplycost_in + 1024;
		supplycost_out = supplycost_out + (hard_coded.lo_chosen_supplycost_bw * 32);

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

	int* d_lo_orderdate  = loadToGPU<int32_t>(h_enc_lo_orderdate, hard_coded.n_tup_line_order, g_allocator);
	int* d_lo_custkey    = loadToGPU<int32_t>(h_enc_lo_custkey, hard_coded.n_tup_line_order, g_allocator);
	int* d_lo_suppkey    = loadToGPU<int32_t>(h_enc_lo_suppkey, hard_coded.n_tup_line_order, g_allocator);
	int* d_lo_revenue    = loadToGPU<int32_t>(h_lo_revenue, hard_coded.n_tup_line_order, g_allocator);
	int* d_lo_partkey    = loadToGPU<int32_t>(h_enc_lo_partkey, hard_coded.n_tup_line_order, g_allocator);
	int* d_lo_supplycost = loadToGPU<int32_t>(h_enc_lo_supplycost, hard_coded.n_tup_line_order, g_allocator);

	int* h_d_datekey      = loadColumn<int>("d_datekey", D_LEN);
	int* h_d_year         = loadColumn<int>("d_year", D_LEN);
	int* h_d_yearmonthnum = loadColumn<int>("d_yearmonthnum", D_LEN);

	int* h_s_suppkey = loadColumn<int>("s_suppkey", S_LEN);
	int* h_s_region  = loadColumn<int>("s_region", S_LEN);

	int* h_p_partkey = loadColumn<int>("p_partkey", P_LEN);
	int* h_p_mfgr    = loadColumn<int>("p_mfgr", P_LEN);

	int* h_c_custkey = loadColumn<int>("c_custkey", C_LEN);
	int* h_c_region  = loadColumn<int>("c_region", C_LEN);
	int* h_c_nation  = loadColumn<int>("c_nation", C_LEN);

	cout << "** LOADED DATA **" << endl;

	int* d_d_datekey = loadToGPU<int>(h_d_datekey, D_LEN, g_allocator);
	int* d_d_year    = loadToGPU<int>(h_d_year, D_LEN, g_allocator);

	int* d_p_partkey = loadToGPU<int>(h_p_partkey, P_LEN, g_allocator);
	int* d_p_mfgr    = loadToGPU<int>(h_p_mfgr, P_LEN, g_allocator);

	int* d_s_suppkey = loadToGPU<int>(h_s_suppkey, S_LEN, g_allocator);
	int* d_s_region  = loadToGPU<int>(h_s_region, S_LEN, g_allocator);

	int* d_c_custkey = loadToGPU<int>(h_c_custkey, C_LEN, g_allocator);
	int* d_c_region  = loadToGPU<int>(h_c_region, C_LEN, g_allocator);
	int* d_c_nation  = loadToGPU<int>(h_c_nation, C_LEN, g_allocator);

	cout << "** LOADED DATA TO GPU **" << endl;

	for (int t = 0; t < num_trials; t++) {
		float time_query;

		time_query = runQuery<32, 8>(d_lo_orderdate,
		                              d_lo_custkey,
		                              d_lo_partkey,
		                              d_lo_suppkey,
		                              d_lo_revenue,
		                              d_lo_supplycost,
		                              LO_LEN,
		                              d_d_datekey,
		                              d_d_year,
		                              D_LEN,
		                              d_p_partkey,
		                              d_p_mfgr,
		                              P_LEN,
		                              d_s_suppkey,
		                              d_s_region,
		                              S_LEN,
		                              d_c_custkey,
		                              d_c_region,
		                              d_c_nation,
		                              C_LEN,
		                              g_allocator);
		cout << "{"
		     << "\"query\":41"
		     << ",\"time_query\":" << time_query << "}" << endl;
	}

	return 0;
}
