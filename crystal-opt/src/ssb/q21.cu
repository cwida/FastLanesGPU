// MIT License

// Copyright (c) 2023 Jiashen Cao

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <curand.h>
#include <iostream>
#include <stdio.h>

#include "cub/test/test_util.h"
#include <cub/util_allocator.cuh>
#include <cuda.h>

#include "crystal.cuh"

#include "gpu_utils.h"
#include "ssb_utils.h"

using namespace std;

/**
 * Globals, constants and typedefs
 */
bool g_verbose = false; // Whether to display input/output to console
cub::CachingDeviceAllocator
    g_allocator(true); // Caching allocator for device memory

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe(int *lo_orderdate, int *lo_partkey, int *lo_suppkey,
                      int *lo_revenue, int lo_len, int *ht_s, int s_len,
                      int *ht_p, int p_len, int *ht_d, int d_len, int *res) {
  // Load a tile striped across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int brand[ITEMS_PER_THREAD];
  int year[ITEMS_PER_THREAD];
  int revenue[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (lo_len + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = lo_len - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_suppkey + tile_offset,
                                                  items, num_tile_items);
  BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, selection_flags, ht_s, s_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) { return; }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_partkey + tile_offset, items, num_tile_items, selection_flags);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, brand, selection_flags, ht_p, p_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) { return; }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_orderdate + tile_offset, items, num_tile_items, selection_flags);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, year, selection_flags, ht_d, d_len, 19920101, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) { return; }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_revenue + tile_offset, revenue, num_tile_items, selection_flags);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) { return; }

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_tile_items) {
      if (selection_flags[ITEM]) {
        int hash = (brand[ITEM] * 7 + (year[ITEM] - 1992)) %
                   ((1998 - 1992 + 1) * (5 * 5 * 40));
        res[hash * 4] = year[ITEM];
        res[hash * 4 + 1] = brand[ITEM];
        atomicAdd(reinterpret_cast<unsigned long long *>(&res[hash * 4 + 2]),
                  (long long)(revenue[ITEM]));
      }
    }
  }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_s(int *filter_col, int *dim_key, int num_tuples,
                                  int *hash_table, int num_slots) {
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset,
                                                  items, num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1, selection_flags,
                                                    num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items,
                                                  num_tile_items);
  BlockBuildSelectivePHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, selection_flags, hash_table, num_slots, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_p(int *filter_col, int *dim_key, int *dim_val,
                                  int num_tuples, int *hash_table,
                                  int num_slots) {
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset,
                                                  items, num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1, selection_flags,
                                                    num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items,
                                                  num_tile_items);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2,
                                                  num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, items2, selection_flags, hash_table, num_slots, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_d(int *dim_key, int *dim_val, int num_tuples,
                                  int *hash_table, int num_slots, int val_min) {
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items,
                                                  num_tile_items);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2,
                                                  num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, items2, selection_flags, hash_table, num_slots, val_min,
      num_tile_items);
}

float runQuery(int *lo_orderdate, int *lo_partkey, int *lo_suppkey,
               int *lo_revenue, int lo_len, int *p_partkey, int *p_brand1,
               int *p_category, int p_len, int *d_datekey, int *d_year,
               int d_len, int *s_suppkey, int *s_region, int s_len,
               cub::CachingDeviceAllocator &g_allocator) {
  SETUP_TIMING();

  float time_query;
  chrono::high_resolution_clock::time_point st, finish;
  st = chrono::high_resolution_clock::now();

  cudaEventRecord(start, 0);

  int *ht_d, *ht_p, *ht_s;
  int d_val_len = 19981230 - 19920101 + 1;
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_d, 2 * d_val_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_p, 2 * p_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_s, 2 * s_len * sizeof(int)));

  CubDebugExit(cudaMemset(ht_d, 0, 2 * d_val_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_p, 0, 2 * p_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_s, 0, 2 * s_len * sizeof(int)));

  int tile_items = 128 * 4;

  build_hashtable_s<128, 4><<<(s_len + tile_items - 1) / tile_items, 128>>>(
      s_region, s_suppkey, s_len, ht_s, s_len);
  /*CHECK_ERROR();*/

  build_hashtable_p<128, 4><<<(p_len + tile_items - 1) / tile_items, 128>>>(
      p_category, p_partkey, p_brand1, p_len, ht_p, p_len);
  /*CHECK_ERROR();*/

  int d_val_min = 19920101;
  build_hashtable_d<128, 4><<<(d_len + tile_items - 1) / tile_items, 128>>>(
      d_datekey, d_year, d_len, ht_d, d_val_len, d_val_min);
  /*CHECK_ERROR();*/

  int *res;
  int res_size = ((1998 - 1992 + 1) * (5 * 5 * 40));
  int res_array_size = res_size * 4;
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&res, res_array_size * sizeof(int)));

  CubDebugExit(cudaMemset(res, 0, res_array_size * sizeof(int)));

  probe<128, 4><<<(lo_len + tile_items - 1) / tile_items, 128>>>(
      lo_orderdate, lo_partkey, lo_suppkey, lo_revenue, lo_len, ht_s, s_len,
      ht_p, p_len, ht_d, d_val_len, res);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_query, start, stop);

  int *h_res = new int[res_array_size];
  CubDebugExit(cudaMemcpy(h_res, res, res_array_size * sizeof(int),
                          cudaMemcpyDeviceToHost));
  finish = chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = finish - st;

  int res_count = 0;
  for (int i = 0; i < res_size; i++) {
    if (h_res[4 * i] != 0) {
      cout << h_res[4 * i] << " " << h_res[4 * i + 1] << " "
           << reinterpret_cast<unsigned long long *>(&h_res[4 * i + 2])[0]
           << endl;
      res_count += 1;
    }
  }

  cout << "Res Count: " << res_count << endl;
  cout << "Time Taken Total: " << diff.count() * 1000 << endl;

  delete[] h_res;

  CLEANUP(res);
  CLEANUP(ht_d);
  CLEANUP(ht_p);
  CLEANUP(ht_s);

  return time_query;
}

/**
 * Main
 */
int main(int argc, char **argv) {
  int num_trials = 1;

  // Initialize command line
  CommandLineArgs args(argc, argv);
  args.GetCmdLineArgument("t", num_trials);

  // Print usage
  if (args.CheckCmdLineFlag("help")) {
    printf("%s "
           "[--t=<num trials>] "
           "[--v] "
           "\n",
           argv[0]);
    exit(0);
  }

  // Initialize device
  CubDebugExit(args.DeviceInit());

  int *h_lo_orderdate = loadColumn<int>("lo_orderdate", LO_LEN);
  int *h_lo_partkey = loadColumn<int>("lo_partkey", LO_LEN);
  int *h_lo_suppkey = loadColumn<int>("lo_suppkey", LO_LEN);
  int *h_lo_revenue = loadColumn<int>("lo_revenue", LO_LEN);
  int *h_p_partkey = loadColumn<int>("p_partkey", P_LEN);
  int *h_p_brand1 = loadColumn<int>("p_brand1", P_LEN);
  int *h_p_category = loadColumn<int>("p_category", P_LEN);

  int *h_d_datekey = loadColumn<int>("d_datekey", D_LEN);
  int *h_d_year = loadColumn<int>("d_year", D_LEN);

  int *h_s_suppkey = loadColumn<int>("s_suppkey", S_LEN);
  int *h_s_region = loadColumn<int>("s_region", S_LEN);

  int *d_lo_orderdate = loadToGPU<int>(h_lo_orderdate, LO_LEN, g_allocator);
  int *d_lo_partkey = loadToGPU<int>(h_lo_partkey, LO_LEN, g_allocator);
  int *d_lo_suppkey = loadToGPU<int>(h_lo_suppkey, LO_LEN, g_allocator);
  int *d_lo_revenue = loadToGPU<int>(h_lo_revenue, LO_LEN, g_allocator);

  int *d_d_datekey = loadToGPU<int>(h_d_datekey, D_LEN, g_allocator);
  int *d_d_year = loadToGPU<int>(h_d_year, D_LEN, g_allocator);

  int *d_p_partkey = loadToGPU<int>(h_p_partkey, P_LEN, g_allocator);
  int *d_p_brand1 = loadToGPU<int>(h_p_brand1, P_LEN, g_allocator);
  int *d_p_category = loadToGPU<int>(h_p_category, P_LEN, g_allocator);

  int *d_s_suppkey = loadToGPU<int>(h_s_suppkey, S_LEN, g_allocator);
  int *d_s_region = loadToGPU<int>(h_s_region, S_LEN, g_allocator);

  for (int t = 0; t < num_trials; t++) {
    float time_query;
    time_query = runQuery(d_lo_orderdate, d_lo_partkey, d_lo_suppkey,
                          d_lo_revenue, LO_LEN, d_p_partkey, d_p_brand1,
                          d_p_category, P_LEN, d_d_datekey, d_d_year, D_LEN,
                          d_s_suppkey, d_s_region, S_LEN, g_allocator);
    cout << "{"
         << "\"query\":21"
         << ",\"time_query\":" << time_query << "}" << endl;
  }

  return 0;
}
