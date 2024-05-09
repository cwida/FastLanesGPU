// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <algorithm>
#include <curand.h>
#include <iostream>
#include <random>
#include <stdio.h>
#include <vector>

#include "cub/test/test_util.h"
#include <cub/util_allocator.cuh>
#include <cuda.h>

#include "crystal/crystal.cuh"

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
__global__ void q11_kernel(int *lo_orderdate, int *lo_discount,
                           int *lo_quantity, int *lo_extendedprice,
                           int lo_num_entries, unsigned long long *revenue)
{
  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];

  long long sum = 0;

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (lo_num_entries + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = lo_num_entries - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_orderdate + tile_offset,
                                                  items, num_tile_items);
  BlockPredGT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 19930000, selection_flags, num_tile_items);
  BlockPredAndLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 19940000, selection_flags, num_tile_items);

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_quantity + tile_offset, items, num_tile_items, selection_flags);
  BlockPredAndLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 25, selection_flags, num_tile_items);

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_discount + tile_offset, items, num_tile_items, selection_flags);
  BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 1, selection_flags, num_tile_items);
  BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 3, selection_flags, num_tile_items);

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_extendedprice + tile_offset, items2, num_tile_items, selection_flags);

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
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

  if (threadIdx.x == 0)
  {
    atomicAdd(revenue, aggregate);
  }
}

void run_q11(int *lo_orderdate, int *lo_discount, int *lo_quantity,
             int *lo_extendedprice, int lo_num_entries,
             cub::CachingDeviceAllocator &g_allocator)
{
  // Setup
  unsigned long long *d_sum = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void **)&d_sum, sizeof(long long)));
  cudaMemset(d_sum, 0, sizeof(long long));

  // Run
  int tile_items = 128 * 4;
  int num_blocks = (lo_num_entries + tile_items - 1) / tile_items;
  q11_kernel<128, 4><<<num_blocks, 128>>>(lo_orderdate, lo_discount,
                                          lo_quantity, lo_extendedprice,
                                          lo_num_entries, d_sum);

  // Finalize results
  unsigned long long revenue;
  CubDebugExit(
      cudaMemcpy(&revenue, d_sum, sizeof(long long), cudaMemcpyDeviceToHost));
  CLEANUP(d_sum);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q12_kernel(int *lo_orderdate, int *lo_discount,
                           int *lo_quantity, int *lo_extendedprice,
                           int lo_num_entries, unsigned long long *revenue)
{
  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];

  long long sum = 0;

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (lo_num_entries + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = lo_num_entries - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_orderdate + tile_offset,
                                                  items, num_tile_items);
  BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 19940101, selection_flags, num_tile_items);
  BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 19940131, selection_flags, num_tile_items);

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_quantity + tile_offset, items, num_tile_items, selection_flags);
  BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 26, selection_flags, num_tile_items);
  BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 35, selection_flags, num_tile_items);

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_discount + tile_offset, items, num_tile_items, selection_flags);
  BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 4, selection_flags, num_tile_items);
  BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 6, selection_flags, num_tile_items);

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_extendedprice + tile_offset, items2, num_tile_items, selection_flags);

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items)
      if (selection_flags[ITEM])
        sum += items[ITEM] * items2[ITEM];
  }

  __syncthreads();

  static __shared__ long long buffer[32];
  unsigned long long aggregate =
      BlockSum<long long, BLOCK_THREADS, ITEMS_PER_THREAD>(sum,
                                                           (long long *)buffer);
  __syncthreads();

  if (threadIdx.x == 0)
  {
    atomicAdd(revenue, aggregate);
  }
}

void run_q12(int *lo_orderdate, int *lo_discount, int *lo_quantity,
             int *lo_extendedprice, int lo_num_entries,
             cub::CachingDeviceAllocator &g_allocator)
{
  // Setup
  unsigned long long *d_sum = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void **)&d_sum, sizeof(long long)));
  cudaMemset(d_sum, 0, sizeof(long long));

  // Run
  int tile_items = 128 * 4;
  q12_kernel<128, 4><<<(lo_num_entries + tile_items - 1) / tile_items, 128>>>(
      lo_orderdate, lo_discount, lo_quantity, lo_extendedprice, lo_num_entries,
      d_sum);

  // Finalize results
  unsigned long long revenue;
  CubDebugExit(
      cudaMemcpy(&revenue, d_sum, sizeof(long long), cudaMemcpyDeviceToHost));
  CLEANUP(d_sum);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q13_kernel(int *lo_orderdate, int *lo_discount,
                           int *lo_quantity, int *lo_extendedprice,
                           int lo_num_entries, unsigned long long *revenue)
{
  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];

  long long sum = 0;

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (lo_num_entries + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = lo_num_entries - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_orderdate + tile_offset,
                                                  items, num_tile_items);
  BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 19940204, selection_flags, num_tile_items);
  BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 19940210, selection_flags, num_tile_items);

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_quantity + tile_offset, items, num_tile_items, selection_flags);
  BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 26, selection_flags, num_tile_items);
  BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 35, selection_flags, num_tile_items);

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_discount + tile_offset, items, num_tile_items, selection_flags);
  BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 5, selection_flags, num_tile_items);
  BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 7, selection_flags, num_tile_items);

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_extendedprice + tile_offset, items2, num_tile_items, selection_flags);

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
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

  if (threadIdx.x == 0)
  {
    atomicAdd(revenue, aggregate);
  }
}

void run_q13(int *lo_orderdate, int *lo_discount, int *lo_quantity,
             int *lo_extendedprice, int lo_num_entries,
             cub::CachingDeviceAllocator &g_allocator)
{
  // Setup
  unsigned long long *d_sum = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void **)&d_sum, sizeof(long long)));
  cudaMemset(d_sum, 0, sizeof(long long));

  // Run
  int tile_items = 128 * 4;
  q13_kernel<128, 4><<<(lo_num_entries + tile_items - 1) / tile_items, 128>>>(
      lo_orderdate, lo_discount, lo_quantity, lo_extendedprice, lo_num_entries,
      d_sum);

  // Finalize results
  unsigned long long revenue;
  CubDebugExit(
      cudaMemcpy(&revenue, d_sum, sizeof(long long), cudaMemcpyDeviceToHost));
  CLEANUP(d_sum);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q21_kernel4(int *lo_orderdate, int *lo_partkey, int *lo_suppkey,
                            int *lo_revenue, int lo_len, int *ht_s, int s_len,
                            int *ht_p, int p_len, int *ht_d, int d_len,
                            int *res)
{
  // Load a tile striped across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int brand[ITEMS_PER_THREAD];
  int year[ITEMS_PER_THREAD];
  int revenue[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (lo_len + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = lo_len - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_suppkey + tile_offset,
                                                  items, num_tile_items);
  BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, selection_flags, ht_s, s_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_partkey + tile_offset, items, num_tile_items, selection_flags);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, brand, selection_flags, ht_p, p_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_orderdate + tile_offset, items, num_tile_items, selection_flags);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, year, selection_flags, ht_d, d_len, 19920101, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_revenue + tile_offset, revenue, num_tile_items, selection_flags);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_tile_items)
    {
      if (selection_flags[ITEM])
      {
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
__global__ void q21_kernel1(int *filter_col, int *dim_key, int num_tuples,
                            int *hash_table, int num_slots)
{
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
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
__global__ void q21_kernel2(int *filter_col, int *dim_key, int *dim_val,
                            int num_tuples, int *hash_table, int num_slots)
{
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
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
__global__ void q21_kernel3(int *dim_key, int *dim_val, int num_tuples,
                            int *hash_table, int num_slots, int val_min)
{
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
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

void run_q21(int *lo_orderdate, int *lo_partkey, int *lo_suppkey,
             int *lo_revenue, int lo_len, int *p_partkey, int *p_brand1,
             int *p_category, int p_len, int *d_datekey, int *d_year, int d_len,
             int *s_suppkey, int *s_region, int s_len,
             cub::CachingDeviceAllocator &g_allocator)
{
  // Setup
  int *ht_d, *ht_p, *ht_s;
  int d_val_len = 19981230 - 19920101 + 1;
  int *res;
  int res_size = ((1998 - 1992 + 1) * (5 * 5 * 40));
  int res_array_size = res_size * 4;
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_d, 2 * d_val_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_p, 2 * p_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_s, 2 * s_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_d, 0, 2 * d_val_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_p, 0, 2 * p_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_s, 0, 2 * s_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&res, res_array_size * sizeof(int)));
  CubDebugExit(cudaMemset(res, 0, res_array_size * sizeof(int)));

  // Run queries
  int tile_items = 128 * 4;
  int d_val_min = 19920101;
  q21_kernel1<128, 4><<<(s_len + tile_items - 1) / tile_items, 128>>>(
      s_region, s_suppkey, s_len, ht_s, s_len);
  q21_kernel2<128, 4><<<(p_len + tile_items - 1) / tile_items, 128>>>(
      p_category, p_partkey, p_brand1, p_len, ht_p, p_len);
  q21_kernel3<128, 4><<<(d_len + tile_items - 1) / tile_items, 128>>>(
      d_datekey, d_year, d_len, ht_d, d_val_len, d_val_min);
  q21_kernel4<128, 4><<<(lo_len + tile_items - 1) / tile_items, 128>>>(
      lo_orderdate, lo_partkey, lo_suppkey, lo_revenue, lo_len, ht_s, s_len,
      ht_p, p_len, ht_d, d_val_len, res);

  // Finalize results
  int *h_res = new int[res_array_size];
  CubDebugExit(cudaMemcpy(h_res, res, res_array_size * sizeof(int),
                          cudaMemcpyDeviceToHost));
  delete[] h_res;
  CLEANUP(res);
  CLEANUP(ht_d);
  CLEANUP(ht_p);
  CLEANUP(ht_s);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q22_kernel4(int *lo_orderdate, int *lo_partkey, int *lo_suppkey,
                            int *lo_revenue, int lo_len, int *ht_s, int s_len,
                            int *ht_p, int p_len, int *ht_d, int d_len,
                            int *res)
{
  // Load a tile striped across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int brand[ITEMS_PER_THREAD];
  int year[ITEMS_PER_THREAD];
  int revenue[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (lo_len + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = lo_len - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_suppkey + tile_offset,
                                                  items, num_tile_items);
  BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, selection_flags, ht_s, s_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_partkey + tile_offset, items, num_tile_items, selection_flags);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, brand, selection_flags, ht_p, p_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_orderdate + tile_offset, items, num_tile_items, selection_flags);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, year, selection_flags, ht_d, d_len, 19920101, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_revenue + tile_offset, revenue, num_tile_items, selection_flags);

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_tile_items)
    {
      if (selection_flags[ITEM])
      {
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
__global__ void q22_kernel1(int *filter_col, int *dim_key, int num_tuples,
                            int *hash_table, int num_slots)
{
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset,
                                                  items, num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 2, selection_flags,
                                                    num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items,
                                                  num_tile_items);
  BlockBuildSelectivePHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, selection_flags, hash_table, num_slots, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q22_kernel2(int *dim_key, int *dim_val, int num_tuples,
                            int *hash_table, int num_slots)
{
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items,
                                                  num_tile_items);
  BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 260, selection_flags, num_tile_items);
  BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 267, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items2,
                                                  num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items2, items, selection_flags, hash_table, num_slots, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q22_kernel3(int *dim_key, int *dim_val, int num_tuples,
                            int *hash_table, int num_slots, int val_min)
{
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
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

void run_q22(int *lo_orderdate, int *lo_partkey, int *lo_suppkey,
             int *lo_revenue, int lo_len, int *p_partkey, int *p_brand1,
             int p_len, int *d_datekey, int *d_year, int d_len, int *s_suppkey,
             int *s_region, int s_len,
             cub::CachingDeviceAllocator &g_allocator)
{
  int *ht_d, *ht_p, *ht_s;
  int d_val_len = 19981230 - 19920101 + 1;
  int *res;
  int res_size = ((1998 - 1992 + 1) * 1000);
  int res_array_size = res_size * 4;
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&res, res_array_size * sizeof(int)));
  CubDebugExit(cudaMemset(res, 0, res_array_size * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_d, 2 * d_val_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_p, 2 * p_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_s, 2 * s_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_d, 0, 2 * d_val_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_p, 0, 2 * p_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_s, 0, 2 * s_len * sizeof(int)));

  // Run queries
  int tile_items = 128 * 4;
  int d_val_min = 19920101;
  q22_kernel1<128, 4><<<(s_len + tile_items - 1) / tile_items, 128>>>(
      s_region, s_suppkey, s_len, ht_s, s_len);
  q22_kernel2<128, 4><<<(p_len + tile_items - 1) / tile_items, 128>>>(
      p_partkey, p_brand1, p_len, ht_p, p_len);
  q22_kernel3<128, 4><<<(d_len + tile_items - 1) / tile_items, 128>>>(
      d_datekey, d_year, d_len, ht_d, d_val_len, d_val_min);
  q22_kernel4<128, 4><<<(lo_len + tile_items - 1) / tile_items, 128>>>(
      lo_orderdate, lo_partkey, lo_suppkey, lo_revenue, lo_len, ht_s, s_len,
      ht_p, p_len, ht_d, d_val_len, res);

  // Finalize results
  int *h_res = new int[res_array_size];
  CubDebugExit(cudaMemcpy(h_res, res, res_array_size * sizeof(int),
                          cudaMemcpyDeviceToHost));
  delete[] h_res;
  CLEANUP(res);
  CLEANUP(ht_d);
  CLEANUP(ht_p);
  CLEANUP(ht_s);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q23_kernel4(int *lo_orderdate, int *lo_partkey, int *lo_suppkey,
                            int *lo_revenue, int lo_len, int *ht_s, int s_len,
                            int *ht_p, int p_len, int *ht_d, int d_len,
                            int *res)
{
  // Load a tile striped across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int brand[ITEMS_PER_THREAD];
  int year[ITEMS_PER_THREAD];
  int revenue[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (lo_len + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = lo_len - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_suppkey + tile_offset,
                                                  items, num_tile_items);
  BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, selection_flags, ht_s, s_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_partkey + tile_offset, items, num_tile_items, selection_flags);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, brand, selection_flags, ht_p, p_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_orderdate + tile_offset, items, num_tile_items, selection_flags);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, year, selection_flags, ht_d, d_len, 19920101, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_revenue + tile_offset, revenue, num_tile_items, selection_flags);

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_tile_items)
    {
      if (selection_flags[ITEM])
      {
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
__global__ void q23_kernel1(int *filter_col, int *dim_key, int num_tuples,
                            int *hash_table, int num_slots)
{
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset,
                                                  items, num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 3, selection_flags,
                                                    num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items,
                                                  num_tile_items);
  BlockBuildSelectivePHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, selection_flags, hash_table, num_slots, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q23_kernel2(int *dim_key, int *dim_val, int num_tuples,
                            int *hash_table, int num_slots)
{
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items,
                                                  num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 260, selection_flags,
                                                    num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items2,
                                                  num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items2, items, selection_flags, hash_table, num_slots, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q23_kernel3(int *dim_key, int *dim_val, int num_tuples,
                            int *hash_table, int num_slots, int val_min)
{
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
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

void run_q23(int *lo_orderdate, int *lo_partkey, int *lo_suppkey,
             int *lo_revenue, int lo_len, int *p_partkey, int *p_brand1,
             int p_len, int *d_datekey, int *d_year, int d_len, int *s_suppkey,
             int *s_region, int s_len,
             cub::CachingDeviceAllocator &g_allocator)
{
  int *ht_d, *ht_p, *ht_s;
  int d_val_len = 19981230 - 19920101 + 1;
  int *res;
  int res_size = ((1998 - 1992 + 1) * 1000);
  int res_array_size = res_size * 4;
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&res, res_array_size * sizeof(int)));
  CubDebugExit(cudaMemset(res, 0, res_array_size * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_d, 2 * d_val_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_p, 2 * p_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_s, 2 * s_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_d, 0, 2 * d_val_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_p, 0, 2 * p_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_s, 0, 2 * s_len * sizeof(int)));

  // Run queries
  int tile_items = 128 * 4;
  int d_val_min = 19920101;
  q23_kernel1<128, 4><<<(s_len + tile_items - 1) / tile_items, 128>>>(
      s_region, s_suppkey, s_len, ht_s, s_len);
  q23_kernel2<128, 4><<<(p_len + tile_items - 1) / tile_items, 128>>>(
      p_partkey, p_brand1, p_len, ht_p, p_len);
  q23_kernel3<128, 4><<<(d_len + tile_items - 1) / tile_items, 128>>>(
      d_datekey, d_year, d_len, ht_d, d_val_len, d_val_min);
  q23_kernel4<128, 4><<<(lo_len + tile_items - 1) / tile_items, 128>>>(
      lo_orderdate, lo_partkey, lo_suppkey, lo_revenue, lo_len, ht_s, s_len,
      ht_p, p_len, ht_d, d_val_len, res);

  // Finalize results
  int *h_res = new int[res_array_size];
  CubDebugExit(cudaMemcpy(h_res, res, res_array_size * sizeof(int),
                          cudaMemcpyDeviceToHost));
  delete[] h_res;
  CLEANUP(res);
  CLEANUP(ht_d);
  CLEANUP(ht_p);
  CLEANUP(ht_s);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q31_kernel4(int *lo_orderdate, int *lo_custkey, int *lo_suppkey,
                            int *lo_revenue, int lo_len, int *ht_s, int s_len,
                            int *ht_c, int c_len, int *ht_d, int d_len,
                            int *res)
{
  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int c_nation[ITEMS_PER_THREAD];
  int s_nation[ITEMS_PER_THREAD];
  int year[ITEMS_PER_THREAD];
  int revenue[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (lo_len + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = lo_len - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_suppkey + tile_offset,
                                                  items, num_tile_items);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, s_nation, selection_flags, ht_s, s_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_custkey + tile_offset, items, num_tile_items, selection_flags);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, c_nation, selection_flags, ht_c, c_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_orderdate + tile_offset, items, num_tile_items, selection_flags);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, year, selection_flags, ht_d, d_len, 19920101, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_revenue + tile_offset, revenue, num_tile_items, selection_flags);

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_tile_items)
    {
      if (selection_flags[ITEM])
      {
        int hash = (s_nation[ITEM] * 25 * 7 + c_nation[ITEM] * 7 +
                    (year[ITEM] - 1992)) %
                   ((1998 - 1992 + 1) * 25 * 25);
        res[hash * 6] = year[ITEM];
        res[hash * 6 + 1] = c_nation[ITEM];
        res[hash * 6 + 2] = s_nation[ITEM];
        /*atomicAdd(&res[hash * 6 + 4], revenue[ITEM]);*/
        atomicAdd(reinterpret_cast<unsigned long long *>(&res[hash * 6 + 4]),
                  (long long)(revenue[ITEM]));
      }
    }
  }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q31_kernel1(int *filter_col, int *dim_key, int *dim_val,
                            int num_tuples, int *hash_table, int num_slots)
{
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset,
                                                  items, num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 2, selection_flags,
                                                    num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items,
                                                  num_tile_items);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2,
                                                  num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, items2, selection_flags, hash_table, num_slots, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q31_kernel2(int *filter_col, int *dim_key, int *dim_val,
                            int num_tuples, int *hash_table, int num_slots)
{
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset,
                                                  items, num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 2, selection_flags,
                                                    num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items,
                                                  num_tile_items);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2,
                                                  num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, items2, selection_flags, hash_table, num_slots, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q31_kernel3(int *dim_key, int *dim_val, int num_tuples,
                            int *hash_table, int num_slots, int val_min)
{
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items,
                                                  num_tile_items);
  BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 1992, selection_flags, num_tile_items);
  BlockPredLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 1997, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items2,
                                                  num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items2, items, selection_flags, hash_table, num_slots, 19920101,
      num_tile_items);
}

void run_q31(int *lo_orderdate, int *lo_custkey, int *lo_suppkey,
             int *lo_revenue, int lo_len, int *d_datekey, int *d_year,
             int d_len, int *s_suppkey, int *s_region, int *s_nation, int s_len,
             int *c_custkey, int *c_region, int *c_nation, int c_len,
             cub::CachingDeviceAllocator &g_allocator)
{
  // Setup
  int *ht_d, *ht_c, *ht_s;
  int d_val_len = 19981230 - 19920101 + 1;
  int *res;
  int res_size = ((1998 - 1992 + 1) * 25 * 25);
  int res_array_size = res_size * 6;
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&res, res_array_size * sizeof(int)));
  CubDebugExit(cudaMemset(res, 0, res_array_size * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_d, 2 * d_val_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_s, 2 * s_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_c, 2 * c_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_d, 0, 2 * d_val_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_s, 0, 2 * s_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_c, 0, 2 * c_len * sizeof(int)));

  // Run queries
  int tile_items = 128 * 4;
  int d_val_min = 19920101;
  q31_kernel1<128, 4><<<(s_len + tile_items - 1) / tile_items, 128>>>(
      s_region, s_suppkey, s_nation, s_len, ht_s, s_len);
  q31_kernel2<128, 4><<<(c_len + tile_items - 1) / tile_items, 128>>>(
      c_region, c_custkey, c_nation, c_len, ht_c, c_len);
  q31_kernel3<128, 4><<<(d_len + tile_items - 1) / tile_items, 128>>>(
      d_datekey, d_year, d_len, ht_d, d_val_len, d_val_min);
  q31_kernel4<128, 4><<<(lo_len + tile_items - 1) / tile_items, 128>>>(
      lo_orderdate, lo_custkey, lo_suppkey, lo_revenue, lo_len, ht_s, s_len,
      ht_c, c_len, ht_d, d_val_len, res);

  // Finalize results
  int *h_res = new int[res_array_size];
  CubDebugExit(cudaMemcpy(h_res, res, res_array_size * sizeof(int),
                          cudaMemcpyDeviceToHost));
  delete[] h_res;
  CLEANUP(ht_d);
  CLEANUP(ht_s);
  CLEANUP(ht_c);
  CLEANUP(res);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q32_kernel4(int *lo_orderdate, int *lo_custkey, int *lo_suppkey,
                            int *lo_revenue, int lo_len, int *ht_s, int s_len,
                            int *ht_c, int c_len, int *ht_d, int d_len,
                            int *res)
{
  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int c_nation[ITEMS_PER_THREAD];
  int s_nation[ITEMS_PER_THREAD];
  int year[ITEMS_PER_THREAD];
  int revenue[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (lo_len + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = lo_len - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_suppkey + tile_offset,
                                                  items, num_tile_items);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, s_nation, selection_flags, ht_s, s_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_custkey + tile_offset, items, num_tile_items, selection_flags);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, c_nation, selection_flags, ht_c, c_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_orderdate + tile_offset, items, num_tile_items, selection_flags);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, year, selection_flags, ht_d, d_len, 19920101, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_revenue + tile_offset, revenue, num_tile_items, selection_flags);

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_tile_items)
    {
      if (selection_flags[ITEM])
      {
        int hash = (s_nation[ITEM] * 250 * 7 + c_nation[ITEM] * 7 +
                    (year[ITEM] - 1992)) %
                   ((1998 - 1992 + 1) * 250 * 250);
        res[hash * 4] = year[ITEM];
        res[hash * 4 + 1] = c_nation[ITEM];
        res[hash * 4 + 2] = s_nation[ITEM];
        atomicAdd(&res[hash * 4 + 3], revenue[ITEM]);
      }
    }
  }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q32_kernel1(int *filter_col, int *dim_key, int *dim_val,
                            int num_tuples, int *hash_table, int num_slots)
{
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset,
                                                  items, num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 24, selection_flags,
                                                    num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items,
                                                  num_tile_items);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2,
                                                  num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, items2, selection_flags, hash_table, num_slots, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q32_kernel2(int *filter_col, int *dim_key, int *dim_val,
                            int num_tuples, int *hash_table, int num_slots)
{
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset,
                                                  items, num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 24, selection_flags,
                                                    num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items,
                                                  num_tile_items);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2,
                                                  num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, items2, selection_flags, hash_table, num_slots, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q32_kernel3(int *dim_key, int *dim_val, int num_tuples,
                            int *hash_table, int num_slots, int val_min)
{
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items,
                                                  num_tile_items);
  BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 1992, selection_flags, num_tile_items);
  BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 1997, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items2,
                                                  num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items2, items, selection_flags, hash_table, num_slots, 19920101,
      num_tile_items);
}

void run_q32(int *lo_orderdate, int *lo_custkey, int *lo_suppkey,
             int *lo_revenue, int lo_len, int *d_datekey, int *d_year,
             int d_len, int *s_suppkey, int *s_nation, int *s_city, int s_len,
             int *c_custkey, int *c_nation, int *c_city, int c_len,
             cub::CachingDeviceAllocator &g_allocator)
{
  // Setup
  int *ht_d, *ht_c, *ht_s;
  int d_val_len = 19981230 - 19920101 + 1;
  int *res;
  int res_size = ((1998 - 1992 + 1) * 250 * 250);
  int res_array_size = res_size * 4;
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&res, res_array_size * sizeof(int)));
  CubDebugExit(cudaMemset(res, 0, res_array_size * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_d, 2 * d_val_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_s, 2 * s_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_c, 2 * c_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_d, 0, 2 * d_val_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_s, 0, 2 * s_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_c, 0, 2 * c_len * sizeof(int)));

  // Run queries
  int tile_items = 128 * 4;
  int d_val_min = 19920101;
  q32_kernel1<128, 4><<<(s_len + tile_items - 1) / tile_items, 128>>>(
      s_nation, s_suppkey, s_city, s_len, ht_s, s_len);
  q32_kernel2<128, 4><<<(c_len + tile_items - 1) / tile_items, 128>>>(
      c_nation, c_custkey, c_city, c_len, ht_c, c_len);
  q32_kernel3<128, 4><<<(d_len + tile_items - 1) / tile_items, 128>>>(
      d_datekey, d_year, d_len, ht_d, d_val_len, d_val_min);
  q32_kernel4<128, 4><<<(lo_len + tile_items - 1) / tile_items, 128>>>(
      lo_orderdate, lo_custkey, lo_suppkey, lo_revenue, lo_len, ht_s, s_len,
      ht_c, c_len, ht_d, d_val_len, res);

  // Finalize results
  int *h_res = new int[res_array_size];
  CubDebugExit(cudaMemcpy(h_res, res, res_array_size * sizeof(int),
                          cudaMemcpyDeviceToHost));
  delete[] h_res;
  CLEANUP(ht_d);
  CLEANUP(ht_s);
  CLEANUP(ht_c);
  CLEANUP(res);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q33_kernel4(int *lo_orderdate, int *lo_custkey, int *lo_suppkey,
                            int *lo_revenue, int lo_len, int *ht_s, int s_len,
                            int *ht_c, int c_len, int *ht_d, int d_len,
                            int *res)
{
  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int c_nation[ITEMS_PER_THREAD];
  int s_nation[ITEMS_PER_THREAD];
  int year[ITEMS_PER_THREAD];
  int revenue[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (lo_len + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = lo_len - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_suppkey + tile_offset,
                                                  items, num_tile_items);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, s_nation, selection_flags, ht_s, s_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_custkey + tile_offset, items, num_tile_items, selection_flags);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, c_nation, selection_flags, ht_c, c_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_orderdate + tile_offset, items, num_tile_items, selection_flags);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, year, selection_flags, ht_d, d_len, 19920101, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_revenue + tile_offset, revenue, num_tile_items, selection_flags);

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_tile_items)
    {
      if (selection_flags[ITEM])
      {
        int hash = (s_nation[ITEM] * 250 * 7 + c_nation[ITEM] * 7 +
                    (year[ITEM] - 1992)) %
                   ((1998 - 1992 + 1) * 250 * 250);
        res[hash * 4] = year[ITEM];
        res[hash * 4 + 1] = c_nation[ITEM];
        res[hash * 4 + 2] = s_nation[ITEM];
        atomicAdd(&res[hash * 4 + 3], revenue[ITEM]);
      }
    }
  }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q33_kernel1(int *dim_key, int *dim_val, int num_tuples,
                            int *hash_table, int num_slots)
{
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items,
                                                  num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 231, selection_flags,
                                                    num_tile_items);
  BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 235, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items2,
                                                  num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items2, items, selection_flags, hash_table, num_slots, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q33_kernel2(int *dim_key, int *dim_val, int num_tuples,
                            int *hash_table, int num_slots)
{
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items,
                                                  num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 231, selection_flags,
                                                    num_tile_items);
  BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 235, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items2,
                                                  num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items2, items, selection_flags, hash_table, num_slots, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q33_kernel3(int *dim_key, int *dim_val, int num_tuples,
                            int *hash_table, int num_slots, int val_min)
{
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items,
                                                  num_tile_items);
  BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 1992, selection_flags, num_tile_items);
  BlockPredLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 1997, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items2,
                                                  num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items2, items, selection_flags, hash_table, num_slots, 19920101,
      num_tile_items);
}

void run_q33(int *lo_orderdate, int *lo_custkey, int *lo_suppkey,
             int *lo_revenue, int lo_len, int *d_datekey, int *d_year,
             int d_len, int *s_suppkey, int *s_city, int s_len, int *c_custkey,
             int *c_city, int c_len, cub::CachingDeviceAllocator &g_allocator)
{
  // Setup
  int *ht_d, *ht_c, *ht_s;
  int d_val_len = 19981230 - 19920101 + 1;
  int *res;
  int res_size = ((1998 - 1992 + 1) * 250 * 250);
  int res_array_size = res_size * 4;
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&res, res_array_size * sizeof(int)));
  CubDebugExit(cudaMemset(res, 0, res_array_size * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_d, 2 * d_val_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_s, 2 * s_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_c, 2 * c_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_d, 0, 2 * d_val_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_s, 0, 2 * s_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_c, 0, 2 * c_len * sizeof(int)));

  // Run queries
  int tile_items = 128 * 4;
  int d_val_min = 19920101;
  q33_kernel1<128, 4><<<(s_len + tile_items - 1) / tile_items, 128>>>(
      s_suppkey, s_city, s_len, ht_s, s_len);
  q33_kernel2<128, 4><<<(c_len + tile_items - 1) / tile_items, 128>>>(
      c_custkey, c_city, c_len, ht_c, c_len);
  q33_kernel3<128, 4><<<(d_len + tile_items - 1) / tile_items, 128>>>(
      d_datekey, d_year, d_len, ht_d, d_val_len, d_val_min);
  q33_kernel4<128, 4><<<(lo_len + tile_items - 1) / tile_items, 128>>>(
      lo_orderdate, lo_custkey, lo_suppkey, lo_revenue, lo_len, ht_s, s_len,
      ht_c, c_len, ht_d, d_val_len, res);

  // Finalize results
  int *h_res = new int[res_array_size];
  CubDebugExit(cudaMemcpy(h_res, res, res_array_size * sizeof(int),
                          cudaMemcpyDeviceToHost));
  delete[] h_res;
  CLEANUP(ht_d);
  CLEANUP(ht_s);
  CLEANUP(ht_c);
  CLEANUP(res);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q34_kernel4(int *lo_orderdate, int *lo_custkey, int *lo_suppkey,
                            int *lo_revenue, int lo_len, int *ht_s, int s_len,
                            int *ht_c, int c_len, int *ht_d, int d_len,
                            int *res)
{
  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int c_nation[ITEMS_PER_THREAD];
  int s_nation[ITEMS_PER_THREAD];
  int year[ITEMS_PER_THREAD];
  int revenue[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (lo_len + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = lo_len - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_suppkey + tile_offset,
                                                  items, num_tile_items);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, s_nation, selection_flags, ht_s, s_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_custkey + tile_offset, items, num_tile_items, selection_flags);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, c_nation, selection_flags, ht_c, c_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_orderdate + tile_offset, items, num_tile_items, selection_flags);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, year, selection_flags, ht_d, d_len, 19920101, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_revenue + tile_offset, revenue, num_tile_items, selection_flags);

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_tile_items)
    {
      if (selection_flags[ITEM])
      {
        int hash = (s_nation[ITEM] * 250 * 7 + c_nation[ITEM] * 7 +
                    (year[ITEM] - 1992)) %
                   ((1998 - 1992 + 1) * 250 * 250);
        res[hash * 4] = year[ITEM];
        res[hash * 4 + 1] = c_nation[ITEM];
        res[hash * 4 + 2] = s_nation[ITEM];
        atomicAdd(&res[hash * 4 + 3], revenue[ITEM]);
      }
    }
  }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q34_kernel1(int *dim_key, int *dim_val, int num_tuples,
                            int *hash_table, int num_slots)
{
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items,
                                                  num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 231, selection_flags,
                                                    num_tile_items);
  BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 235, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items2,
                                                  num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items2, items, selection_flags, hash_table, num_slots, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q34_kernel2(int *dim_key, int *dim_val, int num_tuples,
                            int *hash_table, int num_slots)
{
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items,
                                                  num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 231, selection_flags,
                                                    num_tile_items);
  BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 235, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items2,
                                                  num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items2, items, selection_flags, hash_table, num_slots, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q34_kernel3(int *filter_col, int *dim_key, int *dim_val,
                            int num_tuples, int *hash_table, int num_slots,
                            int val_min)
{
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset,
                                                  items, num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 199712, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items,
                                                  num_tile_items);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2,
                                                  num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, items2, selection_flags, hash_table, num_slots, 19920101,
      num_tile_items);
}

void run_q34(int *lo_orderdate, int *lo_custkey, int *lo_suppkey,
             int *lo_revenue, int lo_len, int *d_datekey, int *d_year,
             int *d_yearmonthnum, int d_len, int *s_suppkey, int *s_city,
             int s_len, int *c_custkey, int *c_city, int c_len,
             cub::CachingDeviceAllocator &g_allocator)
{
  // Setup
  int *ht_d, *ht_c, *ht_s;
  int d_val_len = 19981230 - 19920101 + 1;
  int *res;
  int res_size = ((1998 - 1992 + 1) * 250 * 250);
  int res_array_size = res_size * 4;
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&res, res_array_size * sizeof(int)));
  CubDebugExit(cudaMemset(res, 0, res_array_size * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_d, 2 * d_val_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_s, 2 * s_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_c, 2 * c_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_d, 0, 2 * d_val_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_s, 0, 2 * s_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_c, 0, 2 * c_len * sizeof(int)));

  // Run queries
  int tile_items = 128 * 4;
  int d_val_min = 19920101;
  q34_kernel1<128, 4><<<(s_len + tile_items - 1) / tile_items, 128>>>(
      s_suppkey, s_city, s_len, ht_s, s_len);
  q34_kernel2<128, 4><<<(c_len + tile_items - 1) / tile_items, 128>>>(
      c_custkey, c_city, c_len, ht_c, c_len);
  q34_kernel3<128, 4><<<(d_len + tile_items - 1) / tile_items, 128>>>(
      d_yearmonthnum, d_datekey, d_year, d_len, ht_d, d_val_len, d_val_min);
  q34_kernel4<128, 4><<<(lo_len + tile_items - 1) / tile_items, 128>>>(
      lo_orderdate, lo_custkey, lo_suppkey, lo_revenue, lo_len, ht_s, s_len,
      ht_c, c_len, ht_d, d_val_len, res);

  // Finalize results
  int *h_res = new int[res_array_size];
  CubDebugExit(cudaMemcpy(h_res, res, res_array_size * sizeof(int),
                          cudaMemcpyDeviceToHost));
  delete[] h_res;
  CLEANUP(ht_d);
  CLEANUP(ht_c);
  CLEANUP(ht_s);
  CLEANUP(res);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q41_kernel5(int *lo_orderdate, int *lo_partkey, int *lo_custkey,
                            int *lo_suppkey, int *lo_revenue,
                            int *lo_supplycost, int lo_len, int *ht_p,
                            int p_len, int *ht_s, int s_len, int *ht_c,
                            int c_len, int *ht_d, int d_len, int *res)
{
  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int c_nation[ITEMS_PER_THREAD];
  int s_nation[ITEMS_PER_THREAD];
  int year[ITEMS_PER_THREAD];
  int revenue[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (lo_len + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = lo_len - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_suppkey + tile_offset,
                                                  items, num_tile_items);
  BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, selection_flags, ht_s, s_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_custkey + tile_offset, items, num_tile_items, selection_flags);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, c_nation, selection_flags, ht_c, c_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_partkey + tile_offset, items, num_tile_items, selection_flags);
  BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, selection_flags, ht_p, p_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_orderdate + tile_offset, items, num_tile_items, selection_flags);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, year, selection_flags, ht_d, d_len, 19920101, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_revenue + tile_offset, revenue, num_tile_items, selection_flags);
  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_supplycost + tile_offset, items, num_tile_items, selection_flags);

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items)
    {
      if (selection_flags[ITEM])
      {
        int hash = (c_nation[ITEM] * 7 + (year[ITEM] - 1992)) %
                   ((1998 - 1992 + 1) * 25);
        res[hash * 4] = year[ITEM];
        res[hash * 4 + 1] = c_nation[ITEM];
        /*atomicAdd(&res[hash * 4 + 2], (1));*/
        /*atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 4 + 2]),
         * (long long)(1));*/
        atomicAdd(reinterpret_cast<unsigned long long *>(&res[hash * 4 + 2]),
                  (long long)(revenue[ITEM] - items[ITEM]));
      }
    }
  }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q41_kernel1(int *filter_col, int *dim_key, int num_tuples,
                            int *hash_table, int num_slots)
{
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
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
__global__ void q41_kernel3(int *filter_col, int *dim_key, int num_tuples,
                            int *hash_table, int num_slots)
{
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset,
                                                  items, num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 0, selection_flags,
                                                    num_tile_items);
  BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1, selection_flags,
                                                      num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items,
                                                  num_tile_items);
  BlockBuildSelectivePHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, selection_flags, hash_table, num_slots, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q41_kernel2(int *filter_col, int *dim_key, int *dim_val,
                            int num_tuples, int *hash_table, int num_slots)
{
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
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
__global__ void q41_kernel4(int *dim_key, int *dim_val, int num_tuples,
                            int *hash_table, int num_slots, int val_min)
{
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
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

void run_q41(int *lo_orderdate, int *lo_custkey, int *lo_partkey,
             int *lo_suppkey, int *lo_revenue, int *lo_supplycost, int lo_len,
             int *d_datekey, int *d_year, int d_len, int *p_partkey,
             int *p_mfgr, int p_len, int *s_suppkey, int *s_region, int s_len,
             int *c_custkey, int *c_region, int *c_nation, int c_len,
             cub::CachingDeviceAllocator &g_allocator)
{
  // Setup
  int *ht_d, *ht_c, *ht_s, *ht_p;
  int d_val_len = 19981230 - 19920101 + 1;
  int *res;
  int res_size = ((1998 - 1992 + 1) * 25);
  int ht_entries = 4; // int,int,long long
  int res_array_size = res_size * ht_entries;
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&res, res_array_size * sizeof(int)));
  CubDebugExit(cudaMemset(res, 0, res_array_size * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_d, 2 * d_val_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_s, 2 * s_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_c, 2 * c_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_p, 2 * p_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_d, 0, 2 * d_val_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_s, 0, 2 * s_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_c, 0, 2 * c_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_p, 0, 2 * p_len * sizeof(int)));

  // Run queries
  int tile_items = 128 * 4;
  int d_val_min = 19920101;
  q41_kernel1<128, 4><<<(s_len + tile_items - 1) / tile_items, 128>>>(
      s_region, s_suppkey, s_len, ht_s, s_len);
  q41_kernel2<128, 4><<<(c_len + tile_items - 1) / tile_items, 128>>>(
      c_region, c_custkey, c_nation, c_len, ht_c, c_len);
  q41_kernel3<128, 4><<<(p_len + tile_items - 1) / tile_items, 128>>>(
      p_mfgr, p_partkey, p_len, ht_p, p_len);
  q41_kernel4<128, 4><<<(d_len + tile_items - 1) / tile_items, 128>>>(
      d_datekey, d_year, d_len, ht_d, d_val_len, d_val_min);
  q41_kernel5<128, 4><<<(lo_len + tile_items - 1) / tile_items, 128>>>(
      lo_orderdate, lo_partkey, lo_custkey, lo_suppkey, lo_revenue,
      lo_supplycost, lo_len, ht_p, p_len, ht_s, s_len, ht_c, c_len, ht_d,
      d_val_len, res);

  // Finalize results
  int *h_res = new int[res_array_size];
  CubDebugExit(cudaMemcpy(h_res, res, res_array_size * sizeof(int),
                          cudaMemcpyDeviceToHost));
  delete[] h_res;
  CLEANUP(ht_d);
  CLEANUP(ht_s);
  CLEANUP(ht_c);
  CLEANUP(ht_p);
  CLEANUP(res);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q42_kernel5(int *lo_orderdate, int *lo_partkey, int *lo_custkey,
                            int *lo_suppkey, int *lo_revenue,
                            int *lo_supplycost, int lo_len, int *ht_p,
                            int p_len, int *ht_s, int s_len, int *ht_c,
                            int c_len, int *ht_d, int d_len, int *res)
{
  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int category[ITEMS_PER_THREAD];
  int s_nation[ITEMS_PER_THREAD];
  int year[ITEMS_PER_THREAD];
  int revenue[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (lo_len + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = lo_len - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_suppkey + tile_offset,
                                                  items, num_tile_items);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, s_nation, selection_flags, ht_s, s_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_custkey + tile_offset, items, num_tile_items, selection_flags);
  BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, selection_flags, ht_c, c_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_partkey + tile_offset, items, num_tile_items, selection_flags);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, category, selection_flags, ht_p, p_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_orderdate + tile_offset, items, num_tile_items, selection_flags);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, year, selection_flags, ht_d, d_len, 19920101, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_revenue + tile_offset, revenue, num_tile_items, selection_flags);
  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_supplycost + tile_offset, items, num_tile_items, selection_flags);

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items)
    {
      if (selection_flags[ITEM])
      {
        /*int hash = (category[ITEM] * 7 * 25 + s_nation[ITEM] * 7 +
         * (year[ITEM] - 1992)) % ((1998-1992+1) * 25 * 55);*/
        int hash = ((year[ITEM] - 1992) * 25 * 25 + s_nation[ITEM] * 25 +
                    category[ITEM]) %
                   ((1998 - 1992 + 1) * 25 * 25);
        res[hash * 6] = year[ITEM];
        res[hash * 6 + 1] = s_nation[ITEM];
        res[hash * 6 + 2] = category[ITEM];
        atomicAdd(reinterpret_cast<unsigned long long *>(&res[hash * 6 + 4]),
                  (long long)(revenue[ITEM] - items[ITEM]));
      }
    }
  }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q42_kernel2(int *filter_col, int *dim_key, int num_tuples,
                            int *hash_table, int num_slots)
{
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
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
__global__ void q42_kernel3(int *filter_col, int *dim_key, int *dim_val,
                            int num_tuples, int *hash_table, int num_slots)
{
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset,
                                                  items, num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 0, selection_flags,
                                                    num_tile_items);
  BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1, selection_flags,
                                                      num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items,
                                                  num_tile_items);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2,
                                                  num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, items2, selection_flags, hash_table, num_slots, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q42_kernel1(int *filter_col, int *dim_key, int *dim_val,
                            int num_tuples, int *hash_table, int num_slots)
{
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
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
__global__ void q42_kernel4(int *dim_key, int *dim_val, int num_tuples,
                            int *hash_table, int num_slots, int val_min)
{
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = num_tuples - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items,
                                                  num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 1997, selection_flags, num_tile_items);
  BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 1998, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items2,
                                                  num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items2, items, selection_flags, hash_table, num_slots, val_min,
      num_tile_items);
}

void run_q42(int *lo_orderdate, int *lo_custkey, int *lo_partkey,
             int *lo_suppkey, int *lo_revenue, int *lo_supplycost, int lo_len,
             int *d_datekey, int *d_year, int d_len, int *p_partkey,
             int *p_mfgr, int *p_category, int p_len, int *s_suppkey,
             int *s_region, int *s_nation, int s_len, int *c_custkey,
             int *c_region, int c_len,
             cub::CachingDeviceAllocator &g_allocator)
{
  // Setup
  int *ht_d, *ht_c, *ht_s, *ht_p;
  int d_val_len = 19981230 - 19920101 + 1;
  int *res;
  int res_size = ((1998 - 1992 + 1) * 25 * 25);
  int ht_entries = 6;
  int res_array_size = res_size * ht_entries;
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&res, res_array_size * sizeof(int)));
  CubDebugExit(cudaMemset(res, 0, res_array_size * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_d, 2 * d_val_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_s, 2 * s_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_c, 2 * c_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_p, 2 * p_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_d, 0, 2 * d_val_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_s, 0, 2 * s_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_c, 0, 2 * c_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_p, 0, 2 * p_len * sizeof(int)));

  // Run queries
  int tile_items = 128 * 4;
  int d_val_min = 19920101;
  q42_kernel1<128, 4><<<(s_len + tile_items - 1) / tile_items, 128>>>(
      s_region, s_suppkey, s_nation, s_len, ht_s, s_len);
  q42_kernel2<128, 4><<<(c_len + tile_items - 1) / tile_items, 128>>>(
      c_region, c_custkey, c_len, ht_c, c_len);
  q42_kernel3<128, 4><<<(p_len + tile_items - 1) / tile_items, 128>>>(
      p_mfgr, p_partkey, p_category, p_len, ht_p, p_len);
  q42_kernel4<128, 4><<<(d_len + tile_items - 1) / tile_items, 128>>>(
      d_datekey, d_year, d_len, ht_d, d_val_len, d_val_min);
  q42_kernel5<128, 4><<<(lo_len + tile_items - 1) / tile_items, 128>>>(
      lo_orderdate, lo_partkey, lo_custkey, lo_suppkey, lo_revenue,
      lo_supplycost, lo_len, ht_p, p_len, ht_s, s_len, ht_c, c_len, ht_d,
      d_val_len, res);

  // Finalize results
  int *h_res = new int[res_array_size];
  CubDebugExit(cudaMemcpy(h_res, res, res_array_size * sizeof(int),
                          cudaMemcpyDeviceToHost));
  delete[] h_res;
  CLEANUP(ht_d);
  CLEANUP(ht_s);
  CLEANUP(ht_c);
  CLEANUP(ht_p);
  CLEANUP(res);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q43_kernel5(int *lo_orderdate, int *lo_partkey, int *lo_custkey,
                            int *lo_suppkey, int *lo_revenue,
                            int *lo_supplycost, int lo_len, int *ht_p,
                            int p_len, int *ht_s, int s_len, int *ht_c,
                            int c_len, int *ht_d, int d_len, int *res)
{
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int brand[ITEMS_PER_THREAD];
  int s_city[ITEMS_PER_THREAD];
  int year[ITEMS_PER_THREAD];
  int revenue[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (lo_len + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = lo_len - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_suppkey + tile_offset,
                                                  items, num_tile_items);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, s_city, selection_flags, ht_s, s_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_custkey + tile_offset, items, num_tile_items, selection_flags);
  BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, selection_flags, ht_c, c_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_partkey + tile_offset, items, num_tile_items, selection_flags);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, brand, selection_flags, ht_p, p_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_orderdate + tile_offset, items, num_tile_items, selection_flags);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, year, selection_flags, ht_d, d_len, 19920101, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags))
  {
    return;
  }

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_revenue + tile_offset, revenue, num_tile_items, selection_flags);
  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_supplycost + tile_offset, items, num_tile_items, selection_flags);

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items)
    {
      if (selection_flags[ITEM])
      {
        /*int hash = (category[ITEM] * 7 * 25 + s_nation[ITEM] * 7 +
         * (year[ITEM] - 1992)) % ((1998-1992+1) * 25 * 55);*/
        int hash = ((year[ITEM] - 1992) * 250 * 1000 + s_city[ITEM] * 1000 +
                    brand[ITEM]) %
                   ((1998 - 1992 + 1) * 250 * 1000);
        res[hash * 4] = year[ITEM];
        res[hash * 4 + 1] = s_city[ITEM];
        res[hash * 4 + 2] = brand[ITEM];
        atomicAdd(&res[hash * 4 + 3], (revenue[ITEM] - items[ITEM]));
      }
    }
  }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q43_kernel2(int *filter_col, int *dim_key, int num_tuples,
                            int *hash_table, int num_slots)
{
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
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
__global__ void q43_kernel3(int *filter_col, int *dim_key, int *dim_val,
                            int num_tuples, int *hash_table, int num_slots)
{
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset,
                                                  items, num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 3, selection_flags,
                                                    num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items,
                                                  num_tile_items);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2,
                                                  num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, items2, selection_flags, hash_table, num_slots, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q43_kernel1(int *filter_col, int *dim_key, int *dim_val,
                            int num_tuples, int *hash_table, int num_slots)
{
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset,
                                                  items, num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 24, selection_flags,
                                                    num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items,
                                                  num_tile_items);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2,
                                                  num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, items2, selection_flags, hash_table, num_slots, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q43_kernel4(int *dim_key, int *dim_val, int num_tuples,
                            int *hash_table, int num_slots, int val_min)
{
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1)
  {
    num_tile_items = num_tuples - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items,
                                                  num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 1997, selection_flags, num_tile_items);
  BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 1998, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items2,
                                                  num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items2, items, selection_flags, hash_table, num_slots, val_min,
      num_tile_items);
}

void run_q43(int *lo_orderdate, int *lo_custkey, int *lo_partkey,
             int *lo_suppkey, int *lo_revenue, int *lo_supplycost, int lo_len,
             int *d_datekey, int *d_year, int d_len, int *p_partkey,
             int *p_category, int *p_brand1, int p_len, int *s_suppkey,
             int *s_nation, int *s_city, int s_len, int *c_custkey,
             int *c_region, int c_len,
             cub::CachingDeviceAllocator &g_allocator)
{
  // Setup
  int *ht_d, *ht_c, *ht_s, *ht_p;
  int d_val_len = 19981230 - 19920101 + 1;
  int *res;
  int res_size = ((1998 - 1992 + 1) * 250 * 1000);
  int ht_entries = 4;
  int res_array_size = res_size * ht_entries;
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&res, res_array_size * sizeof(int)));
  CubDebugExit(cudaMemset(res, 0, res_array_size * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_d, 2 * d_val_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_s, 2 * s_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_c, 2 * c_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_p, 2 * p_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_d, 0, 2 * d_val_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_s, 0, 2 * s_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_c, 0, 2 * c_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_p, 0, 2 * p_len * sizeof(int)));

  // Run queries
  int tile_items = 128 * 4;
  int d_val_min = 19920101;
  q43_kernel1<128, 4><<<(s_len + tile_items - 1) / tile_items, 128>>>(
      s_nation, s_suppkey, s_city, s_len, ht_s, s_len);
  q43_kernel2<128, 4><<<(c_len + tile_items - 1) / tile_items, 128>>>(
      c_region, c_custkey, c_len, ht_c, c_len);
  q43_kernel3<128, 4><<<(p_len + tile_items - 1) / tile_items, 128>>>(
      p_category, p_partkey, p_brand1, p_len, ht_p, p_len);
  q43_kernel4<128, 4><<<(d_len + tile_items - 1) / tile_items, 128>>>(
      d_datekey, d_year, d_len, ht_d, d_val_len, d_val_min);
  q43_kernel5<128, 4><<<(lo_len + tile_items - 1) / tile_items, 128>>>(
      lo_orderdate, lo_partkey, lo_custkey, lo_suppkey, lo_revenue,
      lo_supplycost, lo_len, ht_p, p_len, ht_s, s_len, ht_c, c_len, ht_d,
      d_val_len, res);

  // Finalize results
  int *h_res = new int[res_array_size];
  CubDebugExit(cudaMemcpy(h_res, res, res_array_size * sizeof(int),
                          cudaMemcpyDeviceToHost));
  delete[] h_res;
  CLEANUP(ht_d);
  CLEANUP(ht_s);
  CLEANUP(ht_c);
  CLEANUP(ht_p);
  CLEANUP(res);
}

/**
 * Main
 */
int main(int argc, char **argv)
{
  int num_trials = 3;

  // Initialize command line
  CommandLineArgs args(argc, argv);
  args.GetCmdLineArgument("t", num_trials);

  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
    printf("%s "
           "[--t=<num trials>] "
           "[--v] "
           "\n",
           argv[0]);
    exit(0);
  }

  // Initialize device
  CubDebugExit(args.DeviceInit());

  // Load host data
  int *h_lo_orderdate = loadColumn<int>("lo_orderdate", LO_LEN);
  int *h_lo_custkey = loadColumn<int>("lo_custkey", LO_LEN);
  int *h_lo_partkey = loadColumn<int>("lo_partkey", LO_LEN);
  int *h_lo_suppkey = loadColumn<int>("lo_suppkey", LO_LEN);
  int *h_lo_discount = loadColumn<int>("lo_discount", LO_LEN);
  int *h_lo_quantity = loadColumn<int>("lo_quantity", LO_LEN);
  int *h_lo_extendedprice = loadColumn<int>("lo_extendedprice", LO_LEN);
  int *h_lo_revenue = loadColumn<int>("lo_revenue", LO_LEN);
  int *h_lo_supplycost = loadColumn<int>("lo_supplycost", LO_LEN);

  int *h_p_partkey = loadColumn<int>("p_partkey", P_LEN);
  int *h_p_brand1 = loadColumn<int>("p_brand1", P_LEN);
  int *h_p_category = loadColumn<int>("p_category", P_LEN);
  int *h_p_mfgr = loadColumn<int>("p_mfgr", P_LEN);

  int *h_s_suppkey = loadColumn<int>("s_suppkey", S_LEN);
  int *h_s_region = loadColumn<int>("s_region", S_LEN);
  int *h_s_nation = loadColumn<int>("s_nation", S_LEN);
  int *h_s_city = loadColumn<int>("s_city", S_LEN);

  int *h_c_custkey = loadColumn<int>("c_custkey", C_LEN);
  int *h_c_nation = loadColumn<int>("c_nation", C_LEN);
  int *h_c_region = loadColumn<int>("c_region", C_LEN);
  int *h_c_city = loadColumn<int>("c_city", C_LEN);

  int *h_d_datekey = loadColumn<int>("d_datekey", D_LEN);
  int *h_d_year = loadColumn<int>("d_year", D_LEN);
  int *h_d_yearmonthnum = loadColumn<int>("d_yearmonthnum", D_LEN);

  // Load device data
  int *d_lo_orderdate = loadToGPU<int>(h_lo_orderdate, LO_LEN, g_allocator);
  int *d_lo_custkey = loadToGPU<int>(h_lo_custkey, LO_LEN, g_allocator);
  int *d_lo_partkey = loadToGPU<int>(h_lo_partkey, LO_LEN, g_allocator);
  int *d_lo_suppkey = loadToGPU<int>(h_lo_suppkey, LO_LEN, g_allocator);
  int *d_lo_discount = loadToGPU<int>(h_lo_discount, LO_LEN, g_allocator);
  int *d_lo_quantity = loadToGPU<int>(h_lo_quantity, LO_LEN, g_allocator);
  int *d_lo_extendedprice =
      loadToGPU<int>(h_lo_extendedprice, LO_LEN, g_allocator);
  int *d_lo_revenue = loadToGPU<int>(h_lo_revenue, LO_LEN, g_allocator);
  int *d_lo_supplycost = loadToGPU<int>(h_lo_supplycost, LO_LEN, g_allocator);

  int *d_p_partkey = loadToGPU<int>(h_p_partkey, P_LEN, g_allocator);
  int *d_p_brand1 = loadToGPU<int>(h_p_brand1, P_LEN, g_allocator);
  int *d_p_category = loadToGPU<int>(h_p_category, P_LEN, g_allocator);
  int *d_p_mfgr = loadToGPU<int>(h_p_mfgr, P_LEN, g_allocator);

  int *d_s_suppkey = loadToGPU<int>(h_s_suppkey, S_LEN, g_allocator);
  int *d_s_region = loadToGPU<int>(h_s_region, S_LEN, g_allocator);
  int *d_s_nation = loadToGPU<int>(h_s_nation, S_LEN, g_allocator);
  int *d_s_city = loadToGPU<int>(h_s_city, S_LEN, g_allocator);

  int *d_c_custkey = loadToGPU<int>(h_c_custkey, C_LEN, g_allocator);
  int *d_c_region = loadToGPU<int>(h_c_region, C_LEN, g_allocator);
  int *d_c_nation = loadToGPU<int>(h_c_nation, C_LEN, g_allocator);
  int *d_c_city = loadToGPU<int>(h_c_city, C_LEN, g_allocator);

  int *d_d_datekey = loadToGPU<int>(h_d_datekey, D_LEN, g_allocator);
  int *d_d_year = loadToGPU<int>(h_d_year, D_LEN, g_allocator);
  int *d_d_yearmonthnum = loadToGPU<int>(h_d_yearmonthnum, D_LEN, g_allocator);

  // Run queries
  std::vector<int> seeds = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  auto rng = std::default_random_engine{};
  std::shuffle(std::begin(seeds), std::end(seeds), rng);
  for (int t = 0; t < num_trials; t++)
  {
    for (int s : seeds)
    {
      switch (s)
      {
      case 0:
        run_q11(d_lo_orderdate, d_lo_discount, d_lo_quantity,
                d_lo_extendedprice, LO_LEN, g_allocator);
        break;
      case 1:
        run_q12(d_lo_orderdate, d_lo_discount, d_lo_quantity,
                d_lo_extendedprice, LO_LEN, g_allocator);
        break;
      case 2:
        run_q13(d_lo_orderdate, d_lo_discount, d_lo_quantity,
                d_lo_extendedprice, LO_LEN, g_allocator);
        break;
      case 3:
        run_q21(d_lo_orderdate, d_lo_partkey, d_lo_suppkey, d_lo_revenue,
                LO_LEN, d_p_partkey, d_p_brand1, d_p_category, P_LEN,
                d_d_datekey, d_d_year, D_LEN, d_s_suppkey, d_s_region, S_LEN,
                g_allocator);
        break;
      case 4:
        run_q22(d_lo_orderdate, d_lo_partkey, d_lo_suppkey, d_lo_revenue,
                LO_LEN, d_p_partkey, d_p_brand1, P_LEN, d_d_datekey, d_d_year,
                D_LEN, d_s_suppkey, d_s_region, S_LEN, g_allocator);
        break;
      case 5:
        run_q23(d_lo_orderdate, d_lo_partkey, d_lo_suppkey, d_lo_revenue,
                LO_LEN, d_p_partkey, d_p_brand1, P_LEN, d_d_datekey, d_d_year,
                D_LEN, d_s_suppkey, d_s_region, S_LEN, g_allocator);
        break;
      case 6:
        run_q31(d_lo_orderdate, d_lo_custkey, d_lo_suppkey, d_lo_revenue,
                LO_LEN, d_d_datekey, d_d_year, D_LEN, d_s_suppkey, d_s_region,
                d_s_nation, S_LEN, d_c_custkey, d_c_region, d_c_nation, C_LEN,
                g_allocator);
        break;
      case 7:
        run_q32(d_lo_orderdate, d_lo_custkey, d_lo_suppkey, d_lo_revenue,
                LO_LEN, d_d_datekey, d_d_year, D_LEN, d_s_suppkey, d_s_nation,
                d_s_city, S_LEN, d_c_custkey, d_c_nation, d_c_city, C_LEN,
                g_allocator);
        break;
      case 8:
        run_q33(d_lo_orderdate, d_lo_custkey, d_lo_suppkey, d_lo_revenue,
                LO_LEN, d_d_datekey, d_d_year, D_LEN, d_s_suppkey, d_s_city,
                S_LEN, d_c_custkey, d_c_city, C_LEN, g_allocator);
        break;
      case 9:
        run_q34(d_lo_orderdate, d_lo_custkey, d_lo_suppkey, d_lo_revenue,
                LO_LEN, d_d_datekey, d_d_year, d_d_yearmonthnum, D_LEN,
                d_s_suppkey, d_s_city, S_LEN, d_c_custkey, d_c_city, C_LEN,
                g_allocator);
        break;
      case 10:
        run_q41(d_lo_orderdate, d_lo_custkey, d_lo_partkey, d_lo_suppkey,
                d_lo_revenue, d_lo_supplycost, LO_LEN, d_d_datekey, d_d_year,
                D_LEN, d_p_partkey, d_p_mfgr, P_LEN, d_s_suppkey, d_s_region,
                S_LEN, d_c_custkey, d_c_region, d_c_nation, C_LEN, g_allocator);
        break;
      case 11:
        run_q42(d_lo_orderdate, d_lo_custkey, d_lo_partkey, d_lo_suppkey,
                d_lo_revenue, d_lo_supplycost, LO_LEN, d_d_datekey, d_d_year,
                D_LEN, d_p_partkey, d_p_mfgr, d_p_category, P_LEN, d_s_suppkey,
                d_s_region, d_s_nation, S_LEN, d_c_custkey, d_c_region, C_LEN,
                g_allocator);
        break;
      case 12:
        run_q43(d_lo_orderdate, d_lo_custkey, d_lo_partkey, d_lo_suppkey,
                d_lo_revenue, d_lo_supplycost, LO_LEN, d_d_datekey, d_d_year,
                D_LEN, d_p_partkey, d_p_category, d_p_brand1, P_LEN,
                d_s_suppkey, d_s_nation, d_s_city, S_LEN, d_c_custkey,
                d_c_region, C_LEN, g_allocator);
        break;
      }
    }
  }
}
