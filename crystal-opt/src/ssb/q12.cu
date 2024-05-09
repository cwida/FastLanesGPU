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
__global__ void DeviceSelectIf(int *lo_orderdate, int *lo_discount,
                               int *lo_quantity, int *lo_extendedprice,
                               int lo_num_entries,
                               unsigned long long *revenue) {
  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];

  long long sum = 0;

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (lo_num_entries + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
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
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
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

  if (threadIdx.x == 0) {
    atomicAdd(revenue, aggregate);
  }
}

float runQuery(int *lo_orderdate, int *lo_discount, int *lo_quantity,
               int *lo_extendedprice, int lo_num_entries,
               cub::CachingDeviceAllocator &g_allocator) {
  SETUP_TIMING();

  float time_query;
  chrono::high_resolution_clock::time_point st, finish;
  st = chrono::high_resolution_clock::now();

  cudaEventRecord(start, 0);

  unsigned long long *d_sum = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void **)&d_sum, sizeof(long long)));

  cudaMemset(d_sum, 0, sizeof(long long));

  // Run
  int tile_items = 128 * 4;
  DeviceSelectIf<128, 4>
      <<<(lo_num_entries + tile_items - 1) / tile_items, 128>>>(
          lo_orderdate, lo_discount, lo_quantity, lo_extendedprice,
          lo_num_entries, d_sum);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_query, start, stop);

  unsigned long long revenue;
  CubDebugExit(
      cudaMemcpy(&revenue, d_sum, sizeof(long long), cudaMemcpyDeviceToHost));

  finish = chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = finish - st;

  cout << "Revenue: " << revenue << endl;
  cout << "Time Taken Total: " << diff.count() * 1000 << endl;

  CLEANUP(d_sum);

  return time_query;
}

/**
 * Main
 */
int main(int argc, char **argv) {
  int num_trials = 3;

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
  int *h_lo_discount = loadColumn<int>("lo_discount", LO_LEN);
  int *h_lo_quantity = loadColumn<int>("lo_quantity", LO_LEN);
  int *h_lo_extendedprice = loadColumn<int>("lo_extendedprice", LO_LEN);
  int *h_d_datekey = loadColumn<int>("d_datekey", D_LEN);
  int *h_d_year = loadColumn<int>("d_year", D_LEN);

  cout << "** LOADED DATA **" << endl;

  int *d_lo_orderdate = loadToGPU<int>(h_lo_orderdate, LO_LEN, g_allocator);
  int *d_lo_discount = loadToGPU<int>(h_lo_discount, LO_LEN, g_allocator);
  int *d_lo_quantity = loadToGPU<int>(h_lo_quantity, LO_LEN, g_allocator);
  int *d_lo_extendedprice =
      loadToGPU<int>(h_lo_extendedprice, LO_LEN, g_allocator);
  int *d_d_datekey = loadToGPU<int>(h_d_datekey, D_LEN, g_allocator);
  int *d_d_year = loadToGPU<int>(h_d_year, D_LEN, g_allocator);

  cout << "** LOADED DATA TO GPU **" << endl;

  for (int t = 0; t < num_trials; t++) {
    float time_query;
    time_query = runQuery(d_lo_orderdate, d_lo_discount, d_lo_quantity,
                          d_lo_extendedprice, LO_LEN, g_allocator);
    cout << "{"
         << "\"query\":12"
         << ",\"time_query\":" << time_query << "}" << endl;
  }

  return 0;
}
