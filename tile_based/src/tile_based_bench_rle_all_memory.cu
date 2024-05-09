#include "config/tile_based/config.hpp"
#include "debug/pretty_print.hpp"
#include "gpu/helper.hpp"
#include "mixbench-cuda/lcutil.h"
#include "tile_based/rlebinpack_kernel.cuh"
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <iostream>
#include <string>

std::pair<uint, uint>
rleBinPack(uint*& in, uint*& value, uint*& run_length, uint*& val_offsets, uint*& rl_offsets, uint num_entries) {
	uint val_offset = 0;
	uint rl_offset  = 0;

	uint block_size      = 512;
	uint elem_per_thread = 1;
	uint tile_size       = block_size * elem_per_thread;

	// nonblock
	block_size = tile_size;

	uint miniblock_count = 4;
	uint total_count     = num_entries;
	uint first_val       = in[0];

	value[0] = block_size;
	value[1] = miniblock_count;
	value[2] = total_count;
	value[3] = first_val;

	run_length[0] = block_size;
	run_length[1] = miniblock_count;
	run_length[2] = total_count;
	run_length[3] = first_val;

	val_offset += 4;
	rl_offset += 4;

	uint num_tiles = (num_entries + tile_size - 1) / tile_size;

	uint* val = new uint[tile_size]();
	uint* rl  = new uint[tile_size]();

	for (uint tile_start = 0; tile_start < num_entries; tile_start += tile_size) {
		uint block_index = tile_start / block_size;

		uint count = 0;
		val[count] = in[0];
		uint run   = 1;
		for (int i = 1; i < tile_size; i++) {
			if (in[i] != in[i - 1]) {
				rl[count] = run;
				count++;
				val[count] = in[i];
				run        = 1;
			} else {
				run++;
			}
		}
		rl[count] = run;
		count++;

		// non block
		int bl_size     = count;
		int block_start = 0;

		rl_offsets[block_index]  = rl_offset;
		val_offsets[block_index] = val_offset;

		uint min_val = val[block_start];
		uint min_rl  = rl[block_start];
		for (int i = 1; i < bl_size; i++) {
			if (val[block_start + i] < min_val) min_val = val[block_start + i];
			if (rl[block_start + i] < min_rl) min_rl = rl[block_start + i];
		}

		uint val_bitwidth = 0;
		uint rl_bitwidth  = 0;

		for (int i = block_start; i < block_start + bl_size; i++) {
			val[i]        = val[i] - min_val;
			rl[i]         = rl[i] - min_rl;
			uint bitwidth = uint(ceil(log2(val[i] + 1)));
			val_bitwidth  = std::max(val_bitwidth, bitwidth);
			bitwidth      = uint(ceil(log2(rl[i] + 1)));
			rl_bitwidth   = std::max(rl_bitwidth, bitwidth);
		}

		value[val_offset]     = min_val;
		run_length[rl_offset] = min_rl;
		val_offset++;
		rl_offset++;

		value[val_offset]     = val_bitwidth + (val_bitwidth << 8) + (val_bitwidth << 16) + (val_bitwidth << 24);
		run_length[rl_offset] = rl_bitwidth + (rl_bitwidth << 8) + (rl_bitwidth << 16) + (rl_bitwidth << 24);
		val_offset++;
		rl_offset++;

		if (block_start == (bl_size * (elem_per_thread - 1))) { // if last block
			value[val_offset]     = count - bl_size * (elem_per_thread - 1);
			run_length[rl_offset] = count - bl_size * (elem_per_thread - 1);
		} else {
			value[val_offset]     = bl_size;
			run_length[rl_offset] = bl_size;
		}
		val_offset++;
		rl_offset++;

		uint bitwidth = val_bitwidth;
		uint shift    = 0;
		for (int i = block_start; i < block_start + bl_size; i++) {
			if (shift + bitwidth > 32) {
				if (shift != 32) value[val_offset] += val[i] << shift;
				val_offset++;
				shift             = (shift + bitwidth) & (32 - 1);
				value[val_offset] = val[i] >> (bitwidth - shift);
			} else {
				value[val_offset] += val[i] << shift;
				shift += bitwidth;
			}
		}
		val_offset++;

		bitwidth = rl_bitwidth;
		shift    = 0;
		for (int i = block_start; i < block_start + bl_size; i++) {
			if (shift + bitwidth > 32) {
				if (shift != 32) run_length[rl_offset] += rl[i] << shift;
				rl_offset++;
				shift                 = (shift + bitwidth) & (32 - 1);
				run_length[rl_offset] = rl[i] >> (bitwidth - shift);
			} else {
				run_length[rl_offset] += rl[i] << shift;
				shift += bitwidth;
			}
		}
		rl_offset++;

		in += tile_size;
	}

	val_offsets[num_entries / block_size] = val_offset;
	rl_offsets[num_entries / block_size]  = rl_offset;

	return std::make_pair(val_offset, rl_offset);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void
runRBinKernel(int* col, uint* val_block_start, uint* val_data, uint* rl_block_start, uint* rl_data, int num_entries) {
	int tile_size   = BLOCK_THREADS * ITEMS_PER_THREAD;
	int tile_idx    = blockIdx.x;
	int tile_offset = tile_idx * tile_size;

	// Load a segment of consecutive items that are blocked across threads
	int val_block[ITEMS_PER_THREAD];
	int rl_block[ITEMS_PER_THREAD];

	int  num_tiles      = (num_entries + tile_size - 1) / tile_size;
	int  num_tile_items = tile_size;
	bool is_last_tile   = false;
	if (tile_idx == num_tiles - 1) {
		num_tile_items = num_entries - tile_offset;
		is_last_tile   = true;
	}

	extern __shared__ uint shared_buffer[];
	LoadRBinPack<BLOCK_THREADS, ITEMS_PER_THREAD>(val_block_start,
	                                              rl_block_start,
	                                              val_data,
	                                              rl_data,
	                                              shared_buffer,
	                                              val_block,
	                                              rl_block,
	                                              is_last_tile,
	                                              num_tile_items);

	__syncthreads();

	for (int i = 0; i < ITEMS_PER_THREAD; i++) {
		col[tile_size * tile_idx + i * BLOCK_THREADS + threadIdx.x] = val_block[i];
	}
}

namespace tile_based {
template <typename T>
T* loadColumnToGPU(T* src, int len) {
	T* dest = nullptr;
	cudaMalloc((void**)&dest, sizeof(T) * len);
	CUDA_SAFE_CALL(cudaMemcpy(dest, src, sizeof(T) * len, cudaMemcpyHostToDevice));
	return dest;
}

} // namespace tile_based

int main() {
	fastlanes::gpu::helper::print_cuda_info();

	cudaSetDevice(0);
	StoreDeviceInfo(stdout);
	/* Init */
	std::cout << "------------------------------------ \n";
	std::cout << "-- Init :  \n";

	uint64_t  n_tup            = 1 << 28;
	auto*     h_org_arr        = new uint32_t[n_tup];
	int       block_size       = 128;
	int       elem_per_thread  = 4;
	int       tile_size        = block_size * elem_per_thread;
	int       n_blocks         = n_tup / block_size;
	uint64_t  n_ofs            = n_blocks + 1;
	auto*     h_val_arr        = new uint32_t[n_tup]();
	auto*     h_len_arr        = new uint32_t[n_tup]();
	auto*     h_val_ofs_arr    = new uint32_t[n_ofs]();
	auto*     h_len_ofs_arr    = new uint32_t[n_ofs]();
	auto*     h_copy_data      = new uint32_t[n_tup]();
	const int num_threads      = 128;
	const int items_per_thread = 4;
	int*      d_decoded_arr    = nullptr;
	size_t    dg               = (n_tup + tile_size - 1) / tile_size;
	size_t    db               = num_threads;
	auto*     h_decoded_arr    = new uint32_t[n_tup];
	int       num_trials       = 0;
	cudaMalloc((void**)&d_decoded_arr, n_tup * sizeof(uint32_t));

	std::cout << "------------------------------------ \n";
	std::cout << "-- Generate :  \n";
	std::cout << "-- delta " << tile_based::delta << '\n';
	SHOW(n_tup)

	/* generate 0, 5, 10. */
	for (size_t idx = 0, run_idx = 0; idx < n_tup; ++idx) {
		for (size_t i {0}; i < 8; ++i, idx++) {
			h_org_arr[idx] = 200;
		}
		h_val_arr[run_idx] = 200;
		h_len_arr[run_idx] = 8;
		run_idx            = run_idx + 1;

		for (size_t i {0}; i < 8; ++i, idx++) {
			h_org_arr[idx] = 300;
		}
		h_val_arr[run_idx] = 300;
		h_len_arr[run_idx] = 8;
		run_idx            = run_idx + 1;
	}

	std::cout << "------------------------------------ \n";
	std::cout << "-- Encode : \n";

	/* Data needs to be copied. the encoding change the original data. */
	memcpy(h_copy_data, h_org_arr, n_tup * sizeof(int));

	// extend with the last value to make it multiple of 128
	debug::pretty::print_table<uint32_t, 128, 1>(h_copy_data);

	auto pair = rleBinPack(h_copy_data, h_val_arr, h_len_arr, h_val_ofs_arr, h_len_ofs_arr, n_tup);
	SHOW(pair.first)
	SHOW(pair.second)

	debug::pretty::print_table<uint32_t, 128, 1>(h_org_arr);
	debug::pretty::print_table<uint32_t, 128, 1>(h_copy_data);
	debug::pretty::print_table<uint32_t, 128, 1>(h_val_arr);
	debug::pretty::print_table<uint32_t, 128, 1>(h_len_arr);

	std::cout << "------------------------------------ \n";
	std::cout << "-- Load encoded data into GPU : \n";

	uint* d_val_ofs_arr = tile_based::loadColumnToGPU<uint>(h_val_ofs_arr, n_ofs);
	uint* d_val_arr     = tile_based::loadColumnToGPU<uint>(h_val_arr, n_tup);
	uint* d_len_ofs_arr = tile_based::loadColumnToGPU<uint>(h_len_ofs_arr, n_ofs);
	uint* d_len_arr     = tile_based::loadColumnToGPU<uint>(h_len_arr, n_tup);

	cudaDeviceSynchronize();
	std::cout << "------------------------------------ \n";
	std::cout << "-- Decode :  \n";

	runRBinKernel<num_threads, items_per_thread>
	    <<<dg, db, 4096>>>(d_decoded_arr, d_val_ofs_arr, d_val_arr, d_len_ofs_arr, d_len_arr, n_tup);

	std::cout << "------------------------------------ \n";
	std::cout << "-- Copy data to host :  \n";

	cudaDeviceSynchronize();

	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(int) * n_tup, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	std::cout << "------------------------------------ \n";
	std::cout << "-- Test :  \n";
	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << "ERROR:" << i << " " << h_org_arr[i] << " " << h_decoded_arr[i] << '\n';
			return -1;
		}
	}

	std::cout << "-- Inputs match ! " << '\n';
	std::cout << "------------------------------------ \n";
	std::cout << "-- Bench :  \n";

	// Run trials
	for (int t = 0; t < num_trials; t++) {
		// Kernel timing
		float query_time;
		SETUP_TIMING();

		cudaEventRecord(start, nullptr);
		runRBinKernel<num_threads, items_per_thread>
		    <<<dg, db, 4096>>>(d_decoded_arr, d_val_ofs_arr, d_val_arr, d_len_ofs_arr, d_val_arr, n_tup);
		cudaEventRecord(stop, nullptr);

		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&query_time, start, stop);

		CubDebugExit(cudaPeekAtLastError());
		CubDebugExit(cudaDeviceSynchronize());

		std::cout << "-- Query-time: " << std::to_string(t) << " : " << query_time << " ms " << '\n';
		std::cout << "-- Effective-memory-bandwidth: " << std::to_string(t) << " : "
		          << fastlanes::gpu::helper::BWEffective(3 * n_tup / 8, n_tup * 4, query_time) << " GB/s" << '\n';
		cudaDeviceSynchronize();
	}

	return 2;
}