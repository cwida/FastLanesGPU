#include "config.hpp"
#include "cub/util_debug.cuh"
#include "tile_based/kernel.cuh"
#include <cuda_profiler_api.h>

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

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void run_bin_kernel(int* col, uint* col_block_start, uint* col_data) {

	int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
	int tile_idx  = blockIdx.x;

	// Load a segment of consecutive items that are blocked across threads
	int col_block[ITEMS_PER_THREAD];

	extern __shared__ uint shared_buffer[];

	load_bin_pack<BLOCK_THREADS, ITEMS_PER_THREAD>(col_block_start, col_data, shared_buffer, col_block);

	// write unpacked values directly to global memory
	for (int i = 0; i < ITEMS_PER_THREAD; i++) {
		col[tile_size * tile_idx + i * 128 + threadIdx.x] = col_block[i];
	}
}

namespace tile_based {
template <typename T>
T* loadColumnToGPU(T* src, int len) {
	T* dest = nullptr;
	cudaMalloc((void**)&dest, sizeof(T) * len);
	CubDebugExit(cudaMemcpy(dest, src, sizeof(T) * len, cudaMemcpyHostToDevice));
	return dest;
}

} // namespace tile_based

int main(int argc, char** argv) {

	cudaSetDevice(0);
	int bitwidth = 3;

	if (argc > 1) { bitwidth = atoi(argv[1]); }

	std::cout << "Bitwidth set to " << bitwidth << std::endl;

	uint64_t tup_c         = 1 << 28;
	auto*    original_data = new uint32_t[tup_c];
	uint32_t mask          = (1 << bitwidth) - 1;

	/* generate random numbers. */
	for (int i = 0; i < tup_c; i++) {
		original_data[i] = rand() & mask;
	}

	int      block_size      = 128;
	int      elem_per_thread = 4;
	int      tile_size       = block_size * elem_per_thread;
	int      num_blocks      = tup_c / block_size;
	auto*    encoded_data    = new uint32_t[tup_c]();
	uint64_t ofs_c           = num_blocks + 1;
	auto*    ofs_arr         = new uint32_t[ofs_c]();
	auto*    copy_data       = new uint32_t[tup_c];

	/* Data needs to be copied. the encoding change the original data. */
	memcpy(copy_data, original_data, tup_c * sizeof(int));

	// extend with the last value to make it multiple of 128
	uint32_t encoded_data_bsz = bin_pack(copy_data, encoded_data, ofs_arr, tup_c);

	tile_based::encoded_column h_col {ofs_arr, encoded_data, tup_c * 4};

	uint* d_col_block_start = tile_based::loadColumnToGPU<uint>(h_col.block_start, num_blocks + 1);
	uint* d_col_data        = tile_based::loadColumnToGPU<uint>(h_col.data, h_col.data_size / 4);

	tile_based::encoded_column d_col {d_col_block_start, d_col_data};

	cudaDeviceSynchronize();

	const int num_threads      = 128;
	const int items_per_thread = 4;
	int*      col              = nullptr;
	cudaMalloc((void**)&col, tup_c * sizeof(int));
	size_t Dg = (tup_c + tile_size - 1) / tile_size;
	size_t Db = num_threads;
	size_t Ns = 3000;

	run_bin_kernel<num_threads, items_per_thread><<<Dg, Db, Ns>>>(col, d_col.block_start, d_col.data);

	int* temp = new int[tup_c];
	CubDebugExit(cudaMemcpy(temp, col, sizeof(int) * tup_c, cudaMemcpyDeviceToHost));

	for (int i = 0; i < tup_c; i++) {
		if (original_data[i] != temp[i]) {
			std::cout << "ERROR:" << i << " " << original_data[i] << " " << temp[i] << '\n';
			return -1;
		}
	}
	std::cout << "-- Inputs match! " << '\n';
}