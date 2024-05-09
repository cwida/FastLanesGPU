#include "config.hpp"
#include "cub/util_debug.cuh"
#include "kernel.cuh"
#include "binpack_kernel.cuh"
#include <cuda.h>
#include <cuda_profiler_api.h>

uint deltaBinPack(int*& in, int*& out, uint*& block_offsets, uint num_entries) {
	uint offset = 0;

	uint block_size      = 128;
	uint elem_per_thread = 4;
	uint tile_size       = block_size * elem_per_thread;

	uint miniblock_count = 4;
	uint total_count     = num_entries;
	uint first_val       = in[0];

	out[0] = block_size;
	out[1] = miniblock_count;
	out[2] = total_count;
	out[3] = first_val;

	offset += 4;

	for (uint tile_start = 0; tile_start < num_entries; tile_start += tile_size) {
		uint block_index   = tile_start / block_size;
		int  tmp_first_val = in[0];

		out[offset] = tmp_first_val;
		offset++;

		// Compute the deltas
		for (int i = tile_size - 1; i > 0; i--) {
			in[i] = in[i] - in[i - 1];
		}
		in[0] = 0;

		for (int block_start = 0; block_start < block_size * 4; block_start += block_size, block_index += 1) {
			block_offsets[block_index] = offset;

			// For FOR - Find min val
			int min_val = in[0];
			for (int i = 1; i < block_size; i++) {
				if (in[i] < min_val) { min_val = in[i]; }
			}

			min_val = 0; /* HACK */
			for (int i = 0; i < block_size; i++) {
				in[i] = in[i] - min_val;
			}

			out[offset] = min_val;
			offset++;

			// Subtracting min_val ensures that all input vals are >= 0
			// Going forward in and out will both be treated as unsigned integers.
			uint* inp  = (uint*)in;
			uint* outp = (uint*)out;

			uint  miniblock_size      = block_size / miniblock_count;
			uint* miniblock_bitwidths = new uint[miniblock_count];
			for (int i = 0; i < miniblock_count; i++) {
				miniblock_bitwidths[i] = 0;
			}

			for (uint miniblock = 0; miniblock < miniblock_count; miniblock++) {
				for (uint i = 0; i < miniblock_size; i++) {
					uint bitwidth = uint(ceil(log2(inp[miniblock * miniblock_size + i] + 1)));
					if (bitwidth > miniblock_bitwidths[miniblock]) { miniblock_bitwidths[miniblock] = bitwidth; }
				}
			}

			// Extra for Simple BinPack
			uint max_bitwidth = miniblock_bitwidths[0];
			for (int i = 1; i < miniblock_count; i++) {
				max_bitwidth = max(max_bitwidth, miniblock_bitwidths[i]);
			}
			for (int i = 0; i < miniblock_count; i++) {
				miniblock_bitwidths[i] = max_bitwidth;
			}
			outp[offset] = miniblock_bitwidths[0] + (miniblock_bitwidths[1] << 8) + (miniblock_bitwidths[2] << 16) +
			               (miniblock_bitwidths[3] << 24);
			offset++;

			for (int miniblock = 0; miniblock < miniblock_count; miniblock++) {
				uint bitwidth = miniblock_bitwidths[miniblock];
				uint shift    = 0;
				for (int i = 0; i < miniblock_size; i++) {
					if (shift + bitwidth > 32) {
						if (shift != 32) { outp[offset] += inp[miniblock * miniblock_size + i] << shift; }
						offset++;
						shift        = (shift + bitwidth) & (32 - 1);
						outp[offset] = inp[miniblock * miniblock_size + i] >> (bitwidth - shift);
					} else {
						outp[offset] += inp[miniblock * miniblock_size + i] << shift;
						shift += bitwidth;
					}
				}
				offset++;
			}

			// Increment the input pointer by block size
			in += block_size;
		}
	}

	block_offsets[num_entries / block_size] = offset;

	return offset;
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void runDBinKernel(int* col, uint* col_block_start, uint* col_data, int num_entries) {
	int tile_size   = BLOCK_THREADS * ITEMS_PER_THREAD;
	int tile_idx    = blockIdx.x;
	int tile_offset = tile_idx * tile_size;

	// Load a segment of consecutive items that are blocked across threads
	int col_block[ITEMS_PER_THREAD];

	int  num_tiles      = (num_entries + tile_size - 1) / tile_size;
	int  num_tile_items = tile_size;
	bool is_last_tile   = false;
	if (tile_idx == num_tiles - 1) {
		num_tile_items = num_entries - tile_offset;
		is_last_tile   = true;
	}

	extern __shared__ uint shared_buffer[];
	LoadDBinPack<BLOCK_THREADS, ITEMS_PER_THREAD>(
	    col_block_start, col_data, shared_buffer, col_block, is_last_tile, num_tile_items);

	__syncthreads();

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

}

int main() {

	cudaSetDevice(0);

	/* Init */
	std::cout << "------------------------------------ \n";
	std::cout << "-- Init :  \n";

	uint64_t  n_tup            = 1 << 28;
	auto*     h_org_arr        = new uint32_t[n_tup];
	int       block_size       = 128;
	int       elem_per_thread  = 4;
	int       tile_size        = block_size * elem_per_thread;
	int       num_blocks       = n_tup / block_size;
	auto*     encoded_data     = new int[n_tup]();
	uint64_t  ofs_c            = num_blocks + 1;
	auto*     ofs_arr          = new uint[ofs_c]();
	auto*     copy_data        = new int[n_tup];
	const int num_threads      = 128;
	const int items_per_thread = 4;
	int*      col              = nullptr;
	size_t    dg               = (n_tup + tile_size - 1) / tile_size;
	size_t    db               = num_threads;
	size_t    ns               = 3000;
	int*      temp             = new int[n_tup];
	// int       num_trials       = 10;
	cudaMalloc((void**)&col, n_tup * sizeof(int));

	std::cout << "------------------------------------ \n";
	std::cout << "-- Generate :  \n";
	std::cout << "-- delta " << tile_based::delta << '\n';

	/* generate 0, 5, 10. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = i * tile_based::delta;
	}

	std::cout << "------------------------------------ \n";
	std::cout << "-- Encode : \n";

	/* Data needs to be copied. the encoding change the original data. */
	memcpy(copy_data, h_org_arr, n_tup * sizeof(int));

	// extend with the last value to make it multiple of 128
	deltaBinPack(copy_data, encoded_data, ofs_arr, n_tup);

	std::cout << "------------------------------------ \n";
	std::cout << "-- Load encoded data into GPU : \n";

	tile_based::encoded_column h_col {ofs_arr, reinterpret_cast<uint*>(encoded_data), n_tup * 4};
	uint*                      d_col_block_start = tile_based::loadColumnToGPU<uint>(h_col.block_start, num_blocks + 1);
	uint*                      d_col_data        = tile_based::loadColumnToGPU<uint>(h_col.data, h_col.data_size / 4);

	tile_based::encoded_column d_col {d_col_block_start, d_col_data};

	cudaDeviceSynchronize();
	std::cout << "------------------------------------ \n";
	std::cout << "-- Decode :  \n";

	runDBinKernel<num_threads, items_per_thread><<<dg, db, ns>>>(col, d_col.block_start, d_col.data, n_tup);

	std::cout << "------------------------------------ \n";
	std::cout << "-- Copy data to host :  \n";

	CubDebugExit(cudaMemcpy(temp, col, sizeof(int) * n_tup, cudaMemcpyDeviceToHost));

	std::cout << "------------------------------------ \n";
	std::cout << "-- Test :  \n";
	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != temp[i]) {
			std::cout << "ERROR:" << i << " " << h_org_arr[i] << " " << temp[i] << '\n';
			return -1;
		}
	}

	std::cout << "-- Inputs match ! " << '\n';
	std::cout << "------------------------------------ \n";
	std::cout << "-- Bench :  \n";

#if 0
	// Run trials
	for (int t = 0; t < num_trials; t++) {
		// Kernel timing
		float query_time;
		SETUP_TIMING();

		cudaEventRecord(start, nullptr);
		runDBinKernel<num_threads, items_per_thread><<<dg, db, ns>>>(col, d_col.block_start, d_col.data, n_tup);
		cudaEventRecord(stop, nullptr);

		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&query_time, start, stop);

		CubDebugExit(cudaPeekAtLastError());
		CubDebugExit(cudaDeviceSynchronize());

		std::cout << "-- Query-time: " << std::to_string(t) << " : " << query_time << " ms " << '\n';
		cudaDeviceSynchronize();
	}

	return 2;
#endif
}