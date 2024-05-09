// #include "config.hpp"
// #include "crystal/crystal.cuh"
// #include "cub/util_debug.cuh"
// #include "kernel.cuh"
// #include <cuda_profiler_api.h>
// #include <gpu_utils.h>
//
// using namespace std;
// using namespace fastlanes::gpu;
//
// struct QueryMtd {
// 	fastlanes::n_t     n_vec;
// 	uint8_t            bw;
// 	fastlanes::n_t     n_tup;
// 	unsigned long long result;
// };
//
// uint32_t bin_pack(uint32_t*& in, uint32_t*& out, uint32_t*& block_offsets, uint32_t tup_c) {
// 	uint32_t out_ofs = 0;
//
// 	uint32_t block_size      = 128;
// 	uint32_t miniblock_count = 4;
// 	uint32_t miniblock_size  = block_size / miniblock_count;
// 	uint32_t total_count     = tup_c;
// 	uint32_t first_val       = in[0];
//
// 	out[0] = block_size;
// 	out[1] = miniblock_count;
// 	out[2] = total_count;
// 	out[3] = first_val;
//
// 	out_ofs += 4;
//
// 	for (uint32_t idx = 0; idx < tup_c; idx += block_size) {
// 		uint32_t blk_idx       = idx / block_size;
// 		block_offsets[blk_idx] = out_ofs;
//
// 		// Find min val
// 		uint32_t min_val = in[0];
// 		for (int i = 1; i < block_size; i++) {
// 			if (in[i] < min_val) { min_val = in[i]; }
// 		}
//
// 		for (int i = 0; i < block_size; i++) {
// 			in[i] = in[i] - min_val;
// 		}
//
// 		uint32_t* miniblock_bitwidths = new uint32_t[miniblock_count];
// 		for (int i = 0; i < miniblock_count; i++) {
// 			miniblock_bitwidths[i] = 0;
// 		}
//
// 		for (uint32_t miniblock = 0; miniblock < miniblock_count; miniblock++) {
// 			for (uint32_t i = 0; i < miniblock_size; i++) {
// 				uint32_t bitwidth = uint32_t(ceil(log2(in[miniblock * miniblock_size + i] + 1)));
// 				if (bitwidth > miniblock_bitwidths[miniblock]) { miniblock_bitwidths[miniblock] = bitwidth; }
// 			}
// 		}
//
// 		// Extra for Simple BinPack
// 		uint32_t max_bitwidth = miniblock_bitwidths[0];
// 		for (int i = 1; i < miniblock_count; i++) {
// 			max_bitwidth = std::max(max_bitwidth, miniblock_bitwidths[i]);
// 		}
// 		for (int i = 0; i < miniblock_count; i++) {
// 			miniblock_bitwidths[i] = max_bitwidth;
// 		}
//
// 		out[out_ofs] = min_val;
// 		out_ofs++;
//
// 		out[out_ofs] = miniblock_bitwidths[0] + (miniblock_bitwidths[1] << 8) + (miniblock_bitwidths[2] << 16) +
// 		               (miniblock_bitwidths[3] << 24);
// 		out_ofs++;
//
// 		for (int miniblock = 0; miniblock < miniblock_count; miniblock++) {
// 			uint32_t bitwidth = miniblock_bitwidths[miniblock];
// 			uint32_t shift    = 0;
// 			for (int i = 0; i < miniblock_size; i++) {
// 				if (shift + bitwidth > 32) {
// 					if (shift != 32) { out[out_ofs] += in[miniblock * miniblock_size + i] << shift; }
// 					out_ofs++;
// 					shift        = (shift + bitwidth) & (32 - 1);
// 					out[out_ofs] = in[miniblock * miniblock_size + i] >> (bitwidth - shift);
// 				} else {
// 					out[out_ofs] += in[miniblock * miniblock_size + i] << shift;
// 					shift += bitwidth;
// 				}
// 			}
// 			out_ofs++;
// 		}
//
// 		// Increment the input pointer by block size
// 		in += block_size;
// 	}
//
// 	block_offsets[tup_c / block_size] = out_ofs;
//
// 	return out_ofs;
// }
//
// template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
// __global__ void run_bin_kernel(uint* col_block_start, uint* col_data, unsigned long long* revenue) {
// 	uint32_t               items[ITEMS_PER_THREAD];
// 	extern __shared__ uint shared_buffer[];
//
// 	unsigned long long sum = 0;
// 	LoadBinPack<BLOCK_THREADS, ITEMS_PER_THREAD>(col_block_start, col_data, shared_buffer, items);
//
// #pragma unroll
// 	for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
// 		sum += items[ITEM];
// 	}
//
// 	__syncthreads();
//
// 	static __shared__ long long buffer[32];
// 	unsigned long long aggregate = BlockSum<long long, BLOCK_THREADS, ITEMS_PER_THREAD>(sum, (long long*)buffer);
// 	__syncthreads();
//
// 	if (threadIdx.x == 0) { atomicAdd(revenue, aggregate); }
// }
//
// template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
// float query_aggregate(uint*                        col_block_start,
//                       uint*                        col_data,
//                       QueryMtd                     hardcoded,
//                       cub::CachingDeviceAllocator& g_allocator) {
// 	int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
//
// 	SETUP_TIMING();
// 	float                                     time_query;
// 	chrono::high_resolution_clock::time_point st, finish;
// 	st = chrono::high_resolution_clock::now();
// 	cudaEventRecord(start, 0);
// 	unsigned long long* d_sum = NULL;
// 	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_sum, sizeof(long long)));
//
// 	cudaMemset(d_sum, 0, sizeof(long long));
//
// 	// Run
// 	run_bin_kernel<BLOCK_THREADS, ITEMS_PER_THREAD>
// 	    <<<hardcoded.n_vec, BLOCK_THREADS, 3000>>>(col_block_start, col_data, d_sum);
//
// 	cudaEventRecord(stop, 0);
// 	cudaEventSynchronize(stop);
// 	cudaEventElapsedTime(&time_query, start, stop);
//
// 	unsigned long long revenue;
// 	CubDebugExit(cudaMemcpy(&revenue, d_sum, sizeof(uint64_t), cudaMemcpyDeviceToHost));
//
// 	finish                             = chrono::high_resolution_clock::now();
// 	std::chrono::duration<double> diff = finish - st;
//
// 	double total_time_taken {diff.count() * 1000};
// 	FLS_SHOW(total_time_taken)
//
// 	/*Check the result*/
// 	FLS_SHOW(revenue)
// 	if (revenue != hardcoded.result) { throw std::runtime_error("RESULT INCOREECT!"); }
//
// 	CLEANUP(d_sum);
//
// 	return time_query;
// }

int main() {

	// cudaSetDevice(0);
	// constexpr uint64_t n_tile        = 2 * 256 * 1024;
	// constexpr uint64_t tile_sz       = 512;
	// constexpr uint64_t n_tup         = n_tile * tile_sz;
	// constexpr uint64_t block_sz      = 128;
	// int                n_block       = n_tup / block_sz;
	// uint64_t           n_ofc         = n_block + 1;
	// auto*              original_data = new uint32_t[n_tup];
	// auto*              encoded_data  = new uint32_t[n_tup];
	// auto*              copy_data     = new uint32_t[n_tup];
	// auto*              ofs_arr       = new uint32_t[n_ofc];
	// for (uint8_t bitwidth {0}; bitwidth < 33; bitwidth++) {
	//
	// 	uint32_t mask = (1 << bitwidth) - 1;
	// 	uint32_t bw   = bitwidth;
	//
	// 	FLS_SHOW(bw)
	// 	uint64_t sum {0};
	// 	/* generate random numbers. */
	// 	for (int i = 0; i < n_tup; i++) {
	// 		original_data[i] = i & mask;
	// 		sum += original_data[i];
	// 	}
	// 	FLS_SHOW(sum)
	//
	//
	// 	/* Data needs to be copied. the encoding change the original data. */
	// 	memcpy(copy_data, original_data, n_tup * sizeof(int));
	//
	// 	// extend with the last value to make it multiple of 128
	//
	// 	auto copy_data_als = copy_data;
	// 	auto size = bin_pack(copy_data_als, encoded_data, ofs_arr, n_tup);
	// 	double real_bw = size * 32 / double(n_tup);
	// 	FLS_SHOW(real_bw)
	//
	// 	uint* d_col_block_start = load_to_gpu<uint>(ofs_arr, (n_block + 1) * 4, g_allocator);
	// 	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	// 	uint* d_col_data = load_to_gpu<uint>(encoded_data, n_tup * 4, g_allocator);
	// 	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	//
	// 	QueryMtd query_mtd {n_tile, bitwidth, n_tup, sum};
	// 	for (int i {0}; i < 1; ++i) {
	// 		auto time = query_aggregate<128, 4>(d_col_block_start, d_col_data, query_mtd, g_allocator);
	// 		FLS_SHOW(time)
	// 	}
	//
	// 	CLEANUP(d_col_block_start)
	// 	CLEANUP(d_col_data)
	//
	// }
	//
	// delete original_data;
	// delete encoded_data;
	// delete copy_data;
	// delete ofs_arr;
}