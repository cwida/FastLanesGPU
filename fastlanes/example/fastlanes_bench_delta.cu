#include "fastlanes.cuh"
#include "debug.hpp"
#include "fls_gen/pack/pack.hpp"
#include "fls_gen/rsum/rsum.cuh"
#include "fls_gen/transpose/transpose.hpp"
#include "fls_gen/unrsum/unrsum.hpp"
#include <cstring>

__global__  void bfr_3bw_32ow_32crw_1uf_krl_v0(uint32_t* in, uint32_t* out, uint32_t* base) {
	uint32_t trd_idx = threadIdx.x;
	uint32_t blc_idx = blockIdx.x;
	in               = in + ((blc_idx * 3) << 5);
	out              = out + (blc_idx << 10);
	trd_idx          = trd_idx % 32;
	uint32_t r_0;
	uint32_t r_1;

	__shared__ uint32_t sm_arr[1024];

	r_0                        = *(in + (0 * 32) + (trd_idx * 1) + 0);
	r_1                        = (r_0) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (0 * 32)] = r_1;
	r_1                        = (r_0 >> 3) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (1 * 32)] = r_1;
	r_1                        = (r_0 >> 6) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (2 * 32)] = r_1;
	r_1                        = (r_0 >> 9) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (3 * 32)] = r_1;
	r_1                        = (r_0 >> 12) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (4 * 32)] = r_1;
	r_1                        = (r_0 >> 15) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (5 * 32)] = r_1;
	r_1                        = (r_0 >> 18) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (6 * 32)] = r_1;
	r_1                        = (r_0 >> 21) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (7 * 32)] = r_1;
	r_1                        = (r_0 >> 24) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (8 * 32)] = r_1;
	r_1                        = (r_0 >> 27) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (9 * 32)] = r_1;
	r_1                        = (r_0 >> 30) & ((1ULL << 2) - 1);
	r_0                        = *(in + (0 * 32) + (trd_idx * 1) + 32);
	r_1 |= ((r_0) & ((1ULL << 1) - 1)) << 2;
	sm_arr[trd_idx + (10 * 32)] = r_1;
	r_1                         = (r_0 >> 1) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (11 * 32)] = r_1;
	r_1                         = (r_0 >> 4) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (12 * 32)] = r_1;
	r_1                         = (r_0 >> 7) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (13 * 32)] = r_1;
	r_1                         = (r_0 >> 10) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (14 * 32)] = r_1;
	r_1                         = (r_0 >> 13) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (15 * 32)] = r_1;
	r_1                         = (r_0 >> 16) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (16 * 32)] = r_1;
	r_1                         = (r_0 >> 19) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (17 * 32)] = r_1;
	r_1                         = (r_0 >> 22) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (18 * 32)] = r_1;
	r_1                         = (r_0 >> 25) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (19 * 32)] = r_1;
	r_1                         = (r_0 >> 28) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (20 * 32)] = r_1;
	r_1                         = (r_0 >> 31) & ((1ULL << 1) - 1);
	r_0                         = *(in + (0 * 32) + (trd_idx * 1) + 64);
	r_1 |= ((r_0) & ((1ULL << 2) - 1)) << 1;
	sm_arr[trd_idx + (21 * 32)] = r_1;
	r_1                         = (r_0 >> 2) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (22 * 32)] = r_1;
	r_1                         = (r_0 >> 5) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (23 * 32)] = r_1;
	r_1                         = (r_0 >> 8) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (24 * 32)] = r_1;
	r_1                         = (r_0 >> 11) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (25 * 32)] = r_1;
	r_1                         = (r_0 >> 14) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (26 * 32)] = r_1;
	r_1                         = (r_0 >> 17) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (27 * 32)] = r_1;
	r_1                         = (r_0 >> 20) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (28 * 32)] = r_1;
	r_1                         = (r_0 >> 23) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (29 * 32)] = r_1;
	r_1                         = (r_0 >> 26) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (30 * 32)] = r_1;
	r_1                         = (r_0 >> 29) & ((1ULL << 3) - 1);
	sm_arr[trd_idx + (31 * 32)] = r_1;

	d_rsum_32(sm_arr, out, base);
}

int main() {

	/* Init */
	std::cout << "------------------------------------ \n";
	std::cout << "-- Init :  \n";
	cudaDeviceSynchronize();

	const uint64_t warp_sz          = 32;
	// const uint64_t num_trials       = 5;
	const uint64_t n_vec            = 256 * 1024;
	const uint64_t vec_sz           = 1024;
	const uint64_t n_tup            = vec_sz * n_vec;
	const uint64_t v_blc_sz         = 1;
	const uint64_t n_blc            = n_vec / v_blc_sz;
	const uint64_t n_trd            = v_blc_sz * warp_sz;
	auto*          h_org_arr        = new uint32_t[n_tup];
	auto*          h_encoded_data   = new uint32_t[n_tup];
	auto*          h_decoded_arr    = new uint32_t[n_tup];
	auto*          h_transposed_arr = new uint32_t[vec_sz];
	auto*          h_unrsummed_arr  = new uint32_t[vec_sz];
	auto*          h_base_arr       = new uint32_t[32 * n_vec];
	uint64_t       encoded_arr_bsz  = n_tup * sizeof(int);
	uint32_t*      d_base_arr       = nullptr;
	uint32_t*      d_decoded_arr    = nullptr;
	uint32_t*      d_encoded_arr    = nullptr;
	uint8_t        num_bits         = 3;

	CUDA_SAFE_CALL(cudaMalloc((void**)&d_decoded_arr, sizeof(uint32_t) * n_tup));

	static_assert(n_tup % n_trd == 0, "");
	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';
	std::cout << "------------------------------------ \n";
	std::cout << "-- Generate : \n";
	FLS_SHOW(n_vec)
	FLS_SHOW(n_tup)

	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		//		h_org_arr[i] = i * tile_based::delta;
		h_org_arr[i] = i % 1024;
	}

	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';
	std::cout << "------------------------------------ \n";
	std::cout << "-- Encode :  \n";

	auto in_als   = h_org_arr;
	auto out_als  = h_encoded_data;
	auto base_als = h_base_arr;

	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::transpose::fallback::scalar::transpose_i(in_als, h_transposed_arr);

		generated::unrsum::fallback::scalar::unrsum(h_transposed_arr, h_unrsummed_arr);

		std::memcpy(base_als, h_transposed_arr, sizeof(uint32_t) * 32);

		generated::pack::fallback::scalar::pack(h_unrsummed_arr, out_als, num_bits);

		in_als   = in_als + vec_sz;
		out_als  = out_als + (num_bits * vec_sz / 32);
		base_als = base_als + 32;
	}

	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';
	std::cout << "------------------------------------ \n";
	std::cout << "-- Load encoded data into GPU : \n";

	d_encoded_arr = fastlanes::gpu::load_arr(h_encoded_data, encoded_arr_bsz);
	d_base_arr    = fastlanes::gpu::load_arr(h_base_arr, 32 * n_vec * sizeof(uint32_t));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';
	std::cout << "------------------------------------ \n";
	std::cout << "-- Decode : \n";

	bfr_3bw_32ow_32crw_1uf_krl_v0<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';
	std::cout << "------------------------------------ \n";
	std::cout << "-- Copy data to host :  \n";

	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';
	std::cout << "------------------------------------ \n";
	std::cout << "-- Test :  \n";

	for (uint64_t vec_idx = 0, idx = 0; vec_idx < n_vec; ++vec_idx) {
		for (; idx < n_tup; idx++) {
			if (h_transposed_arr[idx % 1024] != h_decoded_arr[idx]) {
				std::cout << fastlanes::debug::red << "-- ERROR: idx | " << idx << " : " << h_org_arr[idx]
				          << " != " << h_decoded_arr[idx] << fastlanes::debug::def << '\n';
				return -1;
			}
		}
	}

	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';
}
