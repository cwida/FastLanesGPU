#include "fastlanes.cuh"
#include "debug.hpp"
#include "fls_gen/pack/pack.hpp"
#include "fls_gen/unpack/unpack.cuh"
#include <iostream>

int main() {
	auto bitwidth = 13;

	/* Init */
	std::cout << "------------------------------------ \n";
	std::cout << "-- Init :  \n";
	cudaDeviceSynchronize();

	const uint64_t warp_sz         = 32;
	const uint64_t n_vec           = 256 * 1024;
	const uint64_t vec_sz          = 1024;
	const uint64_t n_tup           = vec_sz * n_vec;
	const uint64_t v_blc_sz        = 1;
	const uint64_t n_blc           = n_vec / v_blc_sz;
	const uint64_t n_trd           = v_blc_sz * warp_sz;
	auto*          h_org_arr       = new uint32_t[n_tup];
	auto*          h_encoded_data  = new uint32_t[n_tup];
	uint64_t       encoded_arr_bsz = n_tup * sizeof(int);
	uint32_t*      d_decoded_arr   = nullptr;
	auto*          h_decoded_arr   = new uint32_t[n_tup];
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_decoded_arr, sizeof(uint32_t) * n_tup));
	uint32_t mask = (1 << bitwidth) - 1;

	static_assert(n_tup % n_trd == 0, "");
	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';
	std::cout << "------------------------------------ \n";
	std::cout << "-- Generate : \n";

	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = 5 & mask;
	}

	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';
	std::cout << "------------------------------------ \n";
	std::cout << "-- Encode :  \n";

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';
	std::cout << "------------------------------------ \n";
	std::cout << "-- Load encoded data into GPU : \n";

	auto* d_encoded_arr = fastlanes::gpu::load_arr(h_encoded_data, encoded_arr_bsz);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';
	std::cout << "------------------------------------ \n";
	std::cout << "-- Decode : \n";

	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';
	std::cout << "------------------------------------ \n";
	std::cout << "-- Copy data to host :  \n";

	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';
	std::cout << "------------------------------------ \n";
	std::cout << "-- Test :  \n";

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << fastlanes::debug::red << "-- ERROR: idx | " << i << " : " << h_org_arr[i]
			          << " != " << h_decoded_arr[i] << fastlanes::debug::def << '\n';
			return -1;
		}
	}
	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';

}
