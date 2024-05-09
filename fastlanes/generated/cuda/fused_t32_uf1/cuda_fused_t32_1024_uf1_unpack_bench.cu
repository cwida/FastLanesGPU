// generated!

#include "fastlanes.cuh"
#include "fls_gen/pack/pack.hpp"
#include "fls_gen/unpack/unpack_fused.cuh"
#include <iostream>

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

static void bench0_unpack_0bw_32ow_32crw_1uf() {
	auto bitwidth = 0;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench1_unpack_1bw_32ow_32crw_1uf() {
	auto bitwidth = 1;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench2_unpack_2bw_32ow_32crw_1uf() {
	auto bitwidth = 2;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench3_unpack_3bw_32ow_32crw_1uf() {
	auto bitwidth = 3;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench4_unpack_4bw_32ow_32crw_1uf() {
	auto bitwidth = 4;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench5_unpack_5bw_32ow_32crw_1uf() {
	auto bitwidth = 5;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench6_unpack_6bw_32ow_32crw_1uf() {
	auto bitwidth = 6;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench7_unpack_7bw_32ow_32crw_1uf() {
	auto bitwidth = 7;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench8_unpack_8bw_32ow_32crw_1uf() {
	auto bitwidth = 8;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench9_unpack_9bw_32ow_32crw_1uf() {
	auto bitwidth = 9;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench10_unpack_10bw_32ow_32crw_1uf() {
	auto bitwidth = 10;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench11_unpack_11bw_32ow_32crw_1uf() {
	auto bitwidth = 11;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench12_unpack_12bw_32ow_32crw_1uf() {
	auto bitwidth = 12;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench13_unpack_13bw_32ow_32crw_1uf() {
	auto bitwidth = 13;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench14_unpack_14bw_32ow_32crw_1uf() {
	auto bitwidth = 14;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench15_unpack_15bw_32ow_32crw_1uf() {
	auto bitwidth = 15;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench16_unpack_16bw_32ow_32crw_1uf() {
	auto bitwidth = 16;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench17_unpack_17bw_32ow_32crw_1uf() {
	auto bitwidth = 17;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench18_unpack_18bw_32ow_32crw_1uf() {
	auto bitwidth = 18;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench19_unpack_19bw_32ow_32crw_1uf() {
	auto bitwidth = 19;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench20_unpack_20bw_32ow_32crw_1uf() {
	auto bitwidth = 20;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench21_unpack_21bw_32ow_32crw_1uf() {
	auto bitwidth = 21;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench22_unpack_22bw_32ow_32crw_1uf() {
	auto bitwidth = 22;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench23_unpack_23bw_32ow_32crw_1uf() {
	auto bitwidth = 23;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench24_unpack_24bw_32ow_32crw_1uf() {
	auto bitwidth = 24;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench25_unpack_25bw_32ow_32crw_1uf() {
	auto bitwidth = 25;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench26_unpack_26bw_32ow_32crw_1uf() {
	auto bitwidth = 26;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench27_unpack_27bw_32ow_32crw_1uf() {
	auto bitwidth = 27;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench28_unpack_28bw_32ow_32crw_1uf() {
	auto bitwidth = 28;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench29_unpack_29bw_32ow_32crw_1uf() {
	auto bitwidth = 29;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench30_unpack_30bw_32ow_32crw_1uf() {
	auto bitwidth = 30;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench31_unpack_31bw_32ow_32crw_1uf() {
	auto bitwidth = 31;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
static void bench32_unpack_32bw_32ow_32crw_1uf() {
	auto bitwidth = 32;
	if (bitwidth == 32) { bitwidth = 31; };
	/* generate random numbers. */
	for (int i = 0; i < n_tup; i++) {
		h_org_arr[i] = rand() % (1 << bitwidth);
	}

	auto in  = h_org_arr;
	auto out = h_encoded_data;
	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(in, out, bitwidth);
		in  = in + vec_sz;
		out = out + (bitwidth * vec_sz / 32);
	}

	auto* d_encoded_arr = fastlanes::gpu::load_to_gpu(h_encoded_data, encoded_arr_bsz, fastlanes::gpu::g_allocator);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, bitwidth);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != h_decoded_arr[i]) {
			std::cout << bitwidth << " failed!" << std::endl;
			return;
		}
	}
	std::cout << bitwidth << " succes!" << std::endl;

	CLEANUP(d_encoded_arr);
}
void benchmark_all() {
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_decoded_arr, sizeof(uint32_t) * n_tup));
	bench0_unpack_0bw_32ow_32crw_1uf();
	bench1_unpack_1bw_32ow_32crw_1uf();
	bench2_unpack_2bw_32ow_32crw_1uf();
	bench3_unpack_3bw_32ow_32crw_1uf();
	bench4_unpack_4bw_32ow_32crw_1uf();
	bench5_unpack_5bw_32ow_32crw_1uf();
	bench6_unpack_6bw_32ow_32crw_1uf();
	bench7_unpack_7bw_32ow_32crw_1uf();
	bench8_unpack_8bw_32ow_32crw_1uf();
	bench9_unpack_9bw_32ow_32crw_1uf();
	bench10_unpack_10bw_32ow_32crw_1uf();
	bench11_unpack_11bw_32ow_32crw_1uf();
	bench12_unpack_12bw_32ow_32crw_1uf();
	bench13_unpack_13bw_32ow_32crw_1uf();
	bench14_unpack_14bw_32ow_32crw_1uf();
	bench15_unpack_15bw_32ow_32crw_1uf();
	bench16_unpack_16bw_32ow_32crw_1uf();
	bench17_unpack_17bw_32ow_32crw_1uf();
	bench18_unpack_18bw_32ow_32crw_1uf();
	bench19_unpack_19bw_32ow_32crw_1uf();
	bench20_unpack_20bw_32ow_32crw_1uf();
	bench21_unpack_21bw_32ow_32crw_1uf();
	bench22_unpack_22bw_32ow_32crw_1uf();
	bench23_unpack_23bw_32ow_32crw_1uf();
	bench24_unpack_24bw_32ow_32crw_1uf();
	bench25_unpack_25bw_32ow_32crw_1uf();
	bench26_unpack_26bw_32ow_32crw_1uf();
	bench27_unpack_27bw_32ow_32crw_1uf();
	bench28_unpack_28bw_32ow_32crw_1uf();
	bench29_unpack_29bw_32ow_32crw_1uf();
	bench30_unpack_30bw_32ow_32crw_1uf();
	bench31_unpack_31bw_32ow_32crw_1uf();
	bench32_unpack_32bw_32ow_32crw_1uf();
}
int main() { benchmark_all(); }
