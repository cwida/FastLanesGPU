// generated!
#include "cuda_normal_t32_1024_uf1_unpack_helper.hpp"
#include "fls_gen/pack/pack.hpp"
#include "fls_gen/unpack/unpack.cuh"
#include "gtest/gtest.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <fastlanes.cuh>
class cuda_normal_t32_1024_uf1_unpack : public ::testing::Test {

public:
	uint64_t  warp_sz {};
	uint64_t  n_vec {};
	uint64_t  vec_sz {};
	uint64_t  n_tup {};
	uint64_t  v_blc_sz {};
	uint64_t  n_blc {};
	uint64_t  n_trd {};
	uint32_t* d_decoded_arr {nullptr};
	uint32_t* h_decoded_arr {};
	uint32_t* packed32;
	uint32_t* unpacked32;
	uint32_t* d_encoded_arr;

	void SetUp() override {

		n_tup         = 1024;
		n_trd         = 32;
		n_blc         = 1;
		packed32      = new uint32_t[1024]();
		unpacked32    = new uint32_t[1024]();
		h_decoded_arr = new uint32_t[1024]();
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_decoded_arr, sizeof(uint32_t) * n_tup));
	}
	~cuda_normal_t32_1024_uf1_unpack() override {}
};
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_0_bw_0_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_0_b0_w32_arr, packed32, 0);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 0);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_0_b0_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_1_bw_1_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_1_b1_w32_arr, packed32, 1);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 1);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_1_b1_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_2_bw_2_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_2_b2_w32_arr, packed32, 2);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 2);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_2_b2_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_3_bw_3_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_3_b3_w32_arr, packed32, 3);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 3);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_3_b3_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_4_bw_4_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_4_b4_w32_arr, packed32, 4);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 4);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_4_b4_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_5_bw_5_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_5_b5_w32_arr, packed32, 5);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 5);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_5_b5_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_6_bw_6_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_6_b6_w32_arr, packed32, 6);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 6);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_6_b6_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_7_bw_7_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_7_b7_w32_arr, packed32, 7);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 7);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_7_b7_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_8_bw_8_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_8_b8_w32_arr, packed32, 8);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 8);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_8_b8_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_9_bw_9_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_9_b9_w32_arr, packed32, 9);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 9);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_9_b9_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_10_bw_10_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_10_b10_w32_arr, packed32, 10);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 10);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_10_b10_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_11_bw_11_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_11_b11_w32_arr, packed32, 11);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 11);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_11_b11_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_12_bw_12_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_12_b12_w32_arr, packed32, 12);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 12);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_12_b12_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_13_bw_13_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_13_b13_w32_arr, packed32, 13);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 13);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_13_b13_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_14_bw_14_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_14_b14_w32_arr, packed32, 14);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 14);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_14_b14_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_15_bw_15_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_15_b15_w32_arr, packed32, 15);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 15);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_15_b15_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_16_bw_16_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_16_b16_w32_arr, packed32, 16);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 16);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_16_b16_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_17_bw_17_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_17_b17_w32_arr, packed32, 17);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 17);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_17_b17_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_18_bw_18_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_18_b18_w32_arr, packed32, 18);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 18);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_18_b18_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_19_bw_19_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_19_b19_w32_arr, packed32, 19);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 19);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_19_b19_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_20_bw_20_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_20_b20_w32_arr, packed32, 20);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 20);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_20_b20_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_21_bw_21_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_21_b21_w32_arr, packed32, 21);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 21);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_21_b21_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_22_bw_22_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_22_b22_w32_arr, packed32, 22);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 22);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_22_b22_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_23_bw_23_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_23_b23_w32_arr, packed32, 23);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 23);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_23_b23_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_24_bw_24_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_24_b24_w32_arr, packed32, 24);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 24);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_24_b24_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_25_bw_25_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_25_b25_w32_arr, packed32, 25);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 25);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_25_b25_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_26_bw_26_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_26_b26_w32_arr, packed32, 26);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 26);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_26_b26_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_27_bw_27_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_27_b27_w32_arr, packed32, 27);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 27);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_27_b27_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_28_bw_28_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_28_b28_w32_arr, packed32, 28);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 28);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_28_b28_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_29_bw_29_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_29_b29_w32_arr, packed32, 29);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 29);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_29_b29_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_30_bw_30_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_30_b30_w32_arr, packed32, 30);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 30);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_30_b30_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_31_bw_31_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_31_b31_w32_arr, packed32, 31);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 31);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_31_b31_w32_arr[i], h_decoded_arr[i]);
	}
}
TEST_F(cuda_normal_t32_1024_uf1_unpack, test_32_bw_32_ow_32) {

	generated::pack::fallback::scalar::pack(helper::rand_arr_32_b32_w32_arr, packed32, 32);
	d_encoded_arr = fastlanes::gpu::load_arr(packed32, 32 * 1024 / 8);
	unpack_global<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, 32);
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n_tup; i++) {
		ASSERT_EQ(helper::rand_arr_32_b32_w32_arr[i], h_decoded_arr[i]);
	}
}
