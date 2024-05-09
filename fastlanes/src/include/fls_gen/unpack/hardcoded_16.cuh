// generated!
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

// static constexpr uint8_t lo_orderdate_bw     = 16;
// static constexpr uint8_t lo_extendedprice_bw = 24;
// static constexpr uint8_t lo_quantity_bw      = 6;
// static constexpr uint8_t lo_discount_bw      = 4;
// static constexpr uint8_t lo_partkey_bw       = 20;
// static constexpr uint8_t lo_suppkey_bw       = 15;
// static constexpr uint8_t lo_revenue_bw       = 24;
// static constexpr uint8_t lo_custkey_bw       = 19;
// static constexpr uint8_t lo_supplycost_bw    = 17;

namespace hardcoded_16 {

__device__ __forceinline__ void unpack_16bw_32ow_32crw_1uf(const uint32_t* __restrict a_in_p,
                                                           uint16_t* __restrict a_out_p) {
	[[maybe_unused]] auto     out = reinterpret_cast<uint16_t*>(a_out_p);
	[[maybe_unused]] auto     in  = reinterpret_cast<const uint32_t*>(a_in_p);
	[[maybe_unused]] uint32_t register_0;
	[[maybe_unused]] uint32_t tmp_0;
	[[maybe_unused]] uint32_t base_0 = 0ULL;

	int i = threadIdx.x; // THREAD INDEX

	register_0 = *(in + (0 * 32) + (i * 1) + 0);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[0]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[1]     = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 32);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[2]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[3]     = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 64);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[4]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[5]     = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 96);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[6]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[7]     = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 128);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[8]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[9]     = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 160);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[10]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[11]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 192);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[12]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[13]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 224);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[14]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[15]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 256);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[16]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[17]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 288);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[18]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[19]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 320);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[20]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[21]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 352);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[22]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[23]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 384);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[24]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[25]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 416);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[26]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[27]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 448);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[28]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[29]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 480);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[30]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[31]    = tmp_0;
}
__device__ __forceinline__ void unpack_4bw_32ow_32crw_1uf(const uint32_t* __restrict a_in_p,
                                                          uint16_t* __restrict a_out_p) {
	[[maybe_unused]] auto     out = reinterpret_cast<uint16_t*>(a_out_p);
	[[maybe_unused]] auto     in  = reinterpret_cast<const uint32_t*>(a_in_p);
	[[maybe_unused]] uint32_t register_0;
	[[maybe_unused]] uint32_t tmp_0;
	[[maybe_unused]] uint32_t base_0 = 0ULL;

	int i = threadIdx.x; // THREAD INDEX

	register_0 = *(in + (0 * 32) + (i * 1) + 0);
	tmp_0      = (register_0) & ((1ULL << 4) - 1);
	out[0]     = tmp_0;
	tmp_0      = (register_0 >> 4) & ((1ULL << 4) - 1);
	out[1]     = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 4) - 1);
	out[2]     = tmp_0;
	tmp_0      = (register_0 >> 12) & ((1ULL << 4) - 1);
	out[3]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 4) - 1);
	out[4]     = tmp_0;
	tmp_0      = (register_0 >> 20) & ((1ULL << 4) - 1);
	out[5]     = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 4) - 1);
	out[6]     = tmp_0;
	tmp_0      = (register_0 >> 28) & ((1ULL << 4) - 1);
	out[7]     = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 32);
	tmp_0      = (register_0) & ((1ULL << 4) - 1);
	out[8]     = tmp_0;
	tmp_0      = (register_0 >> 4) & ((1ULL << 4) - 1);
	out[9]     = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 4) - 1);
	out[10]    = tmp_0;
	tmp_0      = (register_0 >> 12) & ((1ULL << 4) - 1);
	out[11]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 4) - 1);
	out[12]    = tmp_0;
	tmp_0      = (register_0 >> 20) & ((1ULL << 4) - 1);
	out[13]    = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 4) - 1);
	out[14]    = tmp_0;
	tmp_0      = (register_0 >> 28) & ((1ULL << 4) - 1);
	out[15]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 64);
	tmp_0      = (register_0) & ((1ULL << 4) - 1);
	out[16]    = tmp_0;
	tmp_0      = (register_0 >> 4) & ((1ULL << 4) - 1);
	out[17]    = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 4) - 1);
	out[18]    = tmp_0;
	tmp_0      = (register_0 >> 12) & ((1ULL << 4) - 1);
	out[19]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 4) - 1);
	out[20]    = tmp_0;
	tmp_0      = (register_0 >> 20) & ((1ULL << 4) - 1);
	out[21]    = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 4) - 1);
	out[22]    = tmp_0;
	tmp_0      = (register_0 >> 28) & ((1ULL << 4) - 1);
	out[23]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 96);
	tmp_0      = (register_0) & ((1ULL << 4) - 1);
	out[24]    = tmp_0;
	tmp_0      = (register_0 >> 4) & ((1ULL << 4) - 1);
	out[25]    = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 4) - 1);
	out[26]    = tmp_0;
	tmp_0      = (register_0 >> 12) & ((1ULL << 4) - 1);
	out[27]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 4) - 1);
	out[28]    = tmp_0;
	tmp_0      = (register_0 >> 20) & ((1ULL << 4) - 1);
	out[29]    = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 4) - 1);
	out[30]    = tmp_0;
	tmp_0      = (register_0 >> 28) & ((1ULL << 4) - 1);
	out[31]    = tmp_0;
}
__device__ __forceinline__ void unpack_24bw_32ow_32crw_1uf(const uint32_t* __restrict a_in_p,
                                                           uint16_t* __restrict a_out_p) {
	[[maybe_unused]] auto     out = reinterpret_cast<uint16_t*>(a_out_p);
	[[maybe_unused]] auto     in  = reinterpret_cast<const uint32_t*>(a_in_p);
	[[maybe_unused]] uint32_t register_0;
	[[maybe_unused]] uint32_t tmp_0;
	[[maybe_unused]] uint32_t base_0 = 0ULL;

	int i = threadIdx.x; // THREAD INDEX

	register_0 = *(in + (0 * 32) + (i * 1) + 0);
	tmp_0      = (register_0) & ((1ULL << 24) - 1);
	out[0]     = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 8) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 32);
	tmp_0 |= ((register_0) & ((1ULL << 16) - 1)) << 8;
	out[1]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 64);
	tmp_0 |= ((register_0) & ((1ULL << 8) - 1)) << 16;
	out[2]     = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 24) - 1);
	out[3]     = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 96);
	tmp_0      = (register_0) & ((1ULL << 24) - 1);
	out[4]     = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 8) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 128);
	tmp_0 |= ((register_0) & ((1ULL << 16) - 1)) << 8;
	out[5]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 160);
	tmp_0 |= ((register_0) & ((1ULL << 8) - 1)) << 16;
	out[6]     = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 24) - 1);
	out[7]     = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 192);
	tmp_0      = (register_0) & ((1ULL << 24) - 1);
	out[8]     = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 8) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 224);
	tmp_0 |= ((register_0) & ((1ULL << 16) - 1)) << 8;
	out[9]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 256);
	tmp_0 |= ((register_0) & ((1ULL << 8) - 1)) << 16;
	out[10]    = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 24) - 1);
	out[11]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 288);
	tmp_0      = (register_0) & ((1ULL << 24) - 1);
	out[12]    = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 8) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 320);
	tmp_0 |= ((register_0) & ((1ULL << 16) - 1)) << 8;
	out[13]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 352);
	tmp_0 |= ((register_0) & ((1ULL << 8) - 1)) << 16;
	out[14]    = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 24) - 1);
	out[15]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 384);
	tmp_0      = (register_0) & ((1ULL << 24) - 1);
	out[16]    = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 8) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 416);
	tmp_0 |= ((register_0) & ((1ULL << 16) - 1)) << 8;
	out[17]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 448);
	tmp_0 |= ((register_0) & ((1ULL << 8) - 1)) << 16;
	out[18]    = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 24) - 1);
	out[19]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 480);
	tmp_0      = (register_0) & ((1ULL << 24) - 1);
	out[20]    = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 8) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 512);
	tmp_0 |= ((register_0) & ((1ULL << 16) - 1)) << 8;
	out[21]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 544);
	tmp_0 |= ((register_0) & ((1ULL << 8) - 1)) << 16;
	out[22]    = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 24) - 1);
	out[23]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 576);
	tmp_0      = (register_0) & ((1ULL << 24) - 1);
	out[24]    = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 8) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 608);
	tmp_0 |= ((register_0) & ((1ULL << 16) - 1)) << 8;
	out[25]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 640);
	tmp_0 |= ((register_0) & ((1ULL << 8) - 1)) << 16;
	out[26]    = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 24) - 1);
	out[27]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 672);
	tmp_0      = (register_0) & ((1ULL << 24) - 1);
	out[28]    = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 8) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 704);
	tmp_0 |= ((register_0) & ((1ULL << 16) - 1)) << 8;
	out[29]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 736);
	tmp_0 |= ((register_0) & ((1ULL << 8) - 1)) << 16;
	out[30] = tmp_0;
	tmp_0   = (register_0 >> 8) & ((1ULL << 24) - 1);
	out[31] = tmp_0;
}

__device__ __forceinline__ void unpack_6bw_32ow_32crw_1uf(const uint32_t* __restrict a_in_p,
                                                          uint16_t* __restrict a_out_p) {
	[[maybe_unused]] auto     out = reinterpret_cast<uint16_t*>(a_out_p);
	[[maybe_unused]] auto     in  = reinterpret_cast<const uint32_t*>(a_in_p);
	[[maybe_unused]] uint32_t register_0;
	[[maybe_unused]] uint32_t tmp_0;
	[[maybe_unused]] uint32_t base_0 = 0ULL;

	int i = threadIdx.x; // THREAD INDEX

	register_0 = *(in + (0 * 32) + (i * 1) + 0);
	tmp_0      = (register_0) & ((1ULL << 6) - 1);
	out[0]     = tmp_0;
	tmp_0      = (register_0 >> 6) & ((1ULL << 6) - 1);
	out[1]     = tmp_0;
	tmp_0      = (register_0 >> 12) & ((1ULL << 6) - 1);
	out[2]     = tmp_0;
	tmp_0      = (register_0 >> 18) & ((1ULL << 6) - 1);
	out[3]     = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 6) - 1);
	out[4]     = tmp_0;
	tmp_0      = (register_0 >> 30) & ((1ULL << 2) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 32);
	tmp_0 |= ((register_0) & ((1ULL << 4) - 1)) << 2;
	out[5]     = tmp_0;
	tmp_0      = (register_0 >> 4) & ((1ULL << 6) - 1);
	out[6]     = tmp_0;
	tmp_0      = (register_0 >> 10) & ((1ULL << 6) - 1);
	out[7]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 6) - 1);
	out[8]     = tmp_0;
	tmp_0      = (register_0 >> 22) & ((1ULL << 6) - 1);
	out[9]     = tmp_0;
	tmp_0      = (register_0 >> 28) & ((1ULL << 4) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 64);
	tmp_0 |= ((register_0) & ((1ULL << 2) - 1)) << 4;
	out[10]    = tmp_0;
	tmp_0      = (register_0 >> 2) & ((1ULL << 6) - 1);
	out[11]    = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 6) - 1);
	out[12]    = tmp_0;
	tmp_0      = (register_0 >> 14) & ((1ULL << 6) - 1);
	out[13]    = tmp_0;
	tmp_0      = (register_0 >> 20) & ((1ULL << 6) - 1);
	out[14]    = tmp_0;
	tmp_0      = (register_0 >> 26) & ((1ULL << 6) - 1);
	out[15]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 96);
	tmp_0      = (register_0) & ((1ULL << 6) - 1);
	out[16]    = tmp_0;
	tmp_0      = (register_0 >> 6) & ((1ULL << 6) - 1);
	out[17]    = tmp_0;
	tmp_0      = (register_0 >> 12) & ((1ULL << 6) - 1);
	out[18]    = tmp_0;
	tmp_0      = (register_0 >> 18) & ((1ULL << 6) - 1);
	out[19]    = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 6) - 1);
	out[20]    = tmp_0;
	tmp_0      = (register_0 >> 30) & ((1ULL << 2) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 128);
	tmp_0 |= ((register_0) & ((1ULL << 4) - 1)) << 2;
	out[21]    = tmp_0;
	tmp_0      = (register_0 >> 4) & ((1ULL << 6) - 1);
	out[22]    = tmp_0;
	tmp_0      = (register_0 >> 10) & ((1ULL << 6) - 1);
	out[23]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 6) - 1);
	out[24]    = tmp_0;
	tmp_0      = (register_0 >> 22) & ((1ULL << 6) - 1);
	out[25]    = tmp_0;
	tmp_0      = (register_0 >> 28) & ((1ULL << 4) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 160);
	tmp_0 |= ((register_0) & ((1ULL << 2) - 1)) << 4;
	out[26] = tmp_0;
	tmp_0   = (register_0 >> 2) & ((1ULL << 6) - 1);
	out[27] = tmp_0;
	tmp_0   = (register_0 >> 8) & ((1ULL << 6) - 1);
	out[28] = tmp_0;
	tmp_0   = (register_0 >> 14) & ((1ULL << 6) - 1);
	out[29] = tmp_0;
	tmp_0   = (register_0 >> 20) & ((1ULL << 6) - 1);
	out[30] = tmp_0;
	tmp_0   = (register_0 >> 26) & ((1ULL << 6) - 1);
	out[31] = tmp_0;
}

__device__ __forceinline__ void unpack(const uint32_t* __restrict a_in_p, uint16_t* __restrict a_out_p, uint8_t bw) {
	switch (bw) {
	case 24:
		unpack_24bw_32ow_32crw_1uf(a_in_p, a_out_p);
		break;
	case 4:
		unpack_4bw_32ow_32crw_1uf(a_in_p, a_out_p);
		break;
	case 16:
		unpack_16bw_32ow_32crw_1uf(a_in_p, a_out_p);
		break;
	case 6:
		unpack_6bw_32ow_32crw_1uf(a_in_p, a_out_p);
		break;
	}
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void load_registers(int i, T* out, T* registers) {

#pragma unroll
	for (int j = 0; j < ITEMS_PER_THREAD; j++) {
		out[j * BLOCK_THREADS + i] = registers[j];
	}
}

__global__ void unpack_global(const uint32_t* __restrict in, uint16_t* __restrict out, uint8_t bw) {
	int trd_idx = threadIdx.x;
	int blc_idx = blockIdx.x;
	in          = in + ((blc_idx * bw) << 5);
	out         = out + (blc_idx << 10);

	uint16_t registers[32];
	unpack(in, registers, bw);
	load_registers<uint16_t, 32, 32>(trd_idx, out, registers);
}

__device__ __forceinline__ void unpack_device(const uint32_t* __restrict in, uint16_t* __restrict out, uint8_t bw) {
	unpack(in, out, bw);
}

__device__ __forceinline__ void unpack_device(const int32_t* __restrict in, uint16_t* __restrict out, uint8_t bw) {
	unpack(reinterpret_cast<const uint32_t*>(in), reinterpret_cast<uint16_t*>(out), bw);
}
} // namespace hardcoded_16

namespace unpack_8_at_a_time {
__device__ void unpack_20bw_32ow_32crw_1uf(const uint32_t* __restrict a_in_p, uint32_t* __restrict a_out_p) {
	[[maybe_unused]] auto     out = reinterpret_cast<uint32_t*>(a_out_p);
	[[maybe_unused]] auto     in  = reinterpret_cast<const uint32_t*>(a_in_p);
	[[maybe_unused]] uint32_t register_0;
	[[maybe_unused]] uint32_t tmp_0;
	[[maybe_unused]] uint32_t base_0 = 0ULL;

	int i = threadIdx.x; // THREAD INDEX

	register_0 = *(in + (0 * 32) + (i * 1) + 0);
	tmp_0      = (register_0) & ((1ULL << 20) - 1);
	out[0]     = tmp_0;
	tmp_0      = (register_0 >> 20) & ((1ULL << 12) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 32);
	tmp_0 |= ((register_0) & ((1ULL << 8) - 1)) << 12;
	out[1]     = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 20) - 1);
	out[2]     = tmp_0;
	tmp_0      = (register_0 >> 28) & ((1ULL << 4) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 64);
	tmp_0 |= ((register_0) & ((1ULL << 16) - 1)) << 4;
	out[3]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 96);
	tmp_0 |= ((register_0) & ((1ULL << 4) - 1)) << 16;
	out[4]     = tmp_0;
	tmp_0      = (register_0 >> 4) & ((1ULL << 20) - 1);
	out[5]     = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 8) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 128);
	tmp_0 |= ((register_0) & ((1ULL << 12) - 1)) << 8;
	out[6] = tmp_0;
	tmp_0  = (register_0 >> 12) & ((1ULL << 20) - 1);
	out[7] = tmp_0;
}

__device__ __forceinline__ void unpack_16bw_32ow_32crw_1uf(const uint32_t* __restrict a_in_p,
                                                           uint32_t* __restrict a_out_p) {
	[[maybe_unused]] auto     out = reinterpret_cast<uint32_t*>(a_out_p);
	[[maybe_unused]] auto     in  = reinterpret_cast<const uint32_t*>(a_in_p);
	[[maybe_unused]] uint32_t register_0;
	[[maybe_unused]] uint32_t tmp_0;
	[[maybe_unused]] uint32_t base_0 = 0ULL;

	int i = threadIdx.x; // THREAD INDEX

	register_0 = *(in + (0 * 32) + (i * 1) + 0);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[0]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[1]     = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 32);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[2]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[3]     = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 64);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[4]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[5]     = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 96);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[6]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[7]     = tmp_0;
}

__device__ __forceinline__ void unpack_24bw_32ow_32crw_1uf(const uint32_t* __restrict a_in_p,
                                                           uint32_t* __restrict a_out_p) {
	[[maybe_unused]] auto     out = reinterpret_cast<uint32_t*>(a_out_p);
	[[maybe_unused]] auto     in  = reinterpret_cast<const uint32_t*>(a_in_p);
	[[maybe_unused]] uint32_t register_0;
	[[maybe_unused]] uint32_t tmp_0;
	[[maybe_unused]] uint32_t base_0 = 0ULL;

	int i = threadIdx.x; // THREAD INDEX

	register_0 = *(in + (0 * 32) + (i * 1) + 0);
	tmp_0      = (register_0) & ((1ULL << 24) - 1);
	out[0]     = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 8) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 32);
	tmp_0 |= ((register_0) & ((1ULL << 16) - 1)) << 8;
	out[1]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 64);
	tmp_0 |= ((register_0) & ((1ULL << 8) - 1)) << 16;
	out[2]     = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 24) - 1);
	out[3]     = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 96);
	tmp_0      = (register_0) & ((1ULL << 24) - 1);
	out[4]     = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 8) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 128);
	tmp_0 |= ((register_0) & ((1ULL << 16) - 1)) << 8;
	out[5]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 160);
	tmp_0 |= ((register_0) & ((1ULL << 8) - 1)) << 16;
	out[6]     = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 24) - 1);
	out[7]     = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 192);
	tmp_0      = (register_0) & ((1ULL << 24) - 1);
	out[8]     = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 8) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 224);
	tmp_0 |= ((register_0) & ((1ULL << 16) - 1)) << 8;
	out[9]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 256);
	tmp_0 |= ((register_0) & ((1ULL << 8) - 1)) << 16;
	out[10]    = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 24) - 1);
	out[11]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 288);
	tmp_0      = (register_0) & ((1ULL << 24) - 1);
	out[12]    = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 8) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 320);
	tmp_0 |= ((register_0) & ((1ULL << 16) - 1)) << 8;
	out[13]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 352);
	tmp_0 |= ((register_0) & ((1ULL << 8) - 1)) << 16;
	out[14]    = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 24) - 1);
	out[15]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 384);
	tmp_0      = (register_0) & ((1ULL << 24) - 1);
	out[16]    = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 8) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 416);
	tmp_0 |= ((register_0) & ((1ULL << 16) - 1)) << 8;
	out[17]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 448);
	tmp_0 |= ((register_0) & ((1ULL << 8) - 1)) << 16;
	out[18]    = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 24) - 1);
	out[19]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 480);
	tmp_0      = (register_0) & ((1ULL << 24) - 1);
	out[20]    = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 8) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 512);
	tmp_0 |= ((register_0) & ((1ULL << 16) - 1)) << 8;
	out[21]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 544);
	tmp_0 |= ((register_0) & ((1ULL << 8) - 1)) << 16;
	out[22]    = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 24) - 1);
	out[23]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 576);
	tmp_0      = (register_0) & ((1ULL << 24) - 1);
	out[24]    = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 8) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 608);
	tmp_0 |= ((register_0) & ((1ULL << 16) - 1)) << 8;
	out[25]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 640);
	tmp_0 |= ((register_0) & ((1ULL << 8) - 1)) << 16;
	out[26]    = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 24) - 1);
	out[27]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 672);
	tmp_0      = (register_0) & ((1ULL << 24) - 1);
	out[28]    = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 8) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 704);
	tmp_0 |= ((register_0) & ((1ULL << 16) - 1)) << 8;
	out[29]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 736);
	tmp_0 |= ((register_0) & ((1ULL << 8) - 1)) << 16;
	out[30] = tmp_0;
	tmp_0   = (register_0 >> 8) & ((1ULL << 24) - 1);
	out[31] = tmp_0;
}

__device__ __forceinline__ void unpack_0bw_32ow_32crw_1uf(const uint32_t* __restrict a_in_p,
                                                          uint32_t* __restrict a_out_p) {
	[[maybe_unused]] auto     out = reinterpret_cast<uint32_t*>(a_out_p);
	[[maybe_unused]] auto     in  = reinterpret_cast<const uint32_t*>(a_in_p);
	[[maybe_unused]] uint32_t register_0;
	[[maybe_unused]] uint32_t tmp_0;
	[[maybe_unused]] uint32_t base_0 = 0ULL;

	out[0] = base_0;
	out[1] = base_0;
	out[2] = base_0;
	out[3] = base_0;
	out[4] = base_0;
	out[5] = base_0;
	out[6] = base_0;
	out[7] = base_0;
}

__device__ __forceinline__ void unpack_8bw_32ow_32crw_1uf(const uint32_t* __restrict a_in_p,
                                                          uint32_t* __restrict a_out_p) {
	[[maybe_unused]] auto     out = reinterpret_cast<uint32_t*>(a_out_p);
	[[maybe_unused]] auto     in  = reinterpret_cast<const uint32_t*>(a_in_p);
	[[maybe_unused]] uint32_t register_0;
	[[maybe_unused]] uint32_t tmp_0;
	[[maybe_unused]] uint32_t base_0 = 0ULL;

	int i = threadIdx.x; // THREAD INDEX

	register_0 = *(in + (0 * 32) + (i * 1) + 0);
	tmp_0      = (register_0) & ((1ULL << 8) - 1);
	out[0]     = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 8) - 1);
	out[1]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 8) - 1);
	out[2]     = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 8) - 1);
	out[3]     = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 32);
	tmp_0      = (register_0) & ((1ULL << 8) - 1);
	out[4]     = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 8) - 1);
	out[5]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 8) - 1);
	out[6]     = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 8) - 1);
	out[7]     = tmp_0;
}

__device__ __forceinline__ void unpack_4bw_32ow_32crw_1uf(const uint32_t* __restrict a_in_p,
                                                          uint32_t* __restrict a_out_p) {
	[[maybe_unused]] auto     out = reinterpret_cast<uint32_t*>(a_out_p);
	[[maybe_unused]] auto     in  = reinterpret_cast<const uint32_t*>(a_in_p);
	[[maybe_unused]] uint32_t register_0;
	[[maybe_unused]] uint32_t tmp_0;
	[[maybe_unused]] uint32_t base_0 = 0ULL;

	int i = threadIdx.x; // THREAD INDEX

	register_0 = *(in + (0 * 32) + (i * 1) + 0);
	tmp_0      = (register_0) & ((1ULL << 4) - 1);
	out[0]     = tmp_0;
	tmp_0      = (register_0 >> 4) & ((1ULL << 4) - 1);
	out[1]     = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 4) - 1);
	out[2]     = tmp_0;
	tmp_0      = (register_0 >> 12) & ((1ULL << 4) - 1);
	out[3]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 4) - 1);
	out[4]     = tmp_0;
	tmp_0      = (register_0 >> 20) & ((1ULL << 4) - 1);
	out[5]     = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 4) - 1);
	out[6]     = tmp_0;
	tmp_0      = (register_0 >> 28) & ((1ULL << 4) - 1);
	out[7]     = tmp_0;
}

inline __device__ void unpack_12bw_32ow_32crw_1uf(const uint32_t* __restrict a_in_p, uint32_t* __restrict a_out_p) {
	[[maybe_unused]] auto     out = reinterpret_cast<uint32_t*>(a_out_p);
	[[maybe_unused]] auto     in  = reinterpret_cast<const uint32_t*>(a_in_p);
	[[maybe_unused]] uint32_t register_0;
	[[maybe_unused]] uint32_t tmp_0;
	[[maybe_unused]] uint32_t base_0 = 0ULL;

	int i = threadIdx.x; // THREAD INDEX

	register_0 = *(in + (0 * 32) + (i * 1) + 0);
	tmp_0      = (register_0) & ((1ULL << 12) - 1);
	out[0]     = tmp_0;
	tmp_0      = (register_0 >> 12) & ((1ULL << 12) - 1);
	out[1]     = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 8) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 32);
	tmp_0 |= ((register_0) & ((1ULL << 4) - 1)) << 8;
	out[2]     = tmp_0;
	tmp_0      = (register_0 >> 4) & ((1ULL << 12) - 1);
	out[3]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 12) - 1);
	out[4]     = tmp_0;
	tmp_0      = (register_0 >> 28) & ((1ULL << 4) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 64);
	tmp_0 |= ((register_0) & ((1ULL << 8) - 1)) << 4;
	out[5] = tmp_0;
	tmp_0  = (register_0 >> 8) & ((1ULL << 12) - 1);
	out[6] = tmp_0;
	tmp_0  = (register_0 >> 20) & ((1ULL << 12) - 1);
	out[7] = tmp_0;
}

__device__ __forceinline__ void unpack(const uint32_t* __restrict a_in_p, uint32_t* __restrict a_out_p, uint8_t bw) {
	switch (bw) {
	case 0:
		unpack_0bw_32ow_32crw_1uf(a_in_p, a_out_p);
		break;
	case 4:
		unpack_4bw_32ow_32crw_1uf(a_in_p, a_out_p);
		break;
	case 8:
		unpack_8bw_32ow_32crw_1uf(a_in_p, a_out_p);
		break;
	case 12:
		unpack_12bw_32ow_32crw_1uf(a_in_p, a_out_p);
		break;
	case 16:
		unpack_16bw_32ow_32crw_1uf(a_in_p, a_out_p);
		break;
	case 20:
		unpack_20bw_32ow_32crw_1uf(a_in_p, a_out_p);
		break;
	case 24:
		unpack_24bw_32ow_32crw_1uf(a_in_p, a_out_p);
		break;
	default:
		printf("implement this bw! %u\n", bw);
		asm("trap;");
	}
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void load_registers(int i, T* out, T* registers) {

#pragma unroll
	for (int j = 0; j < ITEMS_PER_THREAD; j++) {
		out[j * BLOCK_THREADS + i] = registers[j];
	}
}

__global__ void unpack_global(const uint32_t* __restrict in, uint32_t* __restrict out, uint8_t bw) {
	int trd_idx = threadIdx.x;
	int blc_idx = blockIdx.x;
	in          = in + ((blc_idx * bw) << 5);
	out         = out + (blc_idx << 10);

	uint32_t registers[32];
	unpack(in, registers, bw);
	load_registers<uint32_t, 32, 32>(trd_idx, out, registers);
}

__device__ __forceinline__ void unpack_device(const uint32_t* __restrict in, uint32_t* __restrict out, uint8_t bw) {
	unpack(in, out, bw);
}

__device__ __forceinline__ void unpack_device(const int32_t* __restrict in, int32_t* __restrict out, uint8_t bw) {
	unpack(reinterpret_cast<const uint32_t*>(in), reinterpret_cast<uint32_t*>(out), bw);
}
} // namespace unpack_8_at_a_time

namespace hardcoded {

__device__ __forceinline__ void unpack_16bw_32ow_32crw_1uf(const uint32_t* __restrict a_in_p,
                                                           uint32_t* __restrict a_out_p) {
	[[maybe_unused]] auto     out = reinterpret_cast<uint32_t*>(a_out_p);
	[[maybe_unused]] auto     in  = reinterpret_cast<const uint32_t*>(a_in_p);
	[[maybe_unused]] uint32_t register_0;
	[[maybe_unused]] uint32_t tmp_0;
	[[maybe_unused]] uint32_t base_0 = 0ULL;

	int i = threadIdx.x; // THREAD INDEX

	register_0 = *(in + (0 * 32) + (i * 1) + 0);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[0]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[1]     = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 32);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[2]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[3]     = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 64);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[4]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[5]     = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 96);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[6]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[7]     = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 128);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[8]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[9]     = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 160);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[10]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[11]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 192);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[12]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[13]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 224);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[14]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[15]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 256);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[16]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[17]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 288);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[18]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[19]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 320);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[20]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[21]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 352);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[22]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[23]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 384);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[24]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[25]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 416);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[26]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[27]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 448);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[28]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[29]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 480);
	tmp_0      = (register_0) & ((1ULL << 16) - 1);
	out[30]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	out[31]    = tmp_0;
}
__device__ __forceinline__ void unpack_4bw_32ow_32crw_1uf(const uint32_t* __restrict a_in_p,
                                                          uint32_t* __restrict a_out_p) {
	[[maybe_unused]] auto     out = reinterpret_cast<uint32_t*>(a_out_p);
	[[maybe_unused]] auto     in  = reinterpret_cast<const uint32_t*>(a_in_p);
	[[maybe_unused]] uint32_t register_0;
	[[maybe_unused]] uint32_t tmp_0;
	[[maybe_unused]] uint32_t base_0 = 0ULL;

	int i = threadIdx.x; // THREAD INDEX

	register_0 = *(in + (0 * 32) + (i * 1) + 0);
	tmp_0      = (register_0) & ((1ULL << 4) - 1);
	out[0]     = tmp_0;
	tmp_0      = (register_0 >> 4) & ((1ULL << 4) - 1);
	out[1]     = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 4) - 1);
	out[2]     = tmp_0;
	tmp_0      = (register_0 >> 12) & ((1ULL << 4) - 1);
	out[3]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 4) - 1);
	out[4]     = tmp_0;
	tmp_0      = (register_0 >> 20) & ((1ULL << 4) - 1);
	out[5]     = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 4) - 1);
	out[6]     = tmp_0;
	tmp_0      = (register_0 >> 28) & ((1ULL << 4) - 1);
	out[7]     = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 32);
	tmp_0      = (register_0) & ((1ULL << 4) - 1);
	out[8]     = tmp_0;
	tmp_0      = (register_0 >> 4) & ((1ULL << 4) - 1);
	out[9]     = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 4) - 1);
	out[10]    = tmp_0;
	tmp_0      = (register_0 >> 12) & ((1ULL << 4) - 1);
	out[11]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 4) - 1);
	out[12]    = tmp_0;
	tmp_0      = (register_0 >> 20) & ((1ULL << 4) - 1);
	out[13]    = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 4) - 1);
	out[14]    = tmp_0;
	tmp_0      = (register_0 >> 28) & ((1ULL << 4) - 1);
	out[15]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 64);
	tmp_0      = (register_0) & ((1ULL << 4) - 1);
	out[16]    = tmp_0;
	tmp_0      = (register_0 >> 4) & ((1ULL << 4) - 1);
	out[17]    = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 4) - 1);
	out[18]    = tmp_0;
	tmp_0      = (register_0 >> 12) & ((1ULL << 4) - 1);
	out[19]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 4) - 1);
	out[20]    = tmp_0;
	tmp_0      = (register_0 >> 20) & ((1ULL << 4) - 1);
	out[21]    = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 4) - 1);
	out[22]    = tmp_0;
	tmp_0      = (register_0 >> 28) & ((1ULL << 4) - 1);
	out[23]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 96);
	tmp_0      = (register_0) & ((1ULL << 4) - 1);
	out[24]    = tmp_0;
	tmp_0      = (register_0 >> 4) & ((1ULL << 4) - 1);
	out[25]    = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 4) - 1);
	out[26]    = tmp_0;
	tmp_0      = (register_0 >> 12) & ((1ULL << 4) - 1);
	out[27]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 4) - 1);
	out[28]    = tmp_0;
	tmp_0      = (register_0 >> 20) & ((1ULL << 4) - 1);
	out[29]    = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 4) - 1);
	out[30]    = tmp_0;
	tmp_0      = (register_0 >> 28) & ((1ULL << 4) - 1);
	out[31]    = tmp_0;
}
__device__ __forceinline__ void unpack_24bw_32ow_32crw_1uf(const uint32_t* __restrict a_in_p,
                                                           uint32_t* __restrict a_out_p) {
	[[maybe_unused]] auto     out = reinterpret_cast<uint32_t*>(a_out_p);
	[[maybe_unused]] auto     in  = reinterpret_cast<const uint32_t*>(a_in_p);
	[[maybe_unused]] uint32_t register_0;
	[[maybe_unused]] uint32_t tmp_0;
	[[maybe_unused]] uint32_t base_0 = 0ULL;

	int i = threadIdx.x; // THREAD INDEX

	register_0 = *(in + (0 * 32) + (i * 1) + 0);
	tmp_0      = (register_0) & ((1ULL << 24) - 1);
	out[0]     = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 8) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 32);
	tmp_0 |= ((register_0) & ((1ULL << 16) - 1)) << 8;
	out[1]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 64);
	tmp_0 |= ((register_0) & ((1ULL << 8) - 1)) << 16;
	out[2]     = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 24) - 1);
	out[3]     = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 96);
	tmp_0      = (register_0) & ((1ULL << 24) - 1);
	out[4]     = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 8) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 128);
	tmp_0 |= ((register_0) & ((1ULL << 16) - 1)) << 8;
	out[5]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 160);
	tmp_0 |= ((register_0) & ((1ULL << 8) - 1)) << 16;
	out[6]     = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 24) - 1);
	out[7]     = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 192);
	tmp_0      = (register_0) & ((1ULL << 24) - 1);
	out[8]     = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 8) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 224);
	tmp_0 |= ((register_0) & ((1ULL << 16) - 1)) << 8;
	out[9]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 256);
	tmp_0 |= ((register_0) & ((1ULL << 8) - 1)) << 16;
	out[10]    = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 24) - 1);
	out[11]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 288);
	tmp_0      = (register_0) & ((1ULL << 24) - 1);
	out[12]    = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 8) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 320);
	tmp_0 |= ((register_0) & ((1ULL << 16) - 1)) << 8;
	out[13]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 352);
	tmp_0 |= ((register_0) & ((1ULL << 8) - 1)) << 16;
	out[14]    = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 24) - 1);
	out[15]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 384);
	tmp_0      = (register_0) & ((1ULL << 24) - 1);
	out[16]    = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 8) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 416);
	tmp_0 |= ((register_0) & ((1ULL << 16) - 1)) << 8;
	out[17]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 448);
	tmp_0 |= ((register_0) & ((1ULL << 8) - 1)) << 16;
	out[18]    = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 24) - 1);
	out[19]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 480);
	tmp_0      = (register_0) & ((1ULL << 24) - 1);
	out[20]    = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 8) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 512);
	tmp_0 |= ((register_0) & ((1ULL << 16) - 1)) << 8;
	out[21]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 544);
	tmp_0 |= ((register_0) & ((1ULL << 8) - 1)) << 16;
	out[22]    = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 24) - 1);
	out[23]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 576);
	tmp_0      = (register_0) & ((1ULL << 24) - 1);
	out[24]    = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 8) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 608);
	tmp_0 |= ((register_0) & ((1ULL << 16) - 1)) << 8;
	out[25]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 640);
	tmp_0 |= ((register_0) & ((1ULL << 8) - 1)) << 16;
	out[26]    = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 24) - 1);
	out[27]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 672);
	tmp_0      = (register_0) & ((1ULL << 24) - 1);
	out[28]    = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 8) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 704);
	tmp_0 |= ((register_0) & ((1ULL << 16) - 1)) << 8;
	out[29]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 16) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 736);
	tmp_0 |= ((register_0) & ((1ULL << 8) - 1)) << 16;
	out[30] = tmp_0;
	tmp_0   = (register_0 >> 8) & ((1ULL << 24) - 1);
	out[31] = tmp_0;
}

__device__ __forceinline__ void unpack_6bw_32ow_32crw_1uf(const uint32_t* __restrict a_in_p,
                                                          uint32_t* __restrict a_out_p) {
	[[maybe_unused]] auto     out = reinterpret_cast<uint32_t*>(a_out_p);
	[[maybe_unused]] auto     in  = reinterpret_cast<const uint32_t*>(a_in_p);
	[[maybe_unused]] uint32_t register_0;
	[[maybe_unused]] uint32_t tmp_0;
	[[maybe_unused]] uint32_t base_0 = 0ULL;

	int i = threadIdx.x; // THREAD INDEX

	register_0 = *(in + (0 * 32) + (i * 1) + 0);
	tmp_0      = (register_0) & ((1ULL << 6) - 1);
	out[0]     = tmp_0;
	tmp_0      = (register_0 >> 6) & ((1ULL << 6) - 1);
	out[1]     = tmp_0;
	tmp_0      = (register_0 >> 12) & ((1ULL << 6) - 1);
	out[2]     = tmp_0;
	tmp_0      = (register_0 >> 18) & ((1ULL << 6) - 1);
	out[3]     = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 6) - 1);
	out[4]     = tmp_0;
	tmp_0      = (register_0 >> 30) & ((1ULL << 2) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 32);
	tmp_0 |= ((register_0) & ((1ULL << 4) - 1)) << 2;
	out[5]     = tmp_0;
	tmp_0      = (register_0 >> 4) & ((1ULL << 6) - 1);
	out[6]     = tmp_0;
	tmp_0      = (register_0 >> 10) & ((1ULL << 6) - 1);
	out[7]     = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 6) - 1);
	out[8]     = tmp_0;
	tmp_0      = (register_0 >> 22) & ((1ULL << 6) - 1);
	out[9]     = tmp_0;
	tmp_0      = (register_0 >> 28) & ((1ULL << 4) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 64);
	tmp_0 |= ((register_0) & ((1ULL << 2) - 1)) << 4;
	out[10]    = tmp_0;
	tmp_0      = (register_0 >> 2) & ((1ULL << 6) - 1);
	out[11]    = tmp_0;
	tmp_0      = (register_0 >> 8) & ((1ULL << 6) - 1);
	out[12]    = tmp_0;
	tmp_0      = (register_0 >> 14) & ((1ULL << 6) - 1);
	out[13]    = tmp_0;
	tmp_0      = (register_0 >> 20) & ((1ULL << 6) - 1);
	out[14]    = tmp_0;
	tmp_0      = (register_0 >> 26) & ((1ULL << 6) - 1);
	out[15]    = tmp_0;
	register_0 = *(in + (0 * 32) + (i * 1) + 96);
	tmp_0      = (register_0) & ((1ULL << 6) - 1);
	out[16]    = tmp_0;
	tmp_0      = (register_0 >> 6) & ((1ULL << 6) - 1);
	out[17]    = tmp_0;
	tmp_0      = (register_0 >> 12) & ((1ULL << 6) - 1);
	out[18]    = tmp_0;
	tmp_0      = (register_0 >> 18) & ((1ULL << 6) - 1);
	out[19]    = tmp_0;
	tmp_0      = (register_0 >> 24) & ((1ULL << 6) - 1);
	out[20]    = tmp_0;
	tmp_0      = (register_0 >> 30) & ((1ULL << 2) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 128);
	tmp_0 |= ((register_0) & ((1ULL << 4) - 1)) << 2;
	out[21]    = tmp_0;
	tmp_0      = (register_0 >> 4) & ((1ULL << 6) - 1);
	out[22]    = tmp_0;
	tmp_0      = (register_0 >> 10) & ((1ULL << 6) - 1);
	out[23]    = tmp_0;
	tmp_0      = (register_0 >> 16) & ((1ULL << 6) - 1);
	out[24]    = tmp_0;
	tmp_0      = (register_0 >> 22) & ((1ULL << 6) - 1);
	out[25]    = tmp_0;
	tmp_0      = (register_0 >> 28) & ((1ULL << 4) - 1);
	register_0 = *(in + (0 * 32) + (i * 1) + 160);
	tmp_0 |= ((register_0) & ((1ULL << 2) - 1)) << 4;
	out[26] = tmp_0;
	tmp_0   = (register_0 >> 2) & ((1ULL << 6) - 1);
	out[27] = tmp_0;
	tmp_0   = (register_0 >> 8) & ((1ULL << 6) - 1);
	out[28] = tmp_0;
	tmp_0   = (register_0 >> 14) & ((1ULL << 6) - 1);
	out[29] = tmp_0;
	tmp_0   = (register_0 >> 20) & ((1ULL << 6) - 1);
	out[30] = tmp_0;
	tmp_0   = (register_0 >> 26) & ((1ULL << 6) - 1);
	out[31] = tmp_0;
}

__device__ __forceinline__ void unpack(const uint32_t* __restrict a_in_p, uint32_t* __restrict a_out_p, uint8_t bw) {
	switch (bw) {
	case 24:
		unpack_24bw_32ow_32crw_1uf(a_in_p, a_out_p);
		break;
	case 4:
		unpack_4bw_32ow_32crw_1uf(a_in_p, a_out_p);
		break;
	case 16:
		unpack_16bw_32ow_32crw_1uf(a_in_p, a_out_p);
		break;
	case 6:
		unpack_6bw_32ow_32crw_1uf(a_in_p, a_out_p);
		break;
	}
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void load_registers(int i, T* out, T* registers) {

#pragma unroll
	for (int j = 0; j < ITEMS_PER_THREAD; j++) {
		out[j * BLOCK_THREADS + i] = registers[j];
	}
}

__global__ void unpack_global(const uint32_t* __restrict in, uint32_t* __restrict out, uint8_t bw) {
	int trd_idx = threadIdx.x;
	int blc_idx = blockIdx.x;
	in          = in + ((blc_idx * bw) << 5);
	out         = out + (blc_idx << 10);

	uint32_t registers[32];
	unpack(in, registers, bw);
	load_registers<uint32_t, 32, 32>(trd_idx, out, registers);
}

__device__ __forceinline__ void unpack_device(const uint32_t* __restrict in, uint32_t* __restrict out, uint8_t bw) {
	unpack(in, out, bw);
}

__device__ __forceinline__ void unpack_device(const int32_t* __restrict in, int32_t* __restrict out, uint8_t bw) {
	unpack(reinterpret_cast<const uint32_t*>(in), reinterpret_cast<uint32_t*>(out), bw);
}
} // namespace hardcoded

namespace NEW_IDEA {

__device__ __forceinline__ void RLE_UNPACK(const uint8_t* __restrict a_in_p, uint8_t* __restrict a_out_p) {

	[[maybe_unused]] auto     out = reinterpret_cast<uint32_t*>(a_out_p);
	[[maybe_unused]] auto     in  = reinterpret_cast<const uint32_t*>(a_in_p);
	[[maybe_unused]] uint32_t register_0;
	[[maybe_unused]] uint32_t tmp_0;
	[[maybe_unused]] uint32_t base_0 = 0ULL;

	int i = threadIdx.x; // THREAD INDEX

	register_0   = *(in + i);
	tmp_0        = (register_0) & (0b00000001000000010000000100000001);
	out[i + 0]   = tmp_0;
	tmp_0        = (register_0 >> 1) & (0b00000001000000010000000100000001);
	out[i + 32]  = tmp_0;
	tmp_0        = (register_0 >> 2) & (0b00000001000000010000000100000001);
	out[i + 64]  = tmp_0;
	tmp_0        = (register_0 >> 3) & (0b00000001000000010000000100000001);
	out[i + 96]  = tmp_0;
	tmp_0        = (register_0 >> 4) & (0b00000001000000010000000100000001);
	out[i + 128] = tmp_0;
	tmp_0        = (register_0 >> 5) & (0b00000001000000010000000100000001);
	out[i + 160] = tmp_0;
	tmp_0        = (register_0 >> 6) & (0b00000001000000010000000100000001);
	out[i + 192] = tmp_0;
	tmp_0        = (register_0 >> 7) & (0b00000001000000010000000100000001);
	out[i + 224] = tmp_0;
}

__device__ __forceinline__ void SIMDIZED_RSUM_8(const uint8_t* a_in_p, uint8_t* a_out_p, const uint8_t* a_base_p) {
	[[maybe_unused]] auto out  = reinterpret_cast<uint32_t*>(a_out_p);
	[[maybe_unused]] auto in   = reinterpret_cast<const uint32_t*>(a_in_p);
	[[maybe_unused]] auto base = reinterpret_cast<const uint32_t*>(a_base_p);

	uint32_t i = threadIdx.x;
	i          = i % 32;
	uint32_t r_0;
	uint32_t r_1;

	r_0          = *(in + i + 0);
	r_1          = *(base + i + 0);
	r_1          = r_1 + r_0;
	out[i + 0]   = r_1;
	r_0          = *(in + i + 32);
	r_1          = r_1 + r_0;
	out[i + 32]  = r_1;
	r_0          = *(in + i + 64);
	r_1          = r_1 + r_0;
	out[i + 64]  = r_1;
	r_0          = *(in + i + 96);
	r_1          = r_1 + r_0;
	out[i + 96]  = r_1;
	r_0          = *(in + i + 128);
	r_1          = r_1 + r_0;
	out[i + 128] = r_1;
	r_0          = *(in + i + 160);
	r_1          = r_1 + r_0;
	out[i + 160] = r_1;
	r_0          = *(in + i + 192);
	r_1          = r_1 + r_0;
	out[i + 192] = r_1;
	r_0          = *(in + i + 224);
	r_1          = r_1 + r_0;
	out[i + 224] = r_1;
}

} // namespace NEW_IDEA