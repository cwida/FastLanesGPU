
# pragma once

__device__ __forceinline__ void d_rsum_32(const uint32_t* in, uint32_t* out, const uint32_t* base) {
	uint32_t trd_idx = threadIdx.x;
	trd_idx          = trd_idx % 32;
	uint32_t r_0;
	uint32_t r_1;

	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 0);
	r_1                                 = base[trd_idx]; // TODO
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 0]   = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 128);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 128] = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 256);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 256] = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 384);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 384] = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 512);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 512] = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 640);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 640] = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 768);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 768] = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 896);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 896] = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 64);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 64]  = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 192);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 192] = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 320);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 320] = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 448);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 448] = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 576);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 576] = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 704);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 704] = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 832);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 832] = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 960);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 960] = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 32);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 32]  = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 160);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 160] = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 288);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 288] = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 416);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 416] = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 544);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 544] = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 672);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 672] = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 800);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 800] = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 928);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 928] = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 96);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 96]  = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 224);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 224] = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 352);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 352] = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 480);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 480] = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 608);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 608] = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 736);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 736] = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 864);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 864] = r_1;
	r_0                                 = *(in + (0 * 32) + (trd_idx * 1) + 992);
	r_1                                 = r_1 + r_0;
	out[(trd_idx * 1) + (0 * 32) + 992] = r_1;
}