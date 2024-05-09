#pragma once

template <typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeDirectAndPHT_2_R_S(int tid,
                                                             K   (&keys)[ITEMS_PER_THREAD],
                                                             V   (&res)[BLOCK_THREADS * ITEMS_PER_THREAD],
                                                             int (&selection_flags)[ITEMS_PER_THREAD],
                                                             K*  ht,
                                                             int ht_len,
                                                             K   keys_min) {
#pragma unroll
	for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
		auto shared_idx = BLOCK_THREADS * ITEM + threadIdx.x;

		if (selection_flags[ITEM]) {
			int hash = HASH(keys[ITEM], ht_len, keys_min);

			uint64_t slot = *reinterpret_cast<uint64_t*>(&ht[hash << 1]);
			if (slot != 0) {
				res[shared_idx] = (slot >> 32);
			} else {
				selection_flags[ITEM] = 0;
			}
		}
	}
}

template <typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeDirectAndPHT_2_R_S(int tid,
                                                             K   (&items)[ITEMS_PER_THREAD],
                                                             V   (&res)[BLOCK_THREADS * ITEMS_PER_THREAD],
                                                             int (&selection_flags)[ITEMS_PER_THREAD],
                                                             K*  ht,
                                                             int ht_len,
                                                             K   keys_min,
                                                             int num_items) {
#pragma unroll
	for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
		auto shared_idx = BLOCK_THREADS * ITEM + threadIdx.x;

		if (tid + (ITEM * BLOCK_THREADS) < num_items) {
			if (selection_flags[ITEM]) {
				int hash = HASH(items[ITEM], ht_len, keys_min);

				uint64_t slot = *reinterpret_cast<uint64_t*>(&ht[hash << 1]);
				if (slot != 0) {
					res[shared_idx] = (slot >> 32);
				} else {
					selection_flags[ITEM] = 0;
				}
			}
		}
	}
}

template <typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeAndPHT_2_R_S(K   (&keys)[ITEMS_PER_THREAD],
                                                       V   (&res)[BLOCK_THREADS * ITEMS_PER_THREAD],
                                                       int (&selection_flags)[ITEMS_PER_THREAD],
                                                       K*  ht,
                                                       int ht_len,
                                                       K   keys_min,
                                                       int num_items) {

	if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
		BlockProbeDirectAndPHT_2_R_S<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(
		    threadIdx.x, keys, res, selection_flags, ht, ht_len, keys_min);
	} else {
		BlockProbeDirectAndPHT_2_R_S<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(
		    threadIdx.x, keys, res, selection_flags, ht, ht_len, keys_min, num_items);
	}
}

template <typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeAndPHT_2_R_S(K   (&keys)[ITEMS_PER_THREAD],
                                                       V   (&res)[BLOCK_THREADS * ITEMS_PER_THREAD],
                                                       int (&selection_flags)[ITEMS_PER_THREAD],
                                                       K*  ht,
                                                       int ht_len,
                                                       int num_items) {
	BlockProbeAndPHT_2_R_S<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, res, selection_flags, ht, ht_len, 0, num_items);
}