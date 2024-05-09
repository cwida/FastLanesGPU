#pragma once

#include <cstdint>
#include <thrust/detail/cstdint.h>

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void InitFlags(int (&selection_flags)[ITEMS_PER_THREAD]) {
#pragma unroll
	for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
		selection_flags[ITEM] = 1;
	}
}

template <typename T, typename ST, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredDirect(int tid, T (&items)[ITEMS_PER_THREAD], SelectOp select_op, ST (&selection_flags)[ITEMS_PER_THREAD]) {
#pragma unroll
	for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
		selection_flags[ITEM] = select_op(items[ITEM]);
	}
}

template <typename T, typename ST, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredDirect(
    int tid, T (&items)[ITEMS_PER_THREAD], SelectOp select_op, ST (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
#pragma unroll
	for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
		if (tid + (ITEM * BLOCK_THREADS) < num_items) { selection_flags[ITEM] = select_op(items[ITEM]); }
	}
}

template <typename T, typename ST, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPred(T (&items)[ITEMS_PER_THREAD], SelectOp select_op, ST (&selection_flags)[ITEMS_PER_THREAD], int num_items) {

	if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
		BlockPredDirect<T, ST, SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(
		    threadIdx.x, items, select_op, selection_flags);
	} else {
		BlockPredDirect<T, ST, SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(
		    threadIdx.x, items, select_op, selection_flags, num_items);
	}
}

template <typename T, typename ST, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredAndDirect(int tid, T (&items)[ITEMS_PER_THREAD], SelectOp select_op, ST (&selection_flags)[ITEMS_PER_THREAD]) {
#pragma unroll
	for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
		selection_flags[ITEM] = selection_flags[ITEM] && select_op(items[ITEM]);
	}
}

template <typename T, typename ST, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndDirect(
    int tid, T (&items)[ITEMS_PER_THREAD], SelectOp select_op, ST (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
#pragma unroll
	for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
		if (tid + (ITEM * BLOCK_THREADS) < num_items) {
			selection_flags[ITEM] = selection_flags[ITEM] && select_op(items[ITEM]);
		}
	}
}

template <typename T, typename ST, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredAnd(T (&items)[ITEMS_PER_THREAD], SelectOp select_op, ST (&selection_flags)[ITEMS_PER_THREAD], int num_items) {

	if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
		BlockPredAndDirect<T, ST, SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(
		    threadIdx.x, items, select_op, selection_flags);
	} else {
		BlockPredAndDirect<T, ST, SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(
		    threadIdx.x, items, select_op, selection_flags, num_items);
	}
}

template <typename T, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredOrDirect(int tid, T (&items)[ITEMS_PER_THREAD], SelectOp select_op, int (&selection_flags)[ITEMS_PER_THREAD]) {
#pragma unroll
	for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
		selection_flags[ITEM] = selection_flags[ITEM] || select_op(items[ITEM]);
	}
}

template <typename T, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredOrDirect(int      tid,
                                                  T        (&items)[ITEMS_PER_THREAD],
                                                  SelectOp select_op,
                                                  int      (&selection_flags)[ITEMS_PER_THREAD],
                                                  int      num_items) {
#pragma unroll
	for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
		if (tid + (ITEM * BLOCK_THREADS) < num_items) {
			selection_flags[ITEM] = selection_flags[ITEM] || select_op(items[ITEM]);
		}
	}
}

template <typename T, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredOr(T (&items)[ITEMS_PER_THREAD], SelectOp select_op, int (&selection_flags)[ITEMS_PER_THREAD], int num_items) {

	if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
		BlockPredOrDirect<T, SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, select_op, selection_flags);
	} else {
		BlockPredOrDirect<T, SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(
		    threadIdx.x, items, select_op, selection_flags, num_items);
	}
}

template <typename T>
struct LessThan {
	T compare;

	__device__ __forceinline__ LessThan(T compare)
	    : compare(compare) {}

	__device__ __forceinline__ bool operator()(const T& a) const { return (a < compare); }
};

template <typename T>
struct GreaterThan {
	T compare;

	__device__ __forceinline__ GreaterThan(T compare)
	    : compare(compare) {}

	__device__ __forceinline__ bool operator()(const T& a) const { return (a > compare); }
};

template <typename T>
struct LessThanEq {
	T compare;

	__device__ __forceinline__ LessThanEq(T compare)
	    : compare(compare) {}

	__device__ __forceinline__ bool operator()(const T& a) const { return (a <= compare); }
};

template <typename T>
struct GreaterThanEq {
	T compare;

	__device__ __forceinline__ GreaterThanEq(T compare)
	    : compare(compare) {}

	__device__ __forceinline__ bool operator()(const T& a) const { return (a >= compare); }
};

template <typename T>
struct Eq {
	T compare;

	__device__ __forceinline__ Eq(T compare)
	    : compare(compare) {}

	__device__ __forceinline__ bool operator()(const T& a) const { return (a == compare); }
};

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredLT(T (&items)[ITEMS_PER_THREAD], T compare, int (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
	LessThan<T> select_op(compare);
	BlockPred<T, LessThan<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template <typename T, typename ST, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredAndLT(T (&items)[ITEMS_PER_THREAD], T compare, ST (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
	LessThan<T> select_op(compare);
	BlockPredAnd<T, ST, LessThan<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template <typename T, typename ST, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredGT(T (&items)[ITEMS_PER_THREAD], T compare, ST (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
	GreaterThan<T> select_op(compare);
	BlockPred<T, ST, GreaterThan<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredAndGT(T (&items)[ITEMS_PER_THREAD], T compare, int (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
	GreaterThan<T> select_op(compare);
	BlockPredAnd<T, GreaterThan<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredLTE(T (&items)[ITEMS_PER_THREAD], T compare, int (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
	LessThanEq<T> select_op(compare);
	BlockPred<T, LessThanEq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template <typename T, typename ST, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredAndLTE(T (&items)[ITEMS_PER_THREAD], T compare, ST (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
	LessThanEq<T> select_op(compare);
	BlockPredAnd<T, ST, LessThanEq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredGTE(T (&items)[ITEMS_PER_THREAD], T compare, int (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
	GreaterThanEq<T> select_op(compare);
	BlockPred<T, GreaterThanEq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template <typename T, typename ST, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredAndGTE(T (&items)[ITEMS_PER_THREAD], T compare, ST (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
	GreaterThanEq<T> select_op(compare);
	BlockPredAnd<T, ST, GreaterThanEq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, select_op, selection_flags, num_items);
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredEQ(T (&items)[ITEMS_PER_THREAD], T compare, int (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
	Eq<T> select_op(compare);
	BlockPred<T, Eq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredAndEQ(T (&items)[ITEMS_PER_THREAD], T compare, T (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
	Eq<T> select_op(compare);
	BlockPredAnd<T, Eq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredOrEQ(T (&items)[ITEMS_PER_THREAD], T compare, int (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
	Eq<T> select_op(compare);
	BlockPredOr<T, Eq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

/* SIMD */

constexpr uint32_t make_simd_const(uint16_t a_compare) {
	uint32_t compare = 0;
	compare          = a_compare;
	compare          = compare << 16;
	compare          = compare | a_compare;
	return compare;
}

/*
 * GreaterThan_int_16_2
 */

struct GreaterThan_int_16_2 {
	uint32_t compare;

	__device__ __forceinline__ GreaterThan_int_16_2(uint32_t a_compare)
	    : compare(a_compare) {}

	__device__ __forceinline__ uint32_t operator()(const uint32_t& a) const {
		// return _vcmpgts2(a, compare);
		auto result = __vcmpgts2(a, compare);

		result = result & 0b00000000000000010000000000000001; // todo
		// printf("%u\n", result);
		return result;
	}
};

struct LessThan_int_16_2 {
	uint32_t compare;

	__device__ __forceinline__ LessThan_int_16_2(uint32_t a_compare)
	    : compare(a_compare) {}

	__device__ __forceinline__ uint32_t operator()(const uint32_t& a) const {
		// return _vcmpgts2(a, compare);
		return __vcmplts2(a, compare);
	}
};

template <typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredDirect_int_16_2(int      tid,
                                                         uint32_t (&items)[ITEMS_PER_THREAD],
                                                         SelectOp select_op,
                                                         uint32_t (&selection_flags)[ITEMS_PER_THREAD],
                                                         int      num_items) {
#pragma unroll
	for (int ITEM = 0; ITEM < ITEMS_PER_THREAD / 2; ITEM++) {
		if (tid + (ITEM * BLOCK_THREADS) < num_items / 2) { selection_flags[ITEM] = select_op(items[ITEM]); }
	}
}

template <typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredDirect_int_16_2(int      tid,
                                                         uint32_t (&items)[ITEMS_PER_THREAD],
                                                         SelectOp select_op,
                                                         uint32_t (&selection_flags)[ITEMS_PER_THREAD]) {
#pragma unroll
	for (int ITEM = 0; ITEM < ITEMS_PER_THREAD / 2; ITEM++) {
		selection_flags[ITEM] = select_op(items[ITEM]);
	}
}

template <typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredDirect_int_16_2(uint32_t (&items)[ITEMS_PER_THREAD],
                                                         SelectOp select_op,
                                                         uint32_t (&selection_flags)[ITEMS_PER_THREAD],
                                                         int      num_items) {

	if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
		BlockPredDirect_int_16_2<SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(
		    threadIdx.x, items, select_op, selection_flags);
	} else {
		BlockPredDirect_int_16_2<SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(
		    threadIdx.x, items, select_op, selection_flags, num_items);
	}
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredGT_int_16_2(uint32_t (&items)[ITEMS_PER_THREAD],
                                                     uint32_t compare,
                                                     uint32_t (&selection_flags)[ITEMS_PER_THREAD],
                                                     int      num_items) {
	GreaterThan_int_16_2 select_op(compare);
	BlockPredDirect_int_16_2<GreaterThan_int_16_2, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, select_op, selection_flags, num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredLT_int_16_2(uint32_t (&items)[ITEMS_PER_THREAD],
                                                     uint32_t compare,
                                                     uint32_t (&selection_flags)[ITEMS_PER_THREAD],
                                                     int      num_items) {
	LessThan_int_16_2 select_op(compare);
	BlockPredDirect_int_16_2<LessThan_int_16_2, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    items, select_op, selection_flags, num_items);
}

template <typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPred_int_16_2(uint32_t (&items)[ITEMS_PER_THREAD],
                                                   SelectOp select_op,
                                                   uint32_t (&selection_flags)[ITEMS_PER_THREAD],
                                                   int      num_items) {

	if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
		BlockPredDirect_int_16_2<SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(
		    threadIdx.x, items, select_op, selection_flags);
	} else {
		BlockPredDirect_int_16_2<SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(
		    threadIdx.x, items, select_op, selection_flags, num_items);
	}
}

struct LessThanX {
	uint32_t compare;

	__device__ __forceinline__ LessThanX(uint32_t a_compare)
	    : compare(a_compare) {}

	__device__ __forceinline__ uint32_t operator()(const uint32_t& a) const {
		// return _vcmpgts2(a, compare);
		return __vcmplts2(a, compare);
	}
};

struct LessThanEqX {
	uint32_t compare;

	__device__ __forceinline__ LessThanEqX(uint32_t a_compare)
	    : compare(a_compare) {}

	__device__ __forceinline__ uint32_t operator()(const uint32_t& a) const {
		// return _vcmpgts2(a, compare);
		return __vcmpleu2(a, compare);
	}
};

struct GreaterThanEqX {
	uint32_t compare;

	__device__ __forceinline__ GreaterThanEqX(uint32_t a_compare)
	    : compare(a_compare) {}

	__device__ __forceinline__ uint32_t operator()(const uint32_t& a) const {
		// return _vcmpgts2(a, compare);
		return __vcmpgeu2(a, compare);
	}
};

template <typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndDirectX(int      tid,
                                                    uint32_t (&items)[ITEMS_PER_THREAD],
                                                    SelectOp select_op,
                                                    uint32_t (&selection_flags)[ITEMS_PER_THREAD]) {
	// printf("not mini\n");

#pragma unroll
	for (int ITEM = 0; ITEM < ITEMS_PER_THREAD / 2; ITEM++) {
		selection_flags[ITEM] = selection_flags[ITEM] & select_op(items[ITEM]);
	}
}

template <typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndDirectX(int      tid,
                                                    uint32_t (&items)[ITEMS_PER_THREAD],
                                                    SelectOp select_op,
                                                    uint32_t (&selection_flags)[ITEMS_PER_THREAD],
                                                    int      num_items) {
	// printf("mini\n");
#pragma unroll
	for (int ITEM = 0; ITEM < ITEMS_PER_THREAD / 2; ITEM++) {
		if (tid + (ITEM * BLOCK_THREADS) < num_items / 2 + 1) {
			selection_flags[ITEM] = selection_flags[ITEM] & select_op(items[ITEM]);
		}
	}
}

template <typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndX(uint32_t (&items)[ITEMS_PER_THREAD],
                                              SelectOp select_op,
                                              uint32_t (&selection_flags)[ITEMS_PER_THREAD],
                                              int      num_items) {

	if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
		BlockPredAndDirectX<SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, select_op, selection_flags);
	} else {
		BlockPredAndDirectX<SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(
		    threadIdx.x, items, select_op, selection_flags, num_items);
	}
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndLTX(uint32_t (&items)[ITEMS_PER_THREAD],
                                                uint32_t compare,
                                                uint32_t (&selection_flags)[ITEMS_PER_THREAD],
                                                int      num_items) {
	LessThanX select_op(compare);
	BlockPredAndX<LessThanX, BLOCK_THREADS, ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndLTEX(uint32_t (&items)[ITEMS_PER_THREAD],
                                                 uint32_t compare,
                                                 uint32_t (&selection_flags)[ITEMS_PER_THREAD],
                                                 int      num_items) {
	LessThanEqX select_op(compare);
	BlockPredAndX<LessThanEqX, BLOCK_THREADS, ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndGTEX(uint32_t (&items)[ITEMS_PER_THREAD],
                                                 uint32_t compare,
                                                 uint32_t (&selection_flags)[ITEMS_PER_THREAD],
                                                 int      num_items) {
	GreaterThanEqX select_op(compare);
	BlockPredAndX<GreaterThanEqX, BLOCK_THREADS, ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}
