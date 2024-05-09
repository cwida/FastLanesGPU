#pragma once

#include <common.cuh>
#include <fstream>
#include <memory>
#include <span>
#include <ssb_utils.h>
#include <string>

class Dir {
public:
	idx_t       id;
	std::string file_path;
};

using namespace std;

#define STATS(COL)                                                                                                     \
	{                                                                                                                  \
		int32_t lo_##COL##_min = h_lo_##COL[0];                                                                        \
		int32_t lo_##COL##_max = h_lo_##COL[0];                                                                        \
		for (size_t i {0}; i < hard_coded.n_tup_line_order; ++i) {                                                     \
			lo_##COL##_min = std::min(lo_##COL##_min, h_lo_##COL[i]);                                                  \
			lo_##COL##_max = std::max(lo_##COL##_max, h_lo_##COL[i]);                                                  \
		}                                                                                                              \
		FLS_SHOW(lo_##COL##_min)                                                                                       \
		FLS_SHOW(lo_##COL##_max)                                                                                       \
		uint16_t x = RANGE_BIT(lo_##COL##_max - lo_##COL##_min);                                                       \
		FLS_SHOW(x)                                                                                                    \
	}

namespace fastlanes::ssb {

#define SF 10

#if SF == 10
class SSB {
public:
	const uint64_t n_tup_line_order;
	const string   name;
	const Dir      dir;

	static constexpr int32_t lo_orderdate_min = 19920101;
	static constexpr int32_t lo_orderdate_max = 19980802;
	static constexpr uint8_t lo_orderdate_bw  = 16;

	static constexpr int32_t lo_extendedprice_min = 90100;
	static constexpr int32_t lo_extendedprice_max = 10494950;
	static constexpr uint8_t lo_extendedprice_bw  = 24;

	static constexpr int32_t lo_quantity_min = 1;
	static constexpr int32_t lo_quantity_max = 50;
	static constexpr uint8_t lo_quantity_bw  = 6;

	static constexpr int32_t lo_discount_min = 1;
	static constexpr int32_t lo_discount_max = 10;
	static constexpr uint8_t lo_discount_bw  = 4;

	static constexpr int32_t lo_partkey_min = 1;
	static constexpr int32_t lo_partkey_max = 200000;
	static constexpr uint8_t lo_partkey_bw  = 20;

	static constexpr int32_t lo_suppkey_min       = 1;
	static constexpr int32_t lo_suppkey_max       = 2000;
	static constexpr uint8_t lo_real_suppkey_bw   = 15;
	static constexpr uint8_t lo_chosen_suppkey_bw = 16;

	static constexpr int32_t lo_revenue_min = 81360;
	static constexpr int32_t lo_revenue_max = 10474950;
	static constexpr uint8_t lo_revenue_bw  = 24;

	static constexpr int32_t lo_custkey_min       = 1;
	static constexpr int32_t lo_custkey_max       = 299999;
	static constexpr uint8_t lo_real_custkey_bw   = 19;
	static constexpr uint8_t lo_chosen_custkey_bw = 20;

	static constexpr int32_t lo_supplycost_min       = 54060;
	static constexpr int32_t lo_supplycost_max       = 125939;
	static constexpr uint8_t lo_supplycost_bw        = 17;
	static constexpr uint8_t lo_chosen_supplycost_bw = 20;

	int32_t n_vec;
};

#elif SF == 1
class SSB {
public:
	const uint64_t n_tup_line_order;
	const string   name;
	const Dir      dir;

	static constexpr int32_t lo_orderdate_min = 19920101;
	static constexpr int32_t lo_orderdate_max = 19980802;
	static constexpr uint8_t lo_orderdate_bw  = 16;

	static constexpr int32_t lo_extendedprice_min = 90100;
	static constexpr int32_t lo_extendedprice_max = 10494950;
	static constexpr uint8_t lo_extendedprice_bw  = 24;

	static constexpr int32_t lo_quantity_min = 1;
	static constexpr int32_t lo_quantity_max = 50;
	static constexpr uint8_t lo_quantity_bw  = 6;

	static constexpr int32_t lo_discount_min = 1;
	static constexpr int32_t lo_discount_max = 10;
	static constexpr uint8_t lo_discount_bw  = 4;

	static constexpr int32_t lo_partkey_min = 1;
	static constexpr int32_t lo_partkey_max = 200000;
	static constexpr uint8_t lo_partkey_bw  = 18;

	static constexpr int32_t lo_suppkey_min = 1;
	static constexpr int32_t lo_suppkey_max = 2000;
	static constexpr uint8_t lo_suppkey_bw  = 11;

	static constexpr int32_t lo_revenue_min = 81360;
	static constexpr int32_t lo_revenue_max = 10474950;
	static constexpr uint8_t lo_revenue_bw  = 24;

	static constexpr int32_t lo_custkey_min = 1;
	static constexpr int32_t lo_custkey_max = 29999;
	static constexpr uint8_t lo_custkey_bw  = 15;

	static constexpr int32_t lo_supplycost_min = 54060;
	static constexpr int32_t lo_supplycost_max = 125939;
	static constexpr uint8_t lo_supplycost_bw  = 17;

	int32_t n_vec;
};
#endif

namespace sample_data::ssb_0_1 {
inline std::string relative_path = std::string("/sample-data/ssb/sf_") + std::string("0_1") + std::string("/");
inline std::string path          = CMAKE_SOURCE_DIR + relative_path;
inline std::string table_path    = path + "tables/";
inline std::string result_path   = path + "result/";
inline std::string binary_path   = path + "binary/";
inline Dir         customer_dir {0, table_path + "customer/customer.tbl"};
inline Dir         lineorder_dir {1, table_path + "lineorder/lineorder.tbl"};
inline Dir         part_dir {2, table_path + "part/part.tbl"};
inline Dir         supplier_dir {3, table_path + "supplier/supplier.tbl"};

} // namespace sample_data::ssb_0_1

namespace sample_data::ssb_1 {
inline std::string relative_path = std::string("/gpu/data/ssb/data/s1") + std::string("/");
inline std::string path          = CMAKE_SOURCE_DIR + relative_path;
inline std::string table_path    = path;
inline std::string result_path   = path + "result/";
inline std::string binary_path   = path + "binary/";
inline Dir         customer_dir {0, table_path + "customer.tbl"};
inline Dir         lineorder_dir {1, table_path + "lineorder.tbl"};
inline Dir         part_dir {2, table_path + "part.tbl"};
inline Dir         supplier_dir {3, table_path + "supplier.tbl"};
} // namespace sample_data::ssb_1

namespace sample_data::ssb_10 {
inline std::string relative_path = std::string("/gpu/data/ssb/data/s10") + std::string("/");
inline std::string path          = CMAKE_SOURCE_DIR + relative_path;
inline std::string table_path    = path;
inline std::string result_path   = path + "result/";
inline std::string binary_path   = path + "binary/";
inline Dir         customer_dir {0, table_path + "customer.tbl"};
inline Dir         lineorder_dir {1, table_path + "lineorder.tbl"};
inline Dir         part_dir {2, table_path + "part.tbl"};
inline Dir         supplier_dir {3, table_path + "supplier.tbl"};
}; // namespace sample_data::ssb_10

inline SSB ssb_0_1 {600597, "SF_0_1", sample_data::ssb_0_1::lineorder_dir, 587};
inline SSB ssb_1 {6001171, "SF_1", sample_data::ssb_1::lineorder_dir, 5861};
inline SSB ssb_10 {59986214, "SF_10", sample_data::ssb_10::lineorder_dir, 58581};

struct SSBQuery1 {
	const uint64_t result;
	const SSB&     ssb;
};

inline SSBQuery1 ssb_q11_0_1 {41307262627, ssb_0_1};
inline SSBQuery1 ssb_q11_10 {4468236714181, ssb_10};

inline SSBQuery1 ssb_q11_1 {446268068091, ssb_1};
inline SSBQuery1 ssb_q12_1 {98314553869, ssb_1};
inline SSBQuery1 ssb_q13_1 {24994512533, ssb_1};

} // namespace fastlanes::ssb

template <int SIZE>
int32_t find_base(const int32_t in[]) {
	auto result = *std::min_element(in, in + SIZE);
	return result;
}

template <int SIZE, typename T>
T find_max(T in[]) {
	auto result = *std::max_element(in, in + SIZE);
	return result;
}

template <int SIZE>
void subtract_base(int32_t in[], const int32_t base) {
	for (size_t i {0}; i < SIZE; ++i) {
		if (in[i] < base) { throw std::runtime_error("base is the minimum!"); }
		in[i] = in[i] - base;
	}
}

template <int SIZE>
void set_zero_after(int32_t in[], int32_t cur_idx) {
	for (int32_t i {cur_idx}; i < SIZE; ++i) {
		in[i] = 0;
	}
}

inline double average(const int32_t* arr, const n_t n) {
	int sum {0};
	for (size_t i {0}; i < n; ++i) {
		sum += arr[i];
	}
	return static_cast<double>(sum) / n;
}

template <int SIZE>
int32_t find_bw(int32_t in[]) {
	auto* in_p = reinterpret_cast<uint32_t*>(in);
	auto  max  = find_max<SIZE, uint32_t>(in_p);
	auto  bw   = std::bit_width(max);

	if (bw == 0) { return 0; }
	if (bw <= 4) { return 4; }
	if (bw <= 8) { return 8; }
	if (bw <= 12) { return 12; }
	if (bw <= 16) { return 16; }
	if (bw <= 20) { return 20; }
	if (bw <= 24) { return 24; }
	if (bw <= 28) { return 28; }
	if (bw <= 32) { return 32; }

	return bw;
}

template <typename T>
bool is_sorted(const T* arr, int n) {

	for (size_t i {0}; i < n - 1; ++i) {
		if (arr[i] > arr[i + 1]) { return false; }
	}
	return true;
}