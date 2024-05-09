#ifndef QUERY_2_HPP
#define QUERY_2_HPP

#include "gpu_utils.h"
#include "ssb_utils.h"
#include <vector>

using namespace std;
using namespace fastlanes::gpu;
using namespace fastlanes;

namespace fastlanes::ssb {

struct SSBQuery2ResultRow {
	int                col_0;
	int                col_1;
	unsigned long long col_2;

	bool operator==(const SSBQuery2ResultRow& rhs) const {
		return (col_0 == rhs.col_0) && (col_1 == rhs.col_1) && (col_2 == rhs.col_2);
	}
	SSBQuery2ResultRow(int col_0, int col_1, unsigned long long col_2)
	    : col_0(col_0)
	    , col_1(col_1)
	    , col_2(col_2) {}
};

using SSBQuery2ResultTable = std::vector<SSBQuery2ResultRow>;

struct SSBQuery2 {
	const SSBQuery2ResultTable& reuslt;
	const SSB&                  ssb;
};

} // namespace fastlanes::ssb

#endif // QUERY_2_HPP
