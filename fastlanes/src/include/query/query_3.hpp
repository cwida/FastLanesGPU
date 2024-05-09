#ifndef QUERY_3_HPP
#define QUERY_3_HPP

#include "gpu_utils.h"
#include "ssb_utils.h"
#include <vector>

using namespace std;
using namespace fastlanes::gpu;
using namespace fastlanes;

namespace fastlanes::ssb {

struct SSBQuery3ResultRow {
	int                col_0;
	int                col_1;
	int                col_2;
	unsigned long long col_3;

	bool operator==(const SSBQuery3ResultRow& rhs) const {
		return (col_0 == rhs.col_0) && (col_1 == rhs.col_1) && (col_2 == rhs.col_2) && (col_3 == rhs.col_3);
	}
	SSBQuery3ResultRow(int col_0, int col_1, int col_2, unsigned long long col_3)
	    : col_0(col_0)
	    , col_1(col_1)
	    , col_2(col_2)
	    , col_3(col_3) {}
};

std::ostream& operator<<(std::ostream& stream, const SSBQuery3ResultRow& row) {
	stream << "{" << row.col_0 << ", " << row.col_1 << ", " << row.col_2 << ", " << row.col_3 << "}";
	return stream;
}

using SSBQuery3ResultTable = std::vector<SSBQuery3ResultRow>;

struct SSBQuery3 {
	const SSBQuery3ResultTable& reuslt;
	const SSB&                  ssb;
};

} // namespace fastlanes::ssb

#endif // QUERY_3_HPP
