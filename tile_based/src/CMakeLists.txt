add_library(tilebased STATIC tile_based.cu)
target_include_directories(tilebased PUBLIC include)

# RLE : ----------------------------------------------------------------------------------------------------------------
#add_executable(tile_based_binpack_query_11 tile_based_binpack_query_11.cu)
#target_link_libraries(tile_based_binpack_query_11 PUBLIC fastlanes tilebased)

# FOR : ----------------------------------------------------------------------------------------------------------------
#add_executable(tile_based_binpack tile_based_binpack.cu)
#target_include_directories(tile_based_binpack PUBLIC include)

#add_executable(tile_based_bench_bitpack tile_based_bench_bitpack.cu)
#target_link_libraries(tile_based_bench_bitpack PUBLIC tilebased)

add_executable(tile_based_bench_bp_sum tile_based_bench_bp_sum.cu)
target_link_libraries(tile_based_bench_bp_sum PUBLIC tilebased)

#add_executable(tile_based_bitpack_shared_memory tile_based_bitpack_shared_memory.cu)
#target_link_libraries(tile_based_bitpack_shared_memory PRIVATE fastlanes gtest_main fastlanes_gpu tilebased)

# Delta : --------------------------------------------------------------------------------------------------------------
add_executable(tile_based_bench_delta tile_based_bench_delta.cu)
target_link_libraries(tile_based_bench_delta PUBLIC tilebased)

#add_executable(tile_based_bench_delta_sum tile_based_bench_delta_sum.cu)
#target_link_libraries(tile_based_bench_delta_sum PUBLIC tilebased)

# RLE : ----------------------------------------------------------------------------------------------------------------
#add_executable(test_rle_on_cpu test_rle_on_cpu.cpp)
#target_link_libraries(test_rle_on_cpu PUBLIC tilebased)

#  loadEncodedColumnToGPURLE does not exist.
#add_executable(tile_based_bench_rle tile_based_bench_rle.cu)
#target_link_libraries(tile_based_bench_rle PUBLIC tilebased)

add_executable(rlebinpack rlebinpack.cpp)
target_link_libraries(rlebinpack PUBLIC tilebased)

add_executable(test_match_rle test_match_rle.cu)
target_link_libraries(test_match_rle PUBLIC tilebased)

add_executable(test_perf_rle test_perf_rle.cu)
target_link_libraries(test_perf_rle PUBLIC tilebased)