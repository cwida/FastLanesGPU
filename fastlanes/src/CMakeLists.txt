add_library(fastlanes_gpu
        SHARED
        fastlanes_gpu.cpp
        pack.cpp
        transpose.cpp
        unrsum.cpp)

set_property(TARGET fastlanes_gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(fastlanes_gpu PUBLIC include crystal crystal-opt)

add_executable(compress_ssb ssb/compress_ssb.cu)
target_link_libraries(compress_ssb PUBLIC fastlanes_gpu gtest_main)

add_executable(compress_ssb_sorted ssb/compress_ssb_sorted.cu)
target_link_libraries(compress_ssb_sorted PUBLIC fastlanes_gpu gtest_main)

add_executable(fls_q11 ssb/fls_q11.cu)
target_link_libraries(fls_q11 PUBLIC fastlanes_gpu gtest_main)

add_executable(fls_q21 ssb/fls_q21.cu)
target_link_libraries(fls_q21 PUBLIC fastlanes_gpu gtest_main)

add_executable(fls_q21_bitpacked_opt_v4 ssb/fls_q21_bitpacked_opt_v4.cu)
target_link_libraries(fls_q21_bitpacked_opt_v4 PUBLIC fastlanes_gpu gtest_main)

add_executable(fls_q31 ssb/fls_q31.cu)
target_link_libraries(fls_q31 PUBLIC fastlanes_gpu gtest_main)

add_executable(fls_q41 ssb/fls_q41.cu)
target_link_libraries(fls_q41 PUBLIC fastlanes_gpu gtest_main)

add_executable(fls_q31_bitpacked_opt_v5 ssb/fls_q31_bitpacked_opt_v5.cu)
target_link_libraries(fls_q31_bitpacked_opt_v5 PUBLIC fastlanes_gpu gtest_main)

add_executable(fls_q41_bitpacked_opt_v3 ssb/fls_q41_bitpacked_opt_v3.cu)
target_link_libraries(fls_q41_bitpacked_opt_v3 PUBLIC fastlanes_gpu gtest_main)

add_executable(fls_q41_bitpacked_opt_v4 ssb/fls_q41_bitpacked_opt_v4.cu)
target_link_libraries(fls_q41_bitpacked_opt_v4 PUBLIC fastlanes_gpu gtest_main)

add_executable(fls_q11_bitpacked_opt_v3 ssb/fls_q11_bitpacked_opt_v3.cu)
target_link_libraries(fls_q11_bitpacked_opt_v3 PUBLIC fastlanes_gpu gtest_main)

add_executable(fls_q11_bitpacked_opt_v4 ssb/fls_q11_bitpacked_opt_v4.cu)
target_link_libraries(fls_q11_bitpacked_opt_v4 PUBLIC fastlanes_gpu gtest_main)

add_executable(bitpack_shared_memory bitpack_shared_memory.cu)
target_link_libraries(bitpack_shared_memory PRIVATE fastlanes_gpu gtest_main)

add_executable(bitpack_register bitpack_register.cu)
target_link_libraries(bitpack_register PRIVATE fastlanes_gpu gtest_main)

add_executable(delta_shared_memory delta_shared_memory.cu)
target_link_libraries(delta_shared_memory PRIVATE fastlanes_gpu gtest_main)

add_executable(delta_global_memory delta_global_memory.cu)
target_link_libraries(delta_global_memory PRIVATE fastlanes_gpu gtest_main)