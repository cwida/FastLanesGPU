add_library(cuda_normal_t32_1024_uf1_unpack OBJECT
        cuda_normal_t32_1024_uf1_unpack_src.cu)
target_compile_definitions(cuda_normal_t32_1024_uf1_unpack PRIVATE IS_SCALAR)

target_compile_options(cuda_normal_t32_1024_uf1_unpack PUBLIC ${FLAG})
cmake_print_properties(TARGETS cuda_normal_t32_1024_uf1_unpack
        PROPERTIES COMPILE_DEFINITIONS
        PROPERTIES COMPILE_OPTIONS)
LIST(APPEND FLS_GENERATED_OBJECT_FILES
        $<TARGET_OBJECTS:cuda_normal_t32_1024_uf1_unpack>)
get_target_property(TARGET_NAME cuda_normal_t32_1024_uf1_unpack NAME)
get_target_property(TARGET_COMPILE_OPTIONS cuda_normal_t32_1024_uf1_unpack COMPILE_OPTIONS)
#------------------------------------------------------------------------------------------------------
add_executable(cuda_normal_t32_1024_uf1_unpack_test cuda_normal_t32_1024_uf1_unpack_test.cu)
target_link_libraries(cuda_normal_t32_1024_uf1_unpack_test PRIVATE cuda_normal_t32_1024_uf1_unpack)
target_link_libraries(cuda_normal_t32_1024_uf1_unpack_test PRIVATE gtest_main fastlanes_gpu)
#------------------------------------------------------------------------------------------------------
add_executable(cuda_normal_t32_1024_uf1_unpack_bench cuda_normal_t32_1024_uf1_unpack_bench.cu)
target_link_libraries(cuda_normal_t32_1024_uf1_unpack_bench PRIVATE cuda_normal_t32_1024_uf1_unpack fastlanes_gpu)
