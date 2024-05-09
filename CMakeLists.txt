cmake_minimum_required(VERSION 3.22)
project(FastLanesGPU)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# Requirements : -------------------------------------------------------------------------------------------------------
if (NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    message(FATAL_ERROR "Only Clang is supported!")
endif ()
if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13)
    message(FATAL_ERROR "Only Clang >= 13 is supported!")
endif ()

# FLAGS : --------------------------------------------------------------------------------------------------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")
if ("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "x86")
    #    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")  # FSST
endif ()

#-----------------------------------------------------------------------------------------------------------------------
include(FetchContent)
include(CheckCXXCompilerFlag)
include(CMakePrintHelpers)
# https://stackoverflow.com/questions/56089330/cmake-creates-lots-of-targets-i-didnt-specify
set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)
include(CTest)

# GTEST : ------------------------------------------------------------------------------------------------------------
message("---------------------------------------------------------------------------------------------------------")
message("- Building GTEST:")
include(GoogleTest)
# Gtest: -----------------------------------------------------------------------------------------------------------
FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG e2239ee6043f73722e7aa812a459f54a28552929 # release-1.11.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Silence clang-tidy warnings from googletest
set_target_properties(gtest PROPERTIES CXX_CLANG_TIDY "")
set_target_properties(gtest_main PROPERTIES CXX_CLANG_TIDY "")
set_target_properties(gmock PROPERTIES CXX_CLANG_TIDY "")
set_target_properties(gmock_main PROPERTIES CXX_CLANG_TIDY "")

# Definitions: ---------------------------------------------------------------------------------------------------------
add_compile_definitions(CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")


if (${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    message("There is no CUDA on Darwin")
    RETURN()
endif ()

set(CMAKE_CUDA_ARCHITECTURES "native")
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

message("---------------------------------------------------------------------------------------------------------")
message("-- CUDA:")
cmake_print_variables(CUDA_INCLUDE_DIRS)
cmake_print_variables(CUDA_LIBRARIES)
cmake_print_variables(CUDA_FOUND)
cmake_print_variables(CMAKE_CUDA_FLAGS)
cmake_print_variables(CMAKE_CUDA_FLAGS_DEBUG)
cmake_print_variables(CMAKE_CUDA_FLAGS_RELEASE)
cmake_print_variables(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
cmake_print_variables(CMAKE_CUDA_COMPILER)
cmake_print_variables(CMAKE_CUDA_COMPILER_VERSION)
cmake_print_variables(CMAKE_CUDA_STANDARD)
cmake_print_variables(CMAKE_CUDA_STANDARD_REQUIRED)
cmake_print_variables(CMAKE_CXX_STANDARD)
cmake_print_variables(CMAKE_CXX_COMPILER)
cmake_print_variables(CMAKE_CXX_COMPILER_VERSION)
cmake_print_variables(CMAKE_CXX_COMPILER_ID)
cmake_print_variables(CMAKE_SOURCE_DIR)
cmake_print_variables(CMAKE_BUILD_TYPE)

# CUDA : ---------------------------------------------------------------------------------------------------------------
# http://knottsgroup.groups.et.byu.net/labbook/index.php?n=Main.CompilingLAMMPSForGPU
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}
#    -gencode=arch=compute_75,code=sm_75
#    -gencode=arch=compute_75,code=compute_75"
#)

# Include : ------------------------------------------------------------------------------------------------------------
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(include)

# Tools : --------------------------------------------------------------------------------------------------------------
add_subdirectory(tool)

# Crystal : ------------------------------------------------------------------------------------------------------------
add_subdirectory(crystal)

# FastLanes : ----------------------------------------------------------------------------------------------------------
add_subdirectory(fastlanes)

#TileBased : ----------------------------------------------------------------------------------------------------------
add_subdirectory(tile_based)
# try https://github.com/azimafroozeh/gpu-compression

# Crystal-Opt : --------------------------------------------------------------------------------------------------------
add_subdirectory(crystal-opt)

