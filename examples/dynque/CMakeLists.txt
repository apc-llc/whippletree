cmake_minimum_required(VERSION 2.8)

project(dynque)

# Block size for shared block matrix multiplication
add_definitions(-DBLOCK_SIZE=32)

include(../../cmake/common.cmake)
include(../../queues.cmake)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-g;-G;-O0;")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")

SET(TEST_SOURCES
  dynque.cu
)

if(CUDA_BUILD_CC35)
  cuda_add_device_executable(dynque ${GENERAL} ${queues_SOURCES} ${TEST_SOURCES})
else()
  cuda_add_executable(dynque ${GENERAL} ${queues_SOURCES} ${TEST_SOURCES})
endif()

cuda_add_cublas_to_target(dynque)
