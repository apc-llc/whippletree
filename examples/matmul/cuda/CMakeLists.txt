cmake_minimum_required(VERSION 2.8)

project(matmul)

# Block size for shared block matrix multiplication
add_definitions(-DBLOCK_SIZE=32)

include(../../../cmake/cuda/common.cmake)
include(../../../cmake/cuda/queues.cmake)

set(SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/..")

SET(TEST_SOURCES
  ${SOURCE_DIR}/matmul.cu
)

if(CUDA_BUILD_CC35)
  cuda_add_device_executable(matmul ${GENERAL} ${queues_SOURCES} ${TEST_SOURCES})
else()
  cuda_add_executable(matmul ${GENERAL} ${queues_SOURCES} ${TEST_SOURCES})
endif()

cuda_add_cublas_to_target(matmul)
