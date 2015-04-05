cmake_minimum_required(VERSION 2.8)

project(matmul)

# Block size for shared block matrix multiplication
add_definitions(-DBLOCK_SIZE=32)

include(../../../cmake/opencl/common.cmake)
include(../../../cmake/opencl/queues.cmake)

set(SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/..")

SET(TEST_SOURCES
  ${SOURCE_DIR}/matmul.cu
)

add_executable(matmul ${GENERAL} ${queues_SOURCES} ${TEST_SOURCES})
target_link_libraries(matmul ${OPENCL_LIBRARIES})

# TODO add clAmdBlas to target
