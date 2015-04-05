cmake_minimum_required(VERSION 2.8)

project(tasman)

# Block size for shared block matrix multiplication
add_definitions(-DBLOCK_SIZE=32)

include(../../../cmake/opencl/common.cmake)
include(../../../cmake/opencl/queues.cmake)

set(SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/..")

SET(TEST_SOURCES
  ${SOURCE_DIR}/DynamicTaskManager.cu
  ${SOURCE_DIR}/DynamicTaskManagerInternal.cu
  ${SOURCE_DIR}/matmul.cu
)

add_executable(tasman ${GENERAL} ${queues_SOURCES} ${TEST_SOURCES})
target_link_libraries(tasman ${OPENCL_LIBRARIES})

# TODO add clAmdBlas to target
