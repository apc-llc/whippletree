cmake_minimum_required(VERSION 2.8)

project(queuing)

include(../../../cmake/cuda/common.cmake)
include(../../../cmake/cuda/queues.cmake)

set(SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/..")

SET(TEST_SOURCES
  ${SOURCE_DIR}/proc0.cuh
  ${SOURCE_DIR}/proc1.cuh
  ${SOURCE_DIR}/proc2.cuh
  ${SOURCE_DIR}/main.cu
)

if(CUDA_BUILD_CC35)
  cuda_add_device_executable(queuing ${GENERAL} ${queues_SOURCES} ${TEST_SOURCES})
else()
  cuda_add_executable(queuing ${GENERAL} ${queues_SOURCES} ${TEST_SOURCES})
endif()
