cmake_minimum_required(VERSION 2.8)

project(queuing)

include(../../cmake/common.cmake)
include(../../queues.cmake)

#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-g;-G;-O0;")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")

SET(TEST_SOURCES
  proc0.cuh
  proc1.cuh
  proc2.cuh
  test.cu
  main.cpp
)

if(CUDA_BUILD_CC35)
  cuda_add_device_executable(queuing ${GENERAL} ${queues_SOURCES} ${TEST_SOURCES})
else()
  cuda_add_executable(queuing ${GENERAL} ${queues_SOURCES} ${TEST_SOURCES})
endif()
