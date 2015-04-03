cmake_minimum_required(VERSION 2.8)

project(queuing)

include(../../cmake/opencl/common.cmake)
include(../../cmake/opencl/queues.cmake)

#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-g;-G;-O0;")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")

SET(TEST_SOURCES
  proc0.cuh
  proc1.cuh
  proc2.cuh
  test.cu
  main.cpp
)

add_executable(queuing ${GENERAL} ${queues_SOURCES} ${TEST_SOURCES})
target_link_libraries(queuing ${OPENCL_LIBRARIES}) 
