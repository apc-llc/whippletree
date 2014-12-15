cmake_minimum_required(VERSION 2.8)

project(matmul)




include(../../cmake/common.cmake)
include(../../queues.cmake)

SET(TEST_SOURCES
  matmul.cu
)

if(CUDA_BUILD_CC35)
  cuda_add_device_executable(matmul ${GENERAL} ${queues_SOURCES} ${TEST_SOURCES})
else()
  cuda_add_executable(matmul ${GENERAL} ${queues_SOURCES} ${TEST_SOURCES})
endif()
