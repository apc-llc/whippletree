cmake_minimum_required(VERSION 2.8)

project(queuing)

include(../../../cmake/opencl/common.cmake)
include(../../../cmake/opencl/queues.cmake)

set(SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/..")

SET(TEST_SOURCES
  ${SOURCE_DIR}/proc0.cuh
  ${SOURCE_DIR}/proc1.cuh
  ${SOURCE_DIR}/proc2.cuh
  ${SOURCE_DIR}/main.cpp
)

add_executable(queuing ${GENERAL} ${queues_SOURCES} ${TEST_SOURCES})
target_link_libraries(queuing ${OPENCL_LIBRARIES}) 
