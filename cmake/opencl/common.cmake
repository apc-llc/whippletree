set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR})

find_package(OpenCL REQUIRED)

# On Linux the code requires C++ compiler with C++11 support.
if (NOT WIN32)
	set(CMAKE_CXX_FLAGS "-std=c++11")
endif ()

include_directories(${OPENCL_INCLUDE_DIRS})

add_definitions(-D_OPENCL)
add_definitions(-D'__device__=')
add_definitions(-D'__host__=')
add_definitions(-D'__global__=__kernel')
add_definitions(-D'__shared__=__local')
add_definitions(-D'threadIdx_x=get_local_id(0)')
add_definitions(-D'blockDim_x=get_local_size(0)')
add_definitions(-D'blockIdx_x=(get_global_id(0) / get_local_size(0))')
add_definitions(-D'gridDim_x=(get_global_size(0) / get_local_size(0))')
