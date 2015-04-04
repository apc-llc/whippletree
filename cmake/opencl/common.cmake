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
