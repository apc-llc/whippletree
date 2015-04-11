cmake_minimum_required(VERSION 2.8)

set(SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

SET(GENERAL
	${SOURCE_DIR}/timing.h
	${SOURCE_DIR}/techniqueInterface.h
	${SOURCE_DIR}/queueInterface.cuh
	${SOURCE_DIR}/procedureInterface.cuh
	${SOURCE_DIR}/procinfoTemplate.cuh
	${SOURCE_DIR}/random.cuh
)
SET(TECHNIQUE_SOURCES
	${SOURCE_DIR}/techniqueMegakernel.cuh
	${SOURCE_DIR}/techniqueMegakernel.cu
	${SOURCE_DIR}/techniqueDynamicParallelism.cuh
	${SOURCE_DIR}/techniqueKernels.cuh
	${SOURCE_DIR}/techniqueKernels.cu
	${SOURCE_DIR}/delay.cuh
	${SOURCE_DIR}/delay.cu
)
SET(QUEUE_SOURCES
	${SOURCE_DIR}/segmentedStorage.cuh
	${SOURCE_DIR}/segmentedStorage.cu
	${SOURCE_DIR}/queueHelpers.cuh
	${SOURCE_DIR}/queueDistLocks.cuh
	${SOURCE_DIR}/queueExternalFetch.cuh
	${SOURCE_DIR}/queueCollector.cuh
	${SOURCE_DIR}/queueShared.cuh  

	${SOURCE_DIR}/queuingMultiPhase.cuh
	${SOURCE_DIR}/queuingPerProc.cuh
)
SET(UTILS_SOURCES
	${SOURCE_DIR}/tools/bitonicSort.cuh
	${SOURCE_DIR}/tools/cl_memory.h
	${SOURCE_DIR}/tools/common.cuh
	${SOURCE_DIR}/tools/cuda_memory.h
	${SOURCE_DIR}/tools/types.h
	${SOURCE_DIR}/tools/utils.h
)
SET(OCL_SOURCES
	${SOURCE_DIR}/OCL/CLBuffer.cpp
	${SOURCE_DIR}/OCL/CLBuffer.h
	${SOURCE_DIR}/OCL/CLBufferShared.cpp
	${SOURCE_DIR}/OCL/CLBufferShared.h
	${SOURCE_DIR}/OCL/CLCommandQueue.cpp
	${SOURCE_DIR}/OCL/CLCommandQueue.h
	${SOURCE_DIR}/OCL/CLContext.cpp
	${SOURCE_DIR}/OCL/CLContext.h
	${SOURCE_DIR}/OCL/CLDevice.cpp
	${SOURCE_DIR}/OCL/CLDevice.h
	${SOURCE_DIR}/OCL/CLKernel.cpp
	${SOURCE_DIR}/OCL/CLKernel.h
	${SOURCE_DIR}/OCL/CLMem.cpp
	${SOURCE_DIR}/OCL/CLMem.h
	${SOURCE_DIR}/OCL/CLPlatform.cpp
	${SOURCE_DIR}/OCL/CLPlatform.h
	${SOURCE_DIR}/OCL/CLProgram.cpp
	${SOURCE_DIR}/OCL/CLProgram.h
)

SOURCE_GROUP("General" FILES
	${GENERAL}
)
SOURCE_GROUP("Queues" FILES
	${QUEUE_SOURCES}
)
SOURCE_GROUP("Techniques" FILES
	${TECHNIQUE_SOURCES}
)
SOURCE_GROUP("Tools" FILES
	${TOOLS_SOURCES}
)
SOURCE_GROUP("OpenCL" FILES
	${OCL_SOURCES}
)

set(queues_SOURCES ${GENERAL} ${TECHNIQUE_SOURCES} ${QUEUE_SOURCES} ${TOOLS_SOURCES} ${OCL_SOURCES})

include_directories(${SOURCE_DIR})

