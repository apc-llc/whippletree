#include "techniqueKernels.cuh"

namespace KernelLaunches
{
#if defined(_CUDA)
	__device__ int queueCountsVar[MaxProcs];
#endif
	__device__ int* queueCounts()
#if defined(_CUDA)
	{ return queueCountsVar; }
#elif defined(_OPENCL)
#error "Implement in OpenCL"
	{ /* TODO */ return NULL; }
#endif
}

