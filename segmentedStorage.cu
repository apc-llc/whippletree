#include "queueInterface.cuh"
#include "queueHelpers.cuh"
#include "segmentedStorage.cuh"

namespace
{
	__device__ void* storage_ = NULL;
}

__device__ void** storage()
#if defined(_CUDA)
{ return &::storage_; }
#elif defined(_OPENCL)
{ /* TODO */ return NULL; }
#endif

void* SegmentedStorage::StoragePointer = 0;

