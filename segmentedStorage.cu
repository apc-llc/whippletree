#include "queueInterface.cuh"
#include "queueHelpers.cuh"
#include "segmentedStorage.cuh"

#if defined(_CUDA)
namespace
{
	__device__ void* storage_ = NULL;
}
#endif

__device__ void** storage()
#if defined(_CUDA)
{ return &::storage_; }
#elif defined(_OPENCL)
#error "Implement in OpenCL"
{ /* TODO */ return NULL; }
#endif

void* SegmentedStorage::StoragePointer = 0;

