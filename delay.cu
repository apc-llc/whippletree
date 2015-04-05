#if defined(_CUDA)
namespace
{
	__device__ volatile float BigData_[1024 * 1024];
}
#endif

__device__ volatile float* BigData()
#if defined(_CUDA)
{ return ::BigData_; }
#elif defined(_OPENCL)
#error "Implement in OpenCL"
{ /* TODO */ return NULL; }
#endif
