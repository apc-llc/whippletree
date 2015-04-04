namespace
{
	__device__ volatile float BigData_[1024 * 1024];
}

__device__ volatile float* BigData()
#if defined(_CUDA)
{ return ::BigData_; }
#elif defined(_OPENCL)
{ /* TODO */ return NULL; }
#endif
