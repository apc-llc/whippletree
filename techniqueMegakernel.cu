namespace Megakernel
{
#if defined(_CUDA)
	__device__ volatile int doneCounterVar = 0;
	__device__ volatile int endCounterVar = 0;
#endif
	__device__ int* doneCounter()
#if defined(_CUDA)
	{
		return (int*)&doneCounterVar;
	}
#elif defined(_OPENCL)
#error "Implement in OpenCL"
	{ /* TODO */ return NULL; }
#endif

	__device__ int* endCounter()
#if defined(_CUDA)
	{
		return (int*)&endCounterVar;
	}
#elif defined(_OPENCL)
#error "Implement in OpenCL"
	{ /* TODO */ return NULL; }
#endif

#if defined(_CUDA)
	__device__ int maxConcurrentBlocksVar = 0;
	__device__ volatile int maxConcurrentBlockEvalDoneVar = 0;
#endif
	__device__ int* maxConcurrentBlocks()
#if defined(_CUDA)
	{
		return &maxConcurrentBlocksVar;
	}
#elif defined(_OPENCL)
#error "Implement in OpenCL"
	{ /* TODO */ return NULL; }
#endif

	__device__ int* maxConcurrentBlockEvalDone()
#if defined(_CUDA)
	{
		return (int*)&maxConcurrentBlockEvalDoneVar;
	}
#elif defined(_OPENCL)
#error "Implement in OpenCL"
	{ /* TODO */ return NULL; }
#endif
}

