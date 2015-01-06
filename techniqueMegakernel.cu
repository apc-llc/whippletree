namespace Megakernel
{
  __device__ volatile int doneCounter = 0;
  __device__ volatile int endCounter = 0;

  __device__ int maxConcurrentBlocks = 0;
  __device__ volatile int maxConcurrentBlockEvalDone = 0;
}

