#include <iostream>
#include <tools/utils.h>

#include "queueDistLocks.cuh"
#include "queueShared.cuh"
#include "queuingPerProc.cuh"
#include "techniqueMegakernel.cuh"
#include "techniqueKernels.cuh"
#include "techniqueDynamicParallelism.cuh"
#include "segmentedStorage.cuh"

#include "procedureInterface.cuh"
#include "procinfoTemplate.cuh"
#include "random.cuh"

class Proc2 : public ::Procedure
{
public:
  typedef int4 ExpectedData;
  static const int NumThreads = 64;
  static const bool ItemInput = false; // false results in a lvl 1  task
  static const int sharedMemory = 2*sizeof(int)*NumThreads;  // shared memory requirements 

  template<class Q, class Context>
  static __device__ __inline__ void execute(int threadId, int numThreads, Q* queue,  ExpectedData* data, volatile uint* shared) 
  { 
    //now we have got 64 threads working here... we might communicate via shared memory
    //or run synchronizations..

    // store something in shared memory
    shared[threadId] = data->x + threadId;
    shared[NumThreads + threadId] = data->x - threadId;

    // run a prefix sum
    int n = 2*NumThreads;
    int offset = 1;  
    for (int d = n/2; d > 0; d/=2)  
    {   
      //use the special sync, as the cuda blocksize might not match the megakernel blocksize
      Context::sync();
      if (threadId < d)  
      {  
        int ai = offset*(2*threadId+1)-1;
        int bi = offset*(2*threadId+2)-1;
        shared[bi] += shared[ai];  
      }  
      offset *= 2;  
    }

    Context::sync();
    if (threadId == 0) 
        shared[n - 1] = 0;  

    for (int d = 1; d < n; d *= 2) 
    {  
      offset /= 2;  
      Context::sync(); 
      if (threadId < d)                       
      { 
        int ai = offset*(2*threadId+1)-1;  
        int bi = offset*(2*threadId+2)-1;  
        float t = shared[ai];  
        shared[ai] = shared[bi];  
        shared[bi] += t;   
      }  
    }  
    Context::sync();

	 if(threadId == numThreads-1)
		printf("thread %d of %d excutes Proc2 for data %d (CUDA thread %d %d) and computed prefix sum: %d\n", threadId, numThreads, data->x, threadIdx.x, blockIdx.x, shared[threadId]);
  }
};

//somehow we need to get something into the queue
//the init proc does that for us
class InitProc
{
public:
  template<class Q>
  __device__ __inline__
  static void init(Q* q, int id)
  {
    //so lets put something into the queues
    int4 d = make_int4(id+1, 0, 1, 2);
    q-> template enqueueInitial<Proc2>(d);
  }
};


typedef ProcInfo<Proc2>TestProcInfo;


//lets use a dist locks queue for each procedure, which can hold 12k elements
template<class ProcInfo>
class MyQueue : public PerProcedureQueueTyping<QueueDistLocksOpt_t, 12*1024, false>::Type<ProcInfo> {};


//and lets use a Megakernel which can execute multiple workpackages concurrently (dynamic)
//and offers a maximum of 16k shared memory

typedef Megakernel::DynamicPointed16336<MyQueue, TestProcInfo> MyTechnique;

void runTest(int cuda_device)
{
  cudaSetDevice(cuda_device);

  //create everything
  MyTechnique technique;
  technique.init();
  
  technique.insertIntoQueue<InitProc>(10);
  float t = technique.execute(0);
  printf("run completed in %fs\n", t);
}

void runTest(int device);
int main(int argc, char** argv)
{
  try
  {
    int cuda_device = argc > 1 ? atoi(argv[1]) : 0;

    int count;
    CUDA_CHECKED_CALL(cudaGetDeviceCount(&count));
    if (!count)
    {
       std::cout << "No CUDA devices available" << std::endl;
       return -1;
    }
    cudaDeviceProp deviceProp;
    CUDA_CHECKED_CALL(cudaGetDeviceProperties(&deviceProp, cuda_device));
    std::cout << "Using device: " << deviceProp.name << std::endl;

	runTest(cuda_device);
#ifdef WIN32
  if(argc < 3)
    getchar();
#endif
	return 0;
	}
  catch (const Tools::CudaError& e)
  {
    std::cout << "CUDA error: " << e.what() << std::endl;
#ifdef WIN32
    getchar();
#endif
    return -1;
  }
  catch (const std::exception& e)
  {
    std::cout << "error: " << e.what() << std::endl;
#ifdef WIN32
    getchar();
#endif
    return -2;
  }
  catch (...)
  {
    std::cout << "unknown exception!" << std::endl;
#ifdef WIN32
    getchar();
#endif
    return -3;
  }
}
