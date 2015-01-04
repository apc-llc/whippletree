// TODO:
// 1) Extend megakernel interface to run infinitely, waiting for tasks
// 2) Implement tasks queuing from host to device (requires 2-way dynamic data exchange?).
// 3) Develop system to determine pointers of task device functions

#include <cstdlib>
#include <iostream>
#include <time.h>
#include <tools/utils.h>
#include <unistd.h>

#include "queueDistLocks.cuh"
#include "queueShared.cuh"
#include "queuingPerProc.cuh"
#include "techniqueMegakernel.cuh"
#include "techniqueKernels.cuh"
#include "techniqueDynamicParallelism.cuh"
#include "segmentedStorage.cuh"

#include "procedureInterface.cuh"
#include "procinfoTemplate.cuh"

typedef void (*DynamicTaskFunction)(int threadId, int numThreads, void* data, volatile uint* shared);

struct DynamicTaskInfo
{
	DynamicTaskFunction func;
	void* data;
};

namespace
{
	__constant__ bool finish;
}

class DynamicTaskManager
{
	cudaStream_t stream1, stream2;
	bool* address;
	
public :

	class Task : public ::Procedure
	{
	public:
		static const int NumThreads = 32;
		static const bool ItemInput = false; // false results in a lvl 1 task
		static const int sharedMemory = 0; // shared memory requirements 

		typedef DynamicTaskInfo ExpectedData;
	
		template<class Q, class Context>
		static __device__ __inline__ void execute(int threadId, int numThreads, Q* queue, ExpectedData* data, volatile uint* shared)
		{
			// Execute given task with the given argument.
			DynamicTaskInfo* task = (DynamicTaskInfo*)data;
			task->func(threadId, numThreads, task->data, shared);
		}

		template<class Q>
		__device__ __inline__ static void init(Q* q, int id)
		{
			// Not supposed to have any initial queue.
			__trap();
		}
	};

	// Lets use a dist locks queue for each procedure, which can hold 96k elements
	typedef PerProcedureQueueTyping<QueueDistLocksOpt_t, 96 * 1024, false> TQueue;

	template<class ProcInfo>
	class MyQueue : public TQueue::Type<ProcInfo>
	{
	public :
		static const int globalMaintainMinThreads = 1;
		
		__inline__ __device__ void globalMaintain()
		{
			if (threadIdx.x == 0)
			{
				printf("Insert something into queue!\n");

				//TQueue::Type<ProcInfo>::template enqueue<Task>(NULL);
			}			 
		}
	};

private :

	typedef Megakernel::SimplePointed16336<MyQueue, ProcInfo<Task>, void, Megakernel::ShutdownIndicator> MyTechnique;

	MyTechnique technique;

public :

	void start()
	{
		// Initialize finishing marker with "false" to make uberkernel
		// to run infinitely.
		bool value = false;
		CUDA_CHECKED_CALL(cudaMemcpyToSymbolAsync(finish, &value, sizeof(bool), 0, cudaMemcpyHostToDevice, stream2));
		CUDA_CHECKED_CALL(cudaStreamSynchronize(stream2));

		// Start megakernel in a dedicated stream.
		technique.init();
		technique.execute(0, stream1, address);
	}
	
	void stop()
	{
		// Signal shut down to uberkernel.
		bool value = true;
		CUDA_CHECKED_CALL(cudaMemcpyToSymbolAsync(finish, &value, sizeof(bool), 0, cudaMemcpyHostToDevice, stream2));
		CUDA_CHECKED_CALL(cudaStreamSynchronize(stream2));
		
		// Wait for uberkernel to finish.
		CUDA_CHECKED_CALL(cudaStreamSynchronize(stream1));
	}

	void EnqueueTask()
	{
	}

	void EnqueueTaskAsync()
	{
	}

	DynamicTaskManager()
	{
		// Create two streams: one for megakernel, and another one -
		// for finish indicator.
		CUDA_CHECKED_CALL(cudaStreamCreate(&stream1));
		CUDA_CHECKED_CALL(cudaStreamCreate(&stream2));
		
		// Determine address of finishing marker to supply it into
		// the technique.
		CUDA_CHECKED_CALL(cudaGetSymbolAddress((void**)&address, finish));
	}

	~DynamicTaskManager()
	{
		// Destroy streams.
		CUDA_CHECKED_CALL(cudaStreamDestroy(stream1));
		CUDA_CHECKED_CALL(cudaStreamDestroy(stream2));
	}
};

int main(int argc, char** argv)
{
	using namespace std;

	{
		int count;
		CUDA_CHECKED_CALL(cudaGetDeviceCount(&count));
		if (!count)
		{
			cerr << "No CUDA devices available" << endl;
			return -1;
		}
		cudaDeviceProp deviceProp;
		CUDA_CHECKED_CALL(cudaGetDeviceProperties(&deviceProp, 0));
		cout << "Using device: " << deviceProp.name << endl;
	}

	DynamicTaskManager dtm;
	
	dtm.start();

	// Make uberkernel to work for a while.
	uint timeout = 10;
	usleep(1000000 * timeout);

	dtm.stop();

	return 0;
}

