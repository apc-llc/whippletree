// TODO:
// 1) we can do a single persistence atomic counter
// 2) status array should be just char-byte
// 3) make each warp to wait for timeout and then - exit. Stop counting, when at least one warp has already exited

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

// TODO: specify regcount

namespace PersistenceTest
{
	static const uint timeout = 2;
	__device__ uint dCount;
	static uint* hFinish;
	__constant__ uint dFinish;

	class Task : public ::Procedure
	{
	public:
		static const int NumThreads = 32;
		static const bool ItemInput = false; // false results in a lvl 1 task
		static const int sharedMemory = 0; // shared memory requirements 

		typedef void* ExpectedData;
	
		template<class Q, class Context>
		static __device__ __inline__ void execute(int threadId, int numThreads, Q* queue, ExpectedData* data, volatile uint* shared)
		{
			// This kernel counts the number of warps that can be launched
			// by device as a single wavefront.
	
			// Count only while not finished
			if ((threadId == 0) && !dFinish)
				atomicAdd(&dCount, 1);

			// Wait till finish signal from host.	
			while (atomicCAS(&dFinish, 1, 1) != 1) { clock64(); }
		}

		template<class Q>
		__device__ __inline__ static void init(Q* q, int id)
		{
			q->template enqueueInitial<PersistenceTest::Task>(NULL);
		}
	};

	// Lets use a dist locks queue for each procedure, which can hold 96k elements
	template<class ProcInfo>
	class MyQueue : public PerProcedureQueueTyping<QueueDistLocksOpt_t, 96 * 1024, false>::Type<ProcInfo>
	{
	};

	typedef Megakernel::SimplePointed16336<MyQueue, ProcInfo<PersistenceTest::Task> > MyTechnique;

	// Find the maximum number of persistent tasks the device can carry.	
	static uint run()
	{
		cudaDeviceProp deviceProp;
		CUDA_CHECKED_CALL(cudaGetDeviceProperties(&deviceProp, 0));

		uint hCount = 0;
		uint zero = 0;
		cudaStream_t stream1, stream2;
		CUDA_CHECKED_CALL(cudaStreamCreate(&stream1));
		CUDA_CHECKED_CALL(cudaStreamCreate(&stream2));
		for (int ntasks = 1, ntasks_max = deviceProp.maxGridSize[0]; ntasks < ntasks_max; ntasks *= 2)
		{
			// Reset counter.
			CUDA_CHECKED_CALL(cudaMemcpyToSymbol(dCount, &zero, sizeof(uint)));
			
			// Reset finish marker.
			CUDA_CHECKED_CALL(cudaMemcpyToSymbol(dFinish, &zero, sizeof(uint)));

			// Launch uberkernel.
			MyTechnique technique;
			technique.init();
			technique.insertIntoQueue<PersistenceTest::Task>(ntasks);

			technique.execute(0, stream1);
		
			// Make uberkernel to work for a while.
			usleep(1000000 * timeout);

			// Signal shut down to uberkernel.
			uint one = 1;
			/*CUDA_CHECKED_CALL(cudaMemcpyAsync(hFinish, &one, sizeof(uint), cudaMemcpyHostToDevice, stream2));
			CUDA_CHECKED_CALL(cudaStreamSynchronize(stream2));*/
			CUDA_CHECKED_CALL(cudaMemcpyToSymbolAsync(dFinish, &one, sizeof(uint), 0, cudaMemcpyHostToDevice, stream2));
			CUDA_CHECKED_CALL(cudaStreamSynchronize(stream2));
		
			CUDA_CHECKED_CALL(cudaStreamSynchronize(stream1));
		
			uint hCountCurrent;
			CUDA_CHECKED_CALL(cudaMemcpyFromSymbol(&hCountCurrent, dCount, sizeof(uint)));
			printf("# persistent warps: %u\n", hCountCurrent);
			if (hCountCurrent <= hCount)
				break;
			hCount = hCountCurrent;
		}
		CUDA_CHECKED_CALL(cudaFreeHost(hFinish));
		printf("# max persistent warps: %u\n", hCount);
		CUDA_CHECKED_CALL(cudaStreamDestroy(stream1));
		CUDA_CHECKED_CALL(cudaStreamDestroy(stream2));
		
		return hCount;
	}
}

struct TaskInfo
{
	uint id;
	uint* ready;
};

static uint* status;

class Task : public ::Procedure
{
public:
	static const int NumThreads = 32;
	static const bool ItemInput = false; // false results in a lvl 1 task
	static const int sharedMemory = 0; // shared memory requirements 
	
	typedef TaskInfo ExpectedData;

	template<class Q, class Context>
	static __device__ __inline__ void execute(int threadId, int numThreads, Q* queue, ExpectedData* ptask, volatile uint* shared)
	{
		while (!atomicCAS(ptask->ready, 1, 0)) { clock64(); }
		
		if (threadId == 0)
			printf("Executing task %04u\n", ptask->id);
	}

	template<class Q>
	__device__ __inline__ static void init(Q* q, int id)
	{
		TaskInfo task;
		task.id = id;
		task.ready = &status[id];
		q->template enqueueInitial<Task>(task);
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

	// Find the maximum number of persistent tasks the device can carry.	
	uint count = PersistenceTest::run();

#if 0
	//
	// 2) Launch the maximum number of persistent tasks the device can carry.
	//
	{
		// Create array of task readiness locks. These will be atomically
		// CAS-ed by host and by whippletree.
		uint* hstatus = NULL;
		CUDA_CHECKED_CALL(cudaHostAlloc(&hstatus, sizeof(uint) * ntasks, cudaHostAllocMapped));
		CUDA_CHECKED_CALL(cudaMemset(hstatus, 0, sizeof(uint) * ntasks));
		CUDA_CHECKED_CALL(cudaMemcpyToSymbol(status, &hstatus, sizeof(uint*)));

		MyTechnique technique;
		technique.init();
		technique.insertIntoQueue<Task>(ntasks);

		cudaStream_t stream;
		CUDA_CHECKED_CALL(cudaStreamCreate(&stream));
		cudaEvent_t a, b;
		CUDA_CHECKED_CALL(cudaEventCreate(&a));
		CUDA_CHECKED_CALL(cudaEventCreate(&b));
		CUDA_CHECKED_CALL(cudaEventRecord(a, stream));

		technique.execute(0, stream);

		CUDA_CHECKED_CALL(cudaEventRecord(b, stream));
		CUDA_CHECKED_CALL(cudaEventSynchronize(b));
		float time;
		CUDA_CHECKED_CALL(cudaEventElapsedTime(&time, a, b));
		time /= 1000.0;
		CUDA_CHECKED_CALL(cudaEventDestroy(a));
		CUDA_CHECKED_CALL(cudaEventDestroy(b));
		CUDA_CHECKED_CALL(cudaStreamDestroy(stream));

		CUDA_CHECKED_CALL(cudaFreeHost(hstatus));
	}
#endif

	return 0;
}

