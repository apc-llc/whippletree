#include <tools/utils.h>

#include "queueDistLocks.cuh"
#include "queueShared.cuh"
#include "queuingPerProc.cuh"
#include "techniqueMegakernel.cuh"
#include "techniqueKernels.cuh"

#include "procedureInterface.cuh"
#include "procinfoTemplate.cuh"

typedef void (*DynamicTaskFunction)(int threadId, int numThreads, void* data, volatile uint* shared);

namespace
{
	struct DynamicTaskInfo;
}

extern __device__ DynamicTaskInfo* submission;

namespace
{
	struct DynamicTaskInfo
	{
		DynamicTaskFunction func;
		void* data;
	};

	template<DynamicTaskFunction Func>	
	__global__ void getfuncaddress(DynamicTaskInfo* info)
	{
		info->func = Func;
	}

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
			DynamicTaskInfo* info = (DynamicTaskInfo*)data;
			info->func(threadId, numThreads, info->data, shared);
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
				if (submission)
				{
					TQueue::Type<ProcInfo>::template enqueue<Task>(*submission);
					submission = NULL;
				}
			}			 
		}
	};

	typedef Megakernel::SimplePointed16336<MyQueue, ProcInfo<Task>, void, Megakernel::ShutdownIndicator> MyTechnique;
}

class DynamicTaskManager;

class DynamicTask
{
	DynamicTaskInfo* info;

	friend class DynamicTaskManager;

public :

	template<DynamicTaskFunction Func>
	static DynamicTask* Create()
	{
		DynamicTask* task = new DynamicTask();

		CUDA_CHECKED_CALL(cudaMalloc(&task->info, sizeof(DynamicTaskInfo)));

		// Determine the given task function address on device.
		getfuncaddress<Func><<<1, 1>>>(task->info);
		CUDA_CHECKED_CALL(cudaDeviceSynchronize());
		
		return task;
	}

	~DynamicTask();
};

class DynamicTaskManager
{
	cudaStream_t stream1, stream2;
	int* address;
	
	MyTechnique technique;

	DynamicTaskManager();

	DynamicTaskManager(DynamicTaskManager const&);
	void operator=(DynamicTaskManager const&);

	~DynamicTaskManager();

public :

	static DynamicTaskManager& get();

	void start();
	
	void stop();

	void enqueue(const DynamicTask* task, void* data) const;
};

