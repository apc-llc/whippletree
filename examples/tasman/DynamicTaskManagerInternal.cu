#include <tools/utils.h>

#include "queueDistLocks.cuh"
#include "queueShared.cuh"
#include "queuingPerProc.cuh"
#include "techniqueMegakernel.cuh"
#include "techniqueKernels.cuh"

#include "procedureInterface.cuh"
#include "procinfoTemplate.cuh"

#include "DynamicTaskManager.h"

namespace tasman
{
	__device__ DynamicTaskInfo* submission;
	__device__ int finish;
}

namespace
{
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
			if (threadIdx_x == 0)
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

	class DynamicTaskManagerInternal : private NonCopyable<DynamicTaskManagerInternal>
	{
		cudaStream_t stream1, stream2;
		int* address;
	
		MyTechnique technique;

		DynamicTaskManagerInternal();

	public :

		static DynamicTaskManagerInternal& get();

		void start(cudaStream_t stream);
	};

	DynamicTaskManagerInternal::DynamicTaskManagerInternal()
	{
		// Determine address of finishing marker to supply it into
		// the technique.
		CHECKED_CALL(cudaGetSymbolAddress((void**)&address, finish));
	}

	DynamicTaskManagerInternal& DynamicTaskManagerInternal::get()
	{
		static DynamicTaskManagerInternal dtmi;
		return dtmi;
	}

	void DynamicTaskManagerInternal::start(cudaStream_t stream)
	{
		// Start megakernel in a dedicated stream.
		technique.init();
		technique.execute(0, stream, address);
	}
}

namespace tasman
{
	extern "C" void dynamicTaskManagerStart(cudaStream_t stream)
	{
		DynamicTaskManagerInternal::get().start(stream);
	}
}

