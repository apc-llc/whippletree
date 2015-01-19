#include <cstdio>

#ifndef CUDA_CHECKED_CALL
#define CUDA_CHECKED_CALL(x) do { cudaError_t err = x; if (( err ) != cudaSuccess ) { \
        printf ("Error \"%s\" at %s :%d \n" , cudaGetErrorString(err), \
                        __FILE__ , __LINE__ ) ; exit(-1);\
}} while (0)
#endif

namespace tasman
{
	typedef void (*DynamicTaskFunction)(int threadId, int numThreads, void* data, volatile uint* shared);

	struct DynamicTaskInfo;
}

namespace
{
	using namespace tasman;

	template<DynamicTaskFunction Func>	
	__global__ void getfuncaddress(DynamicTaskInfo* info);
}

namespace tasman
{
	extern __device__ DynamicTaskInfo* submission;
	extern __device__ int finish;

	struct DynamicTaskInfo
	{
		DynamicTaskFunction func;
		void* data;
	};

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

	template <class T>
	class NonCopyable
	{
	protected:
		NonCopyable() { }
		~NonCopyable() { } // Protected non-virtual destructor

	private: 
		NonCopyable(const NonCopyable &);
		NonCopyable& operator=(const NonCopyable &);
	};

	class DynamicTaskManager : private NonCopyable<DynamicTaskManager>
	{
		cudaStream_t stream1, stream2;
		int* address;
		
		bool started;
	
		DynamicTaskManager();

		~DynamicTaskManager();

	public :

		static DynamicTaskManager& get();

		void start();
	
		void stop();

		void enqueue(const DynamicTask* task, void* data) const;
	};
}

namespace
{
	using namespace tasman;

	template<DynamicTaskFunction Func>	
	__global__ void getfuncaddress(DynamicTaskInfo* info)
	{
		info->func = Func;
	}
}

