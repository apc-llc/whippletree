#include <cstdio>

#ifndef CHECKED_CALL
#define CHECKED_CALL(x) do { cudaError_t err = x; if (( err ) != cudaSuccess ) { \
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
#ifdef __CUDACC__
	template<DynamicTaskFunction Func>	
	__global__ void getfuncaddress(DynamicTaskInfo* info);
#endif
}

namespace tasman
{
#ifdef __CUDACC__
	extern __device__ DynamicTaskInfo* submission;
	extern __device__ int finish;
#endif
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
#ifdef __CUDACC__
			CHECKED_CALL(cudaMalloc(&task->info, sizeof(DynamicTaskInfo)));

			// Determine the given task function address on device.
			getfuncaddress<Func><<<1, 1>>>(task->info);
			CHECKED_CALL(cudaDeviceSynchronize());
#endif		
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

#ifdef __CUDACC__
namespace
{
	using namespace tasman;

	template<DynamicTaskFunction Func>	
	__global__ void getfuncaddress(DynamicTaskInfo* info)
	{
		info->func = Func;
	}
}
#endif

