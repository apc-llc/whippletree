#include "DynamicTaskManager.h"

extern "C" void dynamicTaskManagerStart(cudaStream_t stream);

namespace tasman
{
	DynamicTask::~DynamicTask()
	{
		CUDA_CHECKED_CALL(cudaFree(info));
	}

	DynamicTaskManager::DynamicTaskManager() : started(false)
	{
		// Create two streams: one for megakernel, and another one -
		// for finish indicator.
		CUDA_CHECKED_CALL(cudaStreamCreate(&stream1));
		CUDA_CHECKED_CALL(cudaStreamCreate(&stream2));
	
		// Determine address of finishing marker to supply it into
		// the technique.
		CUDA_CHECKED_CALL(cudaGetSymbolAddress((void**)&address, finish));
	}

	DynamicTaskManager::~DynamicTaskManager()
	{
		// Destroy streams.
		CUDA_CHECKED_CALL(cudaStreamDestroy(stream1));
		CUDA_CHECKED_CALL(cudaStreamDestroy(stream2));
	}

	DynamicTaskManager& DynamicTaskManager::get()
	{
		static DynamicTaskManager dtm;
		return dtm;
	}

	void DynamicTaskManager::start()
	{
		if (started) return;
		started = true;

		// Initialize finishing marker with "false" to make uberkernel
		// to run infinitely.
		int value = 0;
		CUDA_CHECKED_CALL(cudaMemcpyToSymbolAsync(finish, &value, sizeof(int), 0, cudaMemcpyHostToDevice, stream2));
		CUDA_CHECKED_CALL(cudaStreamSynchronize(stream2));
	
		dynamicTaskManagerStart(stream1);
	}

	void DynamicTaskManager::stop()
	{
		// Wait until queue gets empty.
		while (true)
		{
			DynamicTaskInfo* busy = NULL;
			CUDA_CHECKED_CALL(cudaMemcpyFromSymbolAsync(&busy, submission, sizeof(DynamicTaskInfo*), 0, cudaMemcpyDeviceToHost, stream2));
			CUDA_CHECKED_CALL(cudaStreamSynchronize(stream2));
			if (!busy) break;
		}

		// Signal shut down to uberkernel.
		int value = 1;
		CUDA_CHECKED_CALL(cudaMemcpyToSymbolAsync(finish, &value, sizeof(int), 0, cudaMemcpyHostToDevice, stream2));
		CUDA_CHECKED_CALL(cudaStreamSynchronize(stream2));
	
		// Wait for uberkernel to finish.
		CUDA_CHECKED_CALL(cudaStreamSynchronize(stream1));
	
		started = false;
	}

	void DynamicTaskManager::enqueue(const DynamicTask* task, void* data) const
	{	
		// Wait until queue gets empty.
		while (true)
		{
			DynamicTaskInfo* busy = NULL;
			CUDA_CHECKED_CALL(cudaMemcpyFromSymbolAsync(&busy, submission, sizeof(DynamicTaskInfo*), 0, cudaMemcpyDeviceToHost, stream2));
			CUDA_CHECKED_CALL(cudaStreamSynchronize(stream2));
			if (!busy) break;
		}

		// Copy data to device memory.
		CUDA_CHECKED_CALL(cudaMemcpyAsync(&task->info->data, &data, sizeof(void*), cudaMemcpyHostToDevice, stream2));

		// Submit task into queue.
		CUDA_CHECKED_CALL(cudaMemcpyToSymbolAsync(submission, &task->info, sizeof(DynamicTaskInfo*), 0, cudaMemcpyHostToDevice, stream2));
		CUDA_CHECKED_CALL(cudaStreamSynchronize(stream2));
	}
}

