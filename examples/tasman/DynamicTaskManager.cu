#include "DynamicTaskManager.h"

extern "C" void dynamicTaskManagerStart(cudaStream_t stream);

namespace tasman
{
	DynamicTask::~DynamicTask()
	{
		CHECKED_CALL(cudaFree(info));
	}

	DynamicTaskManager::DynamicTaskManager() : started(false)
	{
		// Create two streams: one for megakernel, and another one -
		// for finish indicator.
		CHECKED_CALL(cudaStreamCreate(&stream1));
		CHECKED_CALL(cudaStreamCreate(&stream2));
	
		// Determine address of finishing marker to supply it into
		// the technique.
		CHECKED_CALL(cudaGetSymbolAddress((void**)&address, finish));
	}

	DynamicTaskManager::~DynamicTaskManager()
	{
		// Destroy streams.
		CHECKED_CALL(cudaStreamDestroy(stream1));
		CHECKED_CALL(cudaStreamDestroy(stream2));
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
		CHECKED_CALL(cudaMemcpyToSymbolAsync(finish, &value, sizeof(int), 0, cudaMemcpyHostToDevice, stream2));
		CHECKED_CALL(cudaStreamSynchronize(stream2));
	
		dynamicTaskManagerStart(stream1);
	}

	void DynamicTaskManager::stop()
	{
		// Wait until queue gets empty.
		while (true)
		{
			DynamicTaskInfo* busy = NULL;
			CHECKED_CALL(cudaMemcpyFromSymbolAsync(&busy, submission, sizeof(DynamicTaskInfo*), 0, cudaMemcpyDeviceToHost, stream2));
			CHECKED_CALL(cudaStreamSynchronize(stream2));
			if (!busy) break;
		}

		// Signal shut down to uberkernel.
		int value = 1;
		CHECKED_CALL(cudaMemcpyToSymbolAsync(finish, &value, sizeof(int), 0, cudaMemcpyHostToDevice, stream2));
		CHECKED_CALL(cudaStreamSynchronize(stream2));
	
		// Wait for uberkernel to finish.
		CHECKED_CALL(cudaStreamSynchronize(stream1));
	
		started = false;
	}

	void DynamicTaskManager::enqueue(const DynamicTask* task, void* data) const
	{	
		// Wait until queue gets empty.
		while (true)
		{
			DynamicTaskInfo* busy = NULL;
			CHECKED_CALL(cudaMemcpyFromSymbolAsync(&busy, submission, sizeof(DynamicTaskInfo*), 0, cudaMemcpyDeviceToHost, stream2));
			CHECKED_CALL(cudaStreamSynchronize(stream2));
			if (!busy) break;
		}

		// Copy data to device memory.
		CHECKED_CALL(cudaMemcpyAsync(&task->info->data, &data, sizeof(void*), cudaMemcpyHostToDevice, stream2));

		// Submit task into queue.
		CHECKED_CALL(cudaMemcpyToSymbolAsync(submission, &task->info, sizeof(DynamicTaskInfo*), 0, cudaMemcpyHostToDevice, stream2));
		CHECKED_CALL(cudaStreamSynchronize(stream2));
	}
}

