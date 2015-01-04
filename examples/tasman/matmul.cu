#include "DynamicTaskManager.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <unistd.h>

using namespace std;

void __device__ func1(int threadId, int numThreads, void* data, volatile uint* shared)
{
	if (threadId == 0)
		printf("Task #1 instance #%d processed!\n", *(int*)data);
}

void __device__ func2(int threadId, int numThreads, void* data, volatile uint* shared)
{
	if (threadId == 0)
		printf("Task #2 instance #%d processed!\n", *(int*)data);
}

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << "Usage: " << argv[0] << " <ntasks>" << endl;
		return 1;
	}
	
	int ntasks = atoi(argv[1]);

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

	// Register dynamic tasks.
	// XXX Note: all dynamic tasks must be registered BEFORE
	// starting dynamic task manager.
	unique_ptr<DynamicTask> task1(DynamicTask::Create<func1>());
	unique_ptr<DynamicTask> task2(DynamicTask::Create<func2>());

	// Get dynamic task manager instance (unique singleton atm).
	DynamicTaskManager& dtm = DynamicTaskManager::get();
	
	// Create sample data for the given number of tasks.
	// XXX Note: all device memory allocations must happen BEFORE
	// starting dynamic task manager.
	int* hindexes = new int[ntasks];
	for (int i = 0; i < ntasks; i++)
		hindexes[i] = i;
	int *dindexes = NULL;
	CUDA_CHECKED_CALL(cudaMalloc(&dindexes, sizeof(int) * ntasks));
	CUDA_CHECKED_CALL(cudaMemcpy(dindexes, hindexes, sizeof(int) * ntasks, cudaMemcpyHostToDevice));

	// Launch dynamic task manager (that is, it will be resident in
	// GPU until stopped).
	dtm.start();

	// Dynamically add tasks into task manager.
	for (int i = 0; i < ntasks; i++)
	{
		if (i % 2)
			dtm.enqueue(task2.get(), &dindexes[i]);
		else
			dtm.enqueue(task1.get(), &dindexes[i]);
	}

	// Make uberkernel to work for a while more - to test how it
	// stays resident even when out of tasks.
	uint timeout = 10;
	usleep(1000000 * timeout);

	// Signal dynamic task manager to shutdown (after all tasks
	// are done).
	dtm.stop();

	// Free sample data arrays.	
	CUDA_CHECKED_CALL(cudaFree(dindexes));
	delete[] hindexes;

	return 0;
}

