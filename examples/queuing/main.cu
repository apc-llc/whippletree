//  Project Whippletree
//  http://www.icg.tugraz.at/project/parallel
//
//  Copyright (C) 2014 Institute for Computer Graphics and Vision,
//                     Graz University of Technology
//
//  Author(s):  Markus Steinberger - steinberger ( at ) icg.tugraz.at
//              Michael Kenzel - kenzel ( at ) icg.tugraz.at
//              Pedro Boechat - boechat ( at ) icg.tugraz.at
//              Bernhard Kerbl - kerbl ( at ) icg.tugraz.at
//              Mark Dokter - dokter ( at ) icg.tugraz.at
//              Dieter Schmalstieg - schmalstieg ( at ) icg.tugraz.at
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE.
//

#include "queueDistLocks.cuh"
#include "queueShared.cuh"
#include "queuingPerProc.cuh"
#include "techniqueMegakernel.cuh"
#include "techniqueKernels.cuh"
#include "techniqueDynamicParallelism.cuh"
#include "segmentedStorage.cuh"
#if defined(_OPENCL)
#include "clcode.h"
#endif

#include "proc0.cuh"
#include "proc1.cuh"
#include "proc2.cuh"

#if defined(_OPENCL)
BOLT_FUNCTOR(InitProc,
#endif
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
		q-> template enqueueInitial<Proc0>(d);
	}
};
#if defined(_OPENCL)
);
#endif

typedef ProcInfo<Proc0,N<Proc1,N<Proc2> > >TestProcInfo;

//lets use a dist locks queue for each procedure, which can hold 12k elements
template<class ProcInfo>
class MyQueue : public PerProcedureQueueTyping<QueueDistLocksOpt_t, 12*1024, false>::Type<ProcInfo> {};

//and lets use a Megakernel which can execute multiple workpackages concurrently (dynamic)
//and offers a maximum of 16k shared memory

typedef Megakernel::DynamicPointed16336<MyQueue, TestProcInfo> MyTechnique;

//typedef KernelLaunches::TechniqueMultiple<MyQueue, TestProcInfo> MyTechnique;

//typedef DynamicParallelism::TechniqueQueuedNoCopy<MyQueue, InitProc, TestProcInfo> MyTechnique;

#include <iostream>
#include <tools/utils.h>

int main(int argc, char* argv[])
{
	int count;
	CUDA_CHECKED_CALL(cudaGetDeviceCount(&count));
	if (!count)
	{
		std::cerr << "No CUDA devices available" << std::endl;
#ifdef WIN32
		getchar();
#endif
		return -1;
	}
	cudaDeviceProp deviceProp;
	CUDA_CHECKED_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	std::cout << "Using device: " << deviceProp.name << std::endl;

	//create everything
	MyTechnique technique;
	technique.init();

	float time;

	technique.insertIntoQueue<InitProc>(10);

#if defined(_CUDA)
	cudaStream_t stream;
	CUDA_CHECKED_CALL(cudaStreamCreate(&stream));
	cudaEvent_t a, b;
	CUDA_CHECKED_CALL(cudaEventCreate(&a));
	CUDA_CHECKED_CALL(cudaEventCreate(&b));
	CUDA_CHECKED_CALL(cudaEventRecord(a, stream));
#elif defined(_OPENCL)
	// TODO Measure time
#endif

	technique.execute(0, stream);

#if defined(_CUDA)
	CUDA_CHECKED_CALL(cudaEventRecord(b, stream));
	CUDA_CHECKED_CALL(cudaEventSynchronize(b));
	CUDA_CHECKED_CALL(cudaEventElapsedTime(&time, a, b));
	time /= 1000.0;
	CUDA_CHECKED_CALL(cudaEventDestroy(a));
	CUDA_CHECKED_CALL(cudaEventDestroy(b));
	CUDA_CHECKED_CALL(cudaStreamDestroy(stream));
#elif defined(_OPENCL)
	// TODO Measure time
#endif

	std::cout << "run completed in " << time << " sec" << std::endl;

#ifdef WIN32
	getchar();
#endif

	return 0;
}

