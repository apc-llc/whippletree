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

#pragma once
#include "procedureInterface.cuh"
#include "procinfoTemplate.cuh"
#include "random.cuh"
#include <tools/utils.h>


class Proc2 : public ::Procedure
{
public:
  typedef int4 ExpectedData;
  static const int NumThreads = 64;
  static const bool ItemInput = false; // false results in a lvl 1  task
  static const int sharedMemory = 2*sizeof(int)*NumThreads;  // shared memory requirements 

  template<class Q, class Context>
  static __device__ __inline__ void execute(int threadId, int numThreads, Q* queue,  ExpectedData* data, volatile uint* shared) 
  { 
    //now we have got 64 threads working here... we might communicate via shared memory
    //or run synchronizations..

    // store something in shared memory
    shared[threadId] = data->x + threadId;
    shared[NumThreads + threadId] = data->x - threadId;

    // run a prefix sum
    int n = 2*NumThreads;
    int offset = 1;  
    for (int d = n/2; d > 0; d/=2)  
    {   
      //use the special sync, as the cuda blocksize might not match the megakernel blocksize
      Context::sync();
      if (threadId < d)  
      {  
        int ai = offset*(2*threadId+1)-1;
        int bi = offset*(2*threadId+2)-1;
        shared[bi] += shared[ai];  
      }  
      offset *= 2;  
    }

    Context::sync();
    if (threadId == 0) 
        shared[n - 1] = 0;  

    for (int d = 1; d < n; d *= 2) 
    {  
      offset /= 2;  
      Context::sync(); 
      if (threadId < d)                       
      { 
        int ai = offset*(2*threadId+1)-1;  
        int bi = offset*(2*threadId+2)-1;  
        float t = shared[ai];  
        shared[ai] = shared[bi];  
        shared[bi] += t;   
      }  
    }  
    Context::sync();

	 if(threadId == numThreads-1)
		printf("thread %d of %d excutes Proc2 for data %d (CUDA thread %d %d) and computed prefix sum: %d\n", threadId, numThreads, data->x, threadIdx_x, blockIdx_x, shared[threadId]);
  }
};
