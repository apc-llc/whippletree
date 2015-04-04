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

#include <memory>
#include <vector>
#include <tools/utils.h>
#include <tools/cuda_memory.h>
#include <iostream>
#include "timing.h"
#include "delay.cuh"

#include "techniqueInterface.h"

#include "procinfoTemplate.cuh"
#include "queuingMultiPhase.cuh"


namespace SegmentedStorage
{
  void checkReinitStorage();
}

namespace Megakernel
{
  enum MegakernelStopCriteria
  {
    // Stop megakernel, when the task queue is empty.
    EmptyQueue,

    // Stop megakernel, when the task queue is empty,
    // and "shutdown" indicator is filled with "true" value.
    ShutdownIndicator,
  };

  extern __device__ volatile int doneCounter;
  extern __device__ volatile int endCounter;

  template<class InitProc, class Q>
  __global__ void initData(Q* q, int num)
  {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    for( ; id < num; id += blockDim.x*gridDim.x)
    {
      InitProc::template init<Q>(q, id);
    }
  }

  template<class Q>
  __global__ void recordData(Q* q)
  {
    q->record();
  }
  template<class Q>
  __global__ void resetData(Q* q)
  {
    q->reset();
  }

  template<class Q, class ProcInfo, class PROC, class CUSTOM, bool Itemized,  bool MultiElement>
  class FuncCaller;


  template<class Q, class ProcInfo, class PROC, class CUSTOM>
  class FuncCaller<Q, ProcInfo, PROC, CUSTOM, false, false>
  {
  public:
    __device__ __inline__
    static void call(Q* queue, void* data, int hasData, uint* shared)
    {
      int nThreads;
      if(PROC::NumThreads != 0)
        nThreads = PROC::NumThreads;
      else
        nThreads = blockDim.x;
      if(PROC::NumThreads == 0 || threadIdx.x < nThreads)
        PROC :: template execute<Q, Context<PROC::NumThreads, false, CUSTOM> >(threadIdx.x, nThreads, queue, reinterpret_cast<typename PROC::ExpectedData*>(data), shared);
    }
  };

  template<class Q, class ProcInfo, class PROC, class CUSTOM>
  class FuncCaller<Q, ProcInfo, PROC, CUSTOM, false, true>
  {
  public:
    __device__ __inline__
    static void call(Q* queue, void* data, int hasData, uint* shared)
    {
      
      if(PROC::NumThreads != 0)
      {
        int nThreads;
        nThreads = PROC::NumThreads;
        int tid = threadIdx.x % PROC::NumThreads;
        int offset = threadIdx.x / PROC::NumThreads;
        if(threadIdx.x < hasData)
          PROC :: template execute<Q, Context<PROC::NumThreads, true, CUSTOM> >(tid, nThreads, queue, reinterpret_cast<typename PROC::ExpectedData*>(data), shared + offset*PROC::sharedMemory/sizeof(uint) );
      }
      else
      {
        PROC :: template execute<Q, Context<PROC::NumThreads, true, CUSTOM> >(threadIdx.x, blockDim.x, queue, reinterpret_cast<typename PROC::ExpectedData*>(data), shared);
      }
      
    }
  };

  template<class Q, class ProcInfo, class PROC, class CUSTOM, bool MultiElement>
  class FuncCaller<Q, ProcInfo, PROC, CUSTOM, true, MultiElement>
  {
  public:
    __device__ __inline__
    static void call(Q* queue, void* data, int numData, uint* shared)
    {
      if(threadIdx.x < numData)
        PROC :: template execute<Q, Context<PROC::NumThreads, MultiElement, CUSTOM> >(threadIdx.x, numData, queue, reinterpret_cast<typename PROC::ExpectedData*>(data), shared);
    }
  };

  
  ////////////////////////////////////////////////////////////////////////////////////////
  
  template<class Q, class ProcInfo, bool MultiElement>
  struct ProcCallCopyVisitor
  {
    int* execproc;
    const uint4 & sharedMem;
    Q* q;
    void* execData;
    uint* s_data;
	 int hasResult;
    __inline__ __device__ ProcCallCopyVisitor(Q* q, int *execproc, void * execData, uint* s_data, const uint4& sharedMem, int hasResult ) : execproc(execproc), sharedMem(sharedMem), q(q), execData(execData), s_data(s_data) { }
    template<class TProcedure, class CUSTOM>
    __device__ __inline__ bool visit()
    {
      if(*execproc == findProcId<ProcInfo, TProcedure>::value)
      {
          FuncCaller<Q, ProcInfo, TProcedure, CUSTOM, TProcedure :: ItemInput, MultiElement>::call(q, execData, hasResult, s_data + sharedMem.x + sharedMem.y + sharedMem.w );
          return true;
      }
      return false;
    }
  };

  template<class Q, class ProcInfo, bool MultiElement>
  struct ProcCallNoCopyVisitor
  {
    int* execproc;
    const uint4 & sharedMem;
    Q* q;
    void* execData;
    uint* s_data;
    int hasResult;
    __inline__ __device__ ProcCallNoCopyVisitor(Q* q, int *execproc, void * execData, uint* s_data, const uint4& sharedMem, int hasResult ) : execproc(execproc), sharedMem(sharedMem), q(q), execData(execData), s_data(s_data), hasResult(hasResult) { }
    template<class TProcedure, class CUSTOM>
    __device__ __inline__ bool visit()
    {
      if(*execproc == findProcId<ProcInfo, TProcedure>::value)
      {
          FuncCaller<Q, ProcInfo, TProcedure, CUSTOM, TProcedure :: ItemInput, MultiElement>::call(q, execData, hasResult, s_data + sharedMem.x + sharedMem.y + sharedMem.w );
          int n = TProcedure::NumThreads != 0 ?  hasResult / TProcedure ::NumThreads : (TProcedure ::ItemInput  ? hasResult : 1);
          q-> template finishRead<TProcedure>(execproc[1],  n);
          return true;
      }
      return false;
    }
  };

  #define PROCCALLNOCOPYPART(LAUNCHNUM) \
  template<class Q, class ProcInfo, bool MultiElement> \
  struct ProcCallNoCopyVisitorPart ## LAUNCHNUM \
  { \
    int* execproc; \
    const uint4 & sharedMem; \
    Q* q; \
    void* execData; \
    uint* s_data; \
    int hasResult; \
    __inline__ __device__ ProcCallNoCopyVisitorPart ## LAUNCHNUM  (Q* q, int *execproc, void * execData, uint* s_data, const uint4& sharedMem, int hasResult ) : execproc(execproc), sharedMem(sharedMem), q(q), execData(execData), s_data(s_data), hasResult(hasResult) { }  \
    template<class TProcedure, class CUSTOM>  \
    __device__ __inline__ bool visit()  \
    {  \
      if(*execproc == TProcedure::ProcedureId)  \
      {  \
          FuncCaller<Q, ProcInfo, TProcedure, CUSTOM, TProcedure :: ItemInput, MultiElement>::call(q, execData, hasResult, s_data + sharedMem.x + sharedMem.y + sharedMem.w );   \
          int n = TProcedure::NumThreads != 0 ?  hasResult / TProcedure ::NumThreads : (TProcedure ::ItemInput  ? hasResult : 1); \
          q-> template finishRead ## LAUNCHNUM  <TProcedure>(execproc[1],  n);  \
          return true;  \
      }  \
      return false;   \
    }   \
  };

  PROCCALLNOCOPYPART(1)
  PROCCALLNOCOPYPART(2)
  PROCCALLNOCOPYPART(3)

#undef PROCCALLNOCOPYPART

  extern __device__ int maxConcurrentBlocks;
  extern __device__ volatile int maxConcurrentBlockEvalDone;


  template<class Q, MegakernelStopCriteria StopCriteria, bool Maintainer>
  class MaintainerCaller;

  template<class Q, MegakernelStopCriteria StopCriteria>
  class MaintainerCaller<Q, StopCriteria, true>
  {
  public:
    static __inline__ __device__ bool RunMaintainer(Q* q, int* shutdown)
    {
      
      if(blockIdx.x == 1)
      {
        __shared__ bool run;
        run = true;
        __syncthreads();
        int runs = 0;
        while(run)
        {
          q->globalMaintain();
          __syncthreads();
          if(runs > 10)
          {
            if(endCounter == 0)
            {
              if(StopCriteria == MegakernelStopCriteria::EmptyQueue)
                run = false;
              else if (shutdown)
              {
                if(*shutdown)
                  run = false;
             }
            }
            __syncthreads();
          }
          else
            ++runs;
        }
      }
      return false;
    }
  };
  template<class Q, MegakernelStopCriteria StopCriteria>
  class MaintainerCaller<Q, StopCriteria, false>
  {
  public:
    static __inline__ __device__ bool RunMaintainer(Q* q, int* shutdown)
    {
      return false;
    }
  };

  template<class Q, class PROCINFO, class CUSTOM, bool CopyToShared, bool MultiElement, bool tripleCall>
  class MegakernelLogics;

  template<class Q, class PROCINFO, class CUSTOM, bool MultiElement, bool tripleCall>
  class MegakernelLogics<Q, PROCINFO, CUSTOM, true, MultiElement, tripleCall>
  {
  public:
    static  __device__ __inline__ int  run(Q* q, uint4 sharedMemDist)
    {
      extern __shared__ uint s_data[];
      void* execData = reinterpret_cast<void*>(s_data + sharedMemDist.x + sharedMemDist.w);
      int* execproc = reinterpret_cast<int*>(s_data + sharedMemDist.w);

      int hasResult = q-> template dequeue<MultiElement> (execData, execproc, sizeof(uint)*(sharedMemDist.y + sharedMemDist.z));
      
      __syncthreads();

      if(hasResult)
      {
        ProcCallCopyVisitor<Q, PROCINFO, MultiElement> visitor(q, execproc, execData, s_data, sharedMemDist, hasResult);
        ProcInfoVisitor<PROCINFO, CUSTOM>::template Visit<ProcCallCopyVisitor<Q, PROCINFO, MultiElement> >(visitor);
      }
      return hasResult;
    }
  };

  template<class Q, class PROCINFO, class CUSTOM, bool MultiElement>
  class MegakernelLogics<Q, PROCINFO, CUSTOM, false, MultiElement, false>
  {
  public:
    static  __device__ __inline__ int  run(Q* q, uint4 sharedMemDist)
    {
      extern __shared__ uint s_data[];
      void* execData = reinterpret_cast<void*>(s_data + sharedMemDist.x + sharedMemDist.w);
      int* execproc = reinterpret_cast<int*>(s_data + sharedMemDist.w);

      int hasResult = q-> template dequeueStartRead<MultiElement> (execData, execproc, sizeof(uint)*sharedMemDist.z);
      
      __syncthreads();

      if(hasResult)
      {
        ProcCallNoCopyVisitor<Q, PROCINFO,  MultiElement> visitor(q, execproc, execData, s_data, sharedMemDist, hasResult);
        ProcInfoVisitor<PROCINFO, CUSTOM>::template Visit<ProcCallNoCopyVisitor<Q, PROCINFO, MultiElement> >(visitor);
      }
      return hasResult;
    }
  };

  template<class Q, class PROCINFO, class CUSTOM, bool MultiElement>
  class MegakernelLogics<Q, PROCINFO, CUSTOM, false, MultiElement, true>
  {
  public:
    static  __device__ __inline__ int  run(Q* q, uint4 sharedMemDist)
    {
      extern __shared__ uint s_data[];
      void* execData = reinterpret_cast<void*>(s_data + sharedMemDist.x + sharedMemDist.w);
      int* execproc = reinterpret_cast<int*>(s_data + sharedMemDist.w);

      int hasResult = q-> template dequeueStartRead1<MultiElement> (execData, execproc, sizeof(uint)*sharedMemDist.z);
      
      if(hasResult)
      {
        ProcCallNoCopyVisitorPart1<Q, PROCINFO, MultiElement> visitor(q, execproc, execData, s_data, sharedMemDist, hasResult);
        ProcInfoVisitor<PROCINFO, CUSTOM>::template Visit<ProcCallNoCopyVisitorPart1<Q, PROCINFO, MultiElement> >(visitor);      
        return hasResult;
      }

      hasResult = q-> template dequeueStartRead2<MultiElement> (execData, execproc, sizeof(uint)*sharedMemDist.z);
     
      if(hasResult)
      {
        ProcCallNoCopyVisitorPart2<Q, PROCINFO, MultiElement> visitor(q, execproc, execData, s_data, sharedMemDist, hasResult);
        ProcInfoVisitor<PROCINFO, CUSTOM>::template Visit<ProcCallNoCopyVisitorPart2<Q, PROCINFO, MultiElement> >(visitor);          
        return hasResult;
      }

      hasResult = q-> template dequeueStartRead3<MultiElement> (execData, execproc, sizeof(uint)*sharedMemDist.z);
      
      if(hasResult)
      {
        ProcCallNoCopyVisitorPart3<Q, PROCINFO, MultiElement> visitor(q, execproc, execData, s_data, sharedMemDist, hasResult);
        ProcInfoVisitor<PROCINFO, CUSTOM>::template Visit<ProcCallNoCopyVisitorPart3<Q, PROCINFO, MultiElement> >(visitor);         
      }

      return hasResult;
    }
  };

  template<unsigned long long StaticLimit, bool Dynamic>
  struct TimeLimiter;

  template<>
  struct TimeLimiter<0, false>
  {
    __device__ __inline__ TimeLimiter() { }
    __device__ __inline__ bool stop(int tval)
    {
      return false;
    }
  };

  template<unsigned long long StaticLimit>
  struct TimeLimiter<StaticLimit, false>
  {
    unsigned long long  TimeLimiter_start;
    __device__ __inline__ TimeLimiter() 
    {
      if(threadIdx.x == 0)
        TimeLimiter_start = clock64();
    }
    __device__ __inline__ bool stop(int tval)
    {
      return (clock64() - TimeLimiter_start) > StaticLimit;
    }
  };

  template<>
  struct TimeLimiter<0, true>
  {
    unsigned long long  TimeLimiter_start;
    __device__ __inline__ TimeLimiter() 
    {
      if(threadIdx.x == 0)
        TimeLimiter_start = clock64();
    }
    __device__ __inline__ bool stop(int tval)
    {
      return (clock64() - TimeLimiter_start)/1024 > tval;
    }
  };

  template<class Q, class PROCINFO, class CUSTOM, bool CopyToShared, bool MultiElement, bool Maintainer, class TimeLimiter, MegakernelStopCriteria StopCriteria>
  __global__ void megakernel(Q* q, uint4 sharedMemDist, int t, int* shutdown)
  {
    if(q == 0)
    {
      if(maxConcurrentBlockEvalDone != 0)
        return;
      if(threadIdx.x == 0)
        atomicAdd(&maxConcurrentBlocks, 1);
      DelayFMADS<10000,4>::delay();
      __syncthreads();
      maxConcurrentBlockEvalDone = 1;
      __threadfence();
      return;
    }
    __shared__ volatile int runState;

    if(MaintainerCaller<Q, StopCriteria, Maintainer>::RunMaintainer(q, shutdown))
      return;

    __shared__ TimeLimiter timelimiter;

    if(threadIdx.x == 0)
    {
      if(endCounter == 0)
        runState = 0;
      else
      {
        atomicAdd((int*)&doneCounter,1);
        if(atomicAdd((int*)&endCounter,1) == 2597)
          atomicSub((int*)&endCounter, 2597);
        runState = 1;
      }
    }
    q->workerStart();
    __syncthreads();

    while(runState)
    {
      int hasResult = MegakernelLogics<Q, PROCINFO, CUSTOM, CopyToShared, MultiElement, Q::needTripleCall>::run(q, sharedMemDist);
      if(threadIdx.x == 0)
      {
        if(timelimiter.stop(t))
          runState = 0;
        else if(hasResult)
        {
          if(runState == 3)
          {
            //back on working
            runState = 1;
            atomicAdd((int*)&doneCounter,1);
            atomicAdd((int*)&endCounter,1);
          }
          else if(runState == 2)
          {
            //back on working
            runState = 1;
            atomicAdd((int*)&doneCounter,1);
          }
        }
        else
        {
          //RUNSTATE UPDATES
          if(runState == 1)
          {
            //first time we are out of work
            atomicSub((int*)&doneCounter,1);
            runState = 2;
          }
          else if(runState == 2)
          {
            if(doneCounter == 0)
            {
              //everyone seems to be out of work -> get ready for end
              atomicSub((int*)&endCounter,1);
              runState = 3;
            }
          }
          else if(runState == 3)
          {
            int d = doneCounter;
            int e = endCounter;
            //printf("%d %d %d\n",blockIdx.x, d, e);
            if(doneCounter != 0)
            {
              //someone started to work again
              atomicAdd((int*)&endCounter,1);
              runState = 2;
            }
            else if(endCounter == 0)
            {
              //everyone is really out of work
              if(StopCriteria == MegakernelStopCriteria::EmptyQueue)
                runState = 0;
              else if (shutdown)
              {
                if(*shutdown)
                  runState = 0;
              }
            }
          }
        }
      }

      __syncthreads();
      q->workerMaintain();
    }
    q->workerEnd();
  }




  template<template <class> class QUEUE, class PROCINFO, class ApplicationContext = void, int maxShared = 16336, bool LoadToShared = true, bool MultiElement = true, bool StaticTimelimit  = false, bool DynamicTimelimit = false>
  class TechniqueCore
  {
    friend struct InitPhaseVisitor;
  public:

    typedef MultiPhaseQueue< PROCINFO, QUEUE > Q;

  protected:    
    
    std::unique_ptr<Q, cuda_deleter> q;

    int blockSize[PROCINFO::NumPhases];
    int blocks[PROCINFO::NumPhases];
    uint4 sharedMem[PROCINFO::NumPhases];
    uint sharedMemSum[PROCINFO::NumPhases];

    int freq;

    struct InitPhaseVisitor
    {
      TechniqueCore &technique;
      InitPhaseVisitor(TechniqueCore &technique) : technique(technique) { }
      template<class TProcInfo, class TQueue, int Phase> 
      bool visit()
      {
        technique.blockSize[Phase] = TProcInfo:: template OptimalThreadCount<MultiElement>::Num;
        
        if(TQueue::globalMaintainMinThreads > 0)
         technique.blockSize[Phase] = max(technique.blockSize[Phase],TQueue::globalMaintainMinThreads);

        uint queueSharedMem = TQueue::requiredShared;

        //get shared memory requirement
        technique.sharedMem[Phase] = TProcInfo:: template requiredShared<MultiElement>(technique.blockSize[Phase], LoadToShared, maxShared - queueSharedMem, false);
        //if(!LoadToShared)
        //  sharedMem.x = 16;
        technique.sharedMem[Phase].x /= 4;
        technique.sharedMem[Phase].y = technique.sharedMem[Phase].y/4;
        technique.sharedMem[Phase].z = technique.sharedMem[Phase].z/4;
     
        //x .. procids
        //y .. data
        //z .. shared mem for procedures
        //w .. sum


        //w ... -> shared mem for queues...
        technique.sharedMemSum[Phase] = technique.sharedMem[Phase].w + queueSharedMem;
        technique.sharedMem[Phase].w = queueSharedMem/4;
        
        if(TQueue::globalMaintainMinThreads > 0)
          technique.sharedMemSum[Phase] = max(technique.sharedMemSum[Phase], TQueue::globalMaintainSharedMemory(technique.blockSize[Phase]));

        //get number of blocks to start - gk110 screwes with mutices...
        int nblocks = 0;
        CHECKED_CALL(cudaMemcpyToSymbol(maxConcurrentBlocks, &nblocks, sizeof(int)));
        CHECKED_CALL(cudaMemcpyToSymbol(maxConcurrentBlockEvalDone, &nblocks, sizeof(int)));
        megakernel<TQueue, TProcInfo, ApplicationContext, LoadToShared, MultiElement, (TQueue::globalMaintainMinThreads > 0)?true:false, TimeLimiter<StaticTimelimit?1000:0, DynamicTimelimit>, MegakernelStopCriteria::EmptyQueue> <<<512, technique.blockSize[Phase], technique.sharedMemSum[Phase]>>> (0, technique.sharedMem[Phase], 0, NULL);


        CHECKED_CALL(cudaDeviceSynchronize());
        CHECKED_CALL(cudaMemcpyFromSymbol(&nblocks, maxConcurrentBlocks, sizeof(int)));
        technique.blocks[Phase] = nblocks;
        //std::cout << "blocks: " << blocks << std::endl;
        if(technique.blocks[Phase]  == 0)
          printf("ERROR: in Megakernel confguration: dummy launch failed. Check shared memory consumption?n");
        return false;
      }
    };


    void preCall(cudaStream_t stream)
    {
      int magic = 2597, null = 0;
      CHECKED_CALL(cudaMemcpyToSymbolAsync(doneCounter, &null, sizeof(int), 0, cudaMemcpyHostToDevice, stream));
      CHECKED_CALL(cudaMemcpyToSymbolAsync(endCounter, &magic, sizeof(int), 0, cudaMemcpyHostToDevice, stream));
    }

    void postCall(cudaStream_t stream)
    {
    }

  public:

    void init()
    {
      q = std::unique_ptr<Q, cuda_deleter>(cudaAlloc<Q>());

      int magic = 2597, null = 0;
      CHECKED_CALL(cudaMemcpyToSymbol(doneCounter, &null, sizeof(int)));
      CHECKED_CALL(cudaMemcpyToSymbol(endCounter, &magic, sizeof(int)));

      SegmentedStorage::checkReinitStorage();
      initQueue<Q> <<<512, 512>>>(q.get());
      CHECKED_CALL(cudaDeviceSynchronize());


      InitPhaseVisitor v(*this);
      Q::template staticVisit<InitPhaseVisitor>(v);

      cudaDeviceProp props;
      int dev;
      CHECKED_CALL(cudaGetDevice(&dev));
      CHECKED_CALL(cudaGetDeviceProperties(&props, dev));
      freq = static_cast<int>(static_cast<unsigned long long>(props.clockRate)*1000/1024);
    }

    void resetQueue()
    {
      init();
    }

    void recordQueue()
    {
      if(!Q::supportReuseInit)
        std::cout << "ERROR Megakernel::recordQueue(): queue does not support reuse init\n";
      else
      {
        recordData<Q><<<1, 1>>>(q.get());
        CHECKED_CALL(cudaDeviceSynchronize());
      }
    }

    void restoreQueue()
    {
      if(!Q::supportReuseInit)
        std::cout << "ERROR Megakernel::restoreQueue(): queue does not support reuse init\n";
      else
        resetData<Q><<<1, 1>>>(q.get());
    }


    template<class InsertFunc>
    void insertIntoQueue(int num)
    {
      typedef CurrentMultiphaseQueue<Q, 0> Phase0Q;


      //Phase0Q::pStart();

      //Phase0Q::CurrentPhaseProcInfo::print();

      int b = min((num + 512 - 1)/512,104);
      initData<InsertFunc, Phase0Q><<<b, 512>>>(reinterpret_cast<Phase0Q*>(q.get()), num);
      CHECKED_CALL(cudaDeviceSynchronize());
    }

    int BlockSize(int phase = 0) const
    {
      return blockSize[phase];
    }
    int Blocks(int phase = 0) const
    {
      return blocks[phase];
    }
    uint SharedMem(int phase = 0) const
    {
      return sharedMemSum[phase];
    }

    std::string name() const
    {
      return std::string("Megakernel") + (MultiElement?"Dynamic":"Simple") + (LoadToShared?"":"Globaldata") + ">" + Q::name();
    }

    void release()
    {
      delete this;
    }
  };

  template<template <class> class QUEUE, class PROCINFO, class ApplicationContext = void, MegakernelStopCriteria StopCriteria = EmptyQueue, int maxShared = 16336, bool LoadToShared = true, bool MultiElement = true, bool StaticTimelimit = false, bool DynamicTimelimit = false>
  class Technique;
  
  template<template <class> class QUEUE, class PROCINFO, class ApplicationContext, MegakernelStopCriteria StopCriteria, int maxShared, bool LoadToShared, bool MultiElement>
  class Technique<QUEUE, PROCINFO, ApplicationContext, StopCriteria, maxShared, LoadToShared, MultiElement, false, false> : public TechniqueCore<QUEUE, PROCINFO, ApplicationContext, maxShared, LoadToShared, MultiElement, false, false>
  {
    typedef MultiPhaseQueue< PROCINFO, QUEUE > Q;

    struct LaunchVisitor
    {
      int phase;
      int blocks, blockSize, sharedMemSum;
      uint4 sharedMem;
      Q* q;
      cudaStream_t stream;
      int* shutdown;
      LaunchVisitor(Q* q, int phase, int blocks, int blockSize, int sharedMemSum, uint4 sharedMem, cudaStream_t stream, int* shutdown) :
        phase(phase), blocks(blocks), blockSize(blockSize), sharedMemSum(sharedMemSum), sharedMem(sharedMem), q(q), stream(stream), shutdown(shutdown) { }

      template<class TProcInfo, class TQueue, int Phase> 
      bool visit()
      {
        if(phase == Phase)
        {
          megakernel<TQueue, TProcInfo, ApplicationContext, LoadToShared, MultiElement, (TQueue::globalMaintainMinThreads > 0)?true:false, TimeLimiter<false,false>, StopCriteria><<<blocks, blockSize, sharedMemSum, stream>>> (reinterpret_cast<TQueue*>(q), sharedMem, 0, shutdown);
          return true;
        }
        return false;
      }
    };
  public:
    void execute(int phase = 0, cudaStream_t stream = 0, int* shutdown = NULL)
    {
      typedef TechniqueCore<QUEUE,PROCINFO,ApplicationContext,maxShared,LoadToShared,MultiElement,false,false> TCore;

      TCore::preCall(stream);

      LaunchVisitor v(TCore::q.get(), phase, TCore::blocks[phase], TCore::blockSize[phase], TCore::sharedMemSum[phase], TCore::sharedMem[phase], stream, shutdown);
      Q::template staticVisit<LaunchVisitor>(v);

      TCore::postCall(stream);
    }
  };


  template<template <class> class QUEUE, class PROCINFO, class ApplicationContext, MegakernelStopCriteria StopCriteria, int maxShared, bool LoadToShared, bool MultiElement>
  class Technique<QUEUE, PROCINFO, ApplicationContext, StopCriteria, maxShared, LoadToShared, MultiElement, true, false> : public TechniqueCore<QUEUE, PROCINFO, ApplicationContext, maxShared, LoadToShared, MultiElement, true, false>
  {
    typedef MultiPhaseQueue< PROCINFO, QUEUE > Q;

  public:
    template<int Phase, int TimeLimitInKCycles>
    void execute(cudaStream_t stream = 0, int* shutdown = NULL)
    {
      typedef CurrentMultiphaseQueue<Q, Phase> ThisQ;

      typedef TechniqueCore<QUEUE,PROCINFO,ApplicationContext,maxShared,LoadToShared,MultiElement,true,false> TCore;

      TCore::preCall(stream);

      megakernel<ThisQ, typename ThisQ::CurrentPhaseProcInfo, ApplicationContext, LoadToShared, MultiElement, (ThisQ::globalMaintainMinThreads > 0)?true:false,TimeLimiter<TimeLimitInKCycles,false>, StopCriteria><<<TCore::blocks[Phase], TCore::blockSize[Phase], TCore::sharedMemSum[Phase], stream>>>(TCore::q.get(), TCore::sharedMem[Phase], 0, shutdown);

      TCore::postCall(stream);
    }

    template<int Phase>
    void execute(cudaStream_t stream = 0)
    {
      return execute<Phase, 0>(stream);
    }
  };

  template<template <class> class QUEUE, class PROCINFO, class ApplicationContext, MegakernelStopCriteria StopCriteria, int maxShared, bool LoadToShared, bool MultiElement>
  class Technique<QUEUE, PROCINFO, ApplicationContext, StopCriteria, maxShared, LoadToShared, MultiElement, false, true> : public TechniqueCore<QUEUE, PROCINFO, ApplicationContext, maxShared, LoadToShared, MultiElement, false, true>
  {
    typedef MultiPhaseQueue< PROCINFO, QUEUE > Q;

    struct LaunchVisitor
    {
      int phase;
      int blocks, blockSize, sharedMemSum;
      uint4 sharedMem;
      int timeLimit;
      Q* q;
      int* shutdown;
      LaunchVisitor(Q* q, int phase, int blocks, int blockSize, int sharedMemSum, uint4 sharedMem, int timeLimit, int* shutdown) : phase(phase), blocks(blocks), blockSize(blockSize), sharedMemSum(sharedMemSum), sharedMem(sharedMem), timeLimit(timeLimit), q(q), shutdown(shutdown) { }

      template<class TProcInfo, class TQueue, int Phase> 
      bool visit()
      {
        if(phase == Phase)
        {
          megakernel<TQueue, TProcInfo, ApplicationContext, LoadToShared, MultiElement, (TQueue::globalMaintainMinThreads > 0)?true:false,TimeLimiter<false,true>, StopCriteria><<<blocks, blockSize, sharedMemSum>>>(reinterpret_cast<TQueue*>(q), sharedMem, timeLimit, shutdown);
          return true;
        }
        return false;
      }
    };
  public:
    void execute(int phase = 0, cudaStream_t stream = 0, double timelimitInMs = 0, int* shutdown = NULL)
    {
      typedef TechniqueCore<QUEUE,PROCINFO,ApplicationContext,maxShared,LoadToShared,MultiElement,false,true> TCore;

      TCore::preCall(stream);

      LaunchVisitor v(TCore::q.get(),phase, TCore::blocks[phase], TCore::blockSize[phase], TCore::sharedMemSum[phase], TCore::sharedMem[phase], timelimitInMs/1000*TCore::freq, stream, shutdown);
      Q::template staticVisit<LaunchVisitor>(v);

      TCore::postCall(stream);
    }
  };

  // convenience defines

  template<template <class> class Q, class PROCINFO, class CUSTOM, MegakernelStopCriteria StopCriteria = EmptyQueue, int maxShared = 16336>
  class SimpleShared : public Technique<Q, PROCINFO, CUSTOM, StopCriteria, maxShared, true, false>
  { };
  template<template <class> class Q, class PROCINFO, class CUSTOM, MegakernelStopCriteria StopCriteria = EmptyQueue, int maxShared = 16336>
  class SimplePointed : public Technique<Q, PROCINFO, CUSTOM, StopCriteria, maxShared, false, false>
  { };
  template<template <class> class Q, class PROCINFO, class CUSTOM, MegakernelStopCriteria StopCriteria = EmptyQueue, int maxShared = 16336>
  class DynamicShared : public Technique<Q, PROCINFO, CUSTOM, StopCriteria, maxShared, true, true>
  { };
  template<template <class> class Q, class PROCINFO, class CUSTOM, MegakernelStopCriteria StopCriteria = EmptyQueue, int maxShared = 16336>
  class DynamicPointed : public Technique<Q, PROCINFO, CUSTOM, StopCriteria, maxShared, false, true>
  { };

  template<template <class> class Q, class PROCINFO, class CUSTOM = void, MegakernelStopCriteria StopCriteria = EmptyQueue>
  class SimpleShared16336 : public SimpleShared<Q, PROCINFO, CUSTOM, StopCriteria, 16336>
  { };

    template<template <class> class Q, class PROCINFO, class CUSTOM = void, MegakernelStopCriteria StopCriteria = EmptyQueue>
  class SimpleShared49000: public SimpleShared<Q, PROCINFO, CUSTOM, StopCriteria, 49000>
  { };

  template<template <class> class Q, class PROCINFO, class CUSTOM = void, MegakernelStopCriteria StopCriteria = EmptyQueue>
  class SimplePointed24576 : public SimplePointed<Q, PROCINFO, CUSTOM, StopCriteria, 24576>
  {  };
  template<template <class> class Q, class PROCINFO, class CUSTOM = void, MegakernelStopCriteria StopCriteria = EmptyQueue>
  class SimplePointed16336 : public SimplePointed<Q, PROCINFO, CUSTOM, StopCriteria, 16336>
  {  };
  template<template <class> class Q, class PROCINFO, class CUSTOM = void, MegakernelStopCriteria StopCriteria = EmptyQueue>
  class SimplePointed12000 : public SimplePointed<Q, PROCINFO, CUSTOM, StopCriteria, 12000>
  {  };


  template<template <class> class Q, class PROCINFO, class CUSTOM = void, MegakernelStopCriteria StopCriteria = EmptyQueue>
  class DynamicShared16336 : public DynamicShared<Q, PROCINFO, CUSTOM, StopCriteria, 16336>
  {  };
  template<template <class> class Q, class PROCINFO, class CUSTOM = void, MegakernelStopCriteria StopCriteria = EmptyQueue>
  class DynamicPointed16336 : public DynamicPointed<Q, PROCINFO, CUSTOM, StopCriteria, 16336>
  {  };
  template<template <class> class Q, class PROCINFO, class CUSTOM = void, MegakernelStopCriteria StopCriteria = EmptyQueue>
  class DynamicPointed12000 : public DynamicPointed<Q, PROCINFO, CUSTOM, StopCriteria, 12000>
  {  };
  template<template <class> class Q, class PROCINFO, class CUSTOM = void, MegakernelStopCriteria StopCriteria = EmptyQueue>
  class DynamicPointed11000 : public DynamicPointed<Q,  PROCINFO, CUSTOM, StopCriteria, 11000>
  {  };
}
