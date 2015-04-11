#ifndef CLCOMMANDQUEUE_H__
#define CLCOMMANDQUEUE_H__

#include <CL/cl.h>

#include "CLBuffer.h"
#include "CLBufferShared.h"
#include "CLContext.h"
#include "CLDevice.h"
#include "CLKernel.h"

class CLCommandQueue
{
	friend class CLMem;
	
public:

	CLCommandQueue(cl_command_queue_properties properties = 0);
	CLCommandQueue(const CLContext& context, const CLDevice& device, cl_command_queue_properties properties = 0);
	virtual ~CLCommandQueue();
	
	void finish() const;
	void flush() const;
	void enqueueReadBuffer(const CLBuffer& buffer, size_t dataSz, void* data) const;
	void enqueueNDRangeKernel(
		const CLKernel& kernel,
		cl_uint dimensions,
		const size_t* globalOffset,
		const size_t* globalSize,
		const size_t* localSize,
		cl_event* event = nullptr) const;
	void enqueueAcquireGLBuffer(const CLBufferShared& buffer) const;
	void enqueueReleaseGLBuffer(const CLBufferShared& buffer) const;
	
private:
	cl_command_queue m_command_queue;
	
};

#endif
