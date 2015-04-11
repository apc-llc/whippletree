#include "CLCommandQueue.h"
#include "CLKernel.h"
#include "CLMem.h"
#include "CLProgram.h"
#include <iostream>
#include "clcode.h"
#include "tools/utils.h"

CLMem::CLMem(cl_mem buffer) : m_mem(buffer)
{
}

CLMem::CLMem(const CLMem& copy) : m_mem(copy.m_mem)
{
	CHECKED_CALL(clRetainMemObject(m_mem));
}

CLMem::~CLMem()
{
	CHECKED_CALL(clReleaseMemObject(m_mem));
}

void* CLMem::getDeviceAddress()
{
	static cl_device_type type = CLDevice::getPreferredType();
	static std::unique_ptr<CLKernel> kernel;
	static CLCommandQueue queue;
	static cl_event event;
	static std::unique_ptr<CLMem> output(CLMem::create(CL_MEM_ALLOC_HOST_PTR, sizeof(unsigned int) * 2));
	size_t one = 1;
	if ((kernel.get() == NULL) || (type != CLDevice::getPreferredType()))
	{
		static std::string devaddr = STRINGIFY_CODE(
			__kernel void devaddr(__global void* input, __global unsigned int* output)
			{
				output[0] = ((unsigned int*)&input)[0];
				output[1] = 0;
				if (sizeof(void*) > sizeof(int))
					output[1] = ((unsigned int*)&input)[1];
			}
		);
		static CLProgram program = CLProgram(devaddr.c_str());
		kernel.reset(new CLKernel(program, "devaddr"));
		kernel->setArg(0, *this);
		kernel->setArg(1, *output.get());
	}

	queue.enqueueNDRangeKernel(*kernel.get(), 1, NULL, &one, &one, &event);
	CHECKED_CALL(clWaitForEvents(1, &event));

	void* result;
	output->read(queue, CL_TRUE, 0, sizeof(void*), &result);

	return result;
}

CLMem* CLMem::create(cl_mem_flags flags, size_t dataSz, void* data)
{
	cl_int error;
	cl_mem buffer = clCreateBuffer(CLContext::get().m_context, flags, dataSz, data, &error);
	CHECKED_CALL(error);
	return new CLMem(buffer);
}

CLMem* CLMem::create(const CLContext& context, cl_mem_flags flags, size_t dataSz, void* data)
{
	cl_int error;
	cl_mem buffer = clCreateBuffer(context.m_context, flags, dataSz, data, &error);
	CHECKED_CALL(error);
	return new CLMem(buffer);
}

CLMem* CLMem::createFromGLBuffer(const CLContext& context, cl_mem_flags flags, cl_GLuint name)
{
	cl_int error;
	cl_mem buffer = clCreateFromGLBuffer(context.m_context, flags, name, &error);
	CHECKED_CALL(error);
	return new CLMem(buffer);
}

CLMem* CLMem::createFromGLTexture(const CLContext& context, cl_mem_flags flags,
	cl_GLenum target, cl_GLint miplevel, cl_GLuint texture)
{
	cl_int error;
	cl_mem buffer = clCreateFromGLTexture(context.m_context, flags, target, miplevel, texture, &error);
	CHECKED_CALL(error);
	return new CLMem(buffer);
}

void CLMem::read(CLCommandQueue& queue,
	cl_bool blocking_read, size_t offset, size_t cb,
	void *ptr, cl_uint num_events_in_wait_list,
	const cl_event *event_wait_list, cl_event *event)
{
	CHECKED_CALL(clEnqueueReadBuffer(queue.m_command_queue, m_mem, blocking_read,
		offset, cb, ptr, num_events_in_wait_list, event_wait_list, event));
}

