#ifndef CLMEM_H__
#define CLMEM_H__

#include <CL/cl.h>
#include <CL/cl_gl.h>

#include "CLContext.h"

class CLCommandQueue;

class CLMem
{	
	friend class CLCommandQueue;
	friend class CLKernel;
	
public:
	CLMem(cl_mem buffer);
	CLMem(const CLMem& copy);
	virtual ~CLMem();

	void* getDeviceAddress();

	static CLMem* create(cl_mem_flags flags, size_t dataSz, void* data = NULL);	
	static CLMem* create(const CLContext& context, cl_mem_flags flags, size_t dataSz, void* data = NULL);
	static CLMem* createFromGLBuffer(const CLContext& context, cl_mem_flags flags, cl_GLuint name);
	static CLMem* createFromGLTexture(const CLContext& context, cl_mem_flags flags,
		cl_GLenum target, cl_GLint miplevel, cl_GLuint texture);

	void read(CLCommandQueue& queue,
		cl_bool blocking_read, size_t offset, size_t cb,
		void *ptr, cl_uint num_events_in_wait_list = 0,
		const cl_event *event_wait_list = NULL, cl_event *event = NULL);
	
private:
	cl_mem m_mem;
	
};

#endif

