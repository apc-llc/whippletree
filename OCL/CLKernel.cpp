#include "CLKernel.h"
#include <iostream>
#include "tools/utils.h"

CLKernel::CLKernel(const CLProgram& program, const char* name)
{
	std::cout << "Building OpenCL kernel \"" << name << "\"" << std::endl;
	cl_int error;
	m_kernel = clCreateKernel(program.m_program, name, &error);
	CHECKED_CALL(error);
}

CLKernel::~CLKernel()
{
	CHECKED_CALL(clReleaseKernel(m_kernel));
}

size_t CLKernel::maxWorkGroupSize() const
{
	size_t value = 0;
	CHECKED_CALL(clGetKernelWorkGroupInfo(m_kernel, NULL, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &value, NULL));
	return value;
}

size_t CLKernel::preferredWorkGroupSizeMultiple() const
{
	size_t value = 0;
	CHECKED_CALL(clGetKernelWorkGroupInfo(m_kernel, NULL, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &value, NULL));
	return value;
}

void CLKernel::setArg(cl_uint location, const CLBuffer& buffer) const
{
	CHECKED_CALL(clSetKernelArg(m_kernel, location, sizeof(cl_mem), &buffer.m_buffer));
}

void CLKernel::setArg(cl_uint location, size_t size, const void* data) const
{
	CHECKED_CALL(clSetKernelArg(m_kernel, location, size, data));
}

void CLKernel::setArg(cl_uint location, const CLBufferShared& buffer) const
{
	CHECKED_CALL(clSetKernelArg(m_kernel, location, sizeof(cl_mem), &buffer.m_buffer));
}

void CLKernel::setArg(cl_uint location, const CLMem& buffer) const
{
	CHECKED_CALL(clSetKernelArg(m_kernel, location, sizeof(cl_mem), &buffer.m_mem));
}

