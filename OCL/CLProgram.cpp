#include "CLProgram.h"
#include <iostream>
#include <vector>
#include "tools/utils.h"

CLProgram::CLProgram(const char* source) : CLProgram(CLContext::get(), CLDevice::get(), source)
{
}

CLProgram::CLProgram(const CLContext& context, const CLDevice& device, const char* source)
{
	cl_int error;
	m_program = clCreateProgramWithSource(context.m_context, 1, &source, NULL, &error);
	CHECKED_CALL(error);
	error = clBuildProgram(m_program, 1, &device.m_device_id, NULL, NULL, NULL);
	if (error != CL_SUCCESS)
	{
		size_t size = 0;
		CHECKED_CALL(clGetProgramBuildInfo(m_program, device.m_device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &size));
		std::vector<char> log(size);
		CHECKED_CALL(clGetProgramBuildInfo(m_program, device.m_device_id, CL_PROGRAM_BUILD_LOG, size, &log[0], NULL));
		std::cout << (std::string)(char*)&log[0] << std::endl;
	}	
	CHECKED_CALL(error);
}

CLProgram::~CLProgram()
{
	CHECKED_CALL(clReleaseProgram(m_program));
}
