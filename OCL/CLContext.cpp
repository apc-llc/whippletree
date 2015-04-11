#include "CLContext.h"
#include <iostream>
#include "tools/utils.h"

CLContext& CLContext::get()
{
	static CLContext context;
	return context;
}

CLContext::CLContext() : CLContext(CLDevice::get())
{
}

CLContext::CLContext(const CLDevice& device) : CLContext(nullptr, device)
{
}

CLContext::CLContext(const cl_context_properties* properties, const CLDevice& device)
{
	cl_int error;
	m_context = clCreateContext(properties, 1, &device.m_device_id, &printError, nullptr, &error);
	CHECKED_CALL(error);
}

CLContext::~CLContext()
{
	CHECKED_CALL(clReleaseContext(m_context));
}

void CLContext::printError(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
	std::cout << "Context Error : " << errinfo << std::endl;
}
