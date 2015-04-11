#include "CLPlatform.h"
#include <cstdlib>
#include <iostream>
#include "tools/utils.h"

CLPlatform& CLPlatform::get()
{
	static std::unique_ptr<CLPlatform> platforms = CLPlatform::list();
	if (platforms.get() == NULL)
	{
		std::cerr << "No OpenCL platforms available" << std::endl;
		exit(1);
	}
	return platforms.get()[0];
}

CLPlatform::CLPlatform(cl_platform_id platform_id) : m_platform_id(platform_id)
{
}

CLPlatform::~CLPlatform()
{
}

std::unique_ptr<CLPlatform> CLPlatform::list()
{
	cl_uint platformCount = 0;
	CHECKED_CALL(clGetPlatformIDs(0, NULL, &platformCount));
	CLPlatform* platforms = NULL;
	if (platformCount == 0)
		return std::unique_ptr<CLPlatform>(platforms);
	std::vector<cl_platform_id> platformIDs(platformCount);
	CHECKED_CALL(clGetPlatformIDs(platformCount, platformIDs.data(), NULL));
	platforms = (CLPlatform*)malloc(sizeof(CLPlatform) * platformCount);
	for (int index = 0; index < platformCount; ++index)
		new (platforms + index) CLPlatform(platformIDs[index]);
	return std::unique_ptr<CLPlatform>(platforms);
}

std::string CLPlatform::info(cl_platform_info name) const
{
	size_t valueLength;
	CHECKED_CALL(clGetPlatformInfo(m_platform_id, name, 0, NULL, &valueLength));
	std::vector<char> valueBuffer(valueLength);
	CHECKED_CALL(clGetPlatformInfo(m_platform_id, name, valueLength, valueBuffer.data(), NULL));
	return valueBuffer.data();
}

