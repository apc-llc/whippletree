#include "CLDevice.h"
#include <cstdlib>
#include <iostream>
#include "tools/utils.h"

namespace
{
	static cl_device_type m_type = CL_DEVICE_TYPE_DEFAULT;
}

cl_device_type CLDevice::getPreferredType()
{
	return m_type;
}

void CLDevice::setPreferredType(cl_device_type type)
{
	m_type = type;
}

CLDevice& CLDevice::get()
{
	static cl_device_type type = CLDevice::getPreferredType();
	static std::unique_ptr<CLDevice> devices(nullptr);
	if ((devices.get() == NULL) || (type != CLDevice::getPreferredType()))
	{
		cl_device_type type = CLDevice::getPreferredType();
		devices = CLDevice::list(CLPlatform::get(), type);
		m_type = CLDevice::getPreferredType();
	}
	if (devices.get() == NULL)
	{
		std::cerr << "No OpenCL devices for selected platform" << std::endl;
		exit(1);
	}
	return devices.get()[0];
}

CLDevice::CLDevice(cl_device_id device_id) : m_device_id(device_id)
{
}

CLDevice::~CLDevice()
{
}

std::unique_ptr<CLDevice> CLDevice::list(const CLPlatform& platform, cl_device_type type)
{
	cl_uint deviceCount = 0;
	CHECKED_CALL(clGetDeviceIDs(platform.m_platform_id, type, 0, NULL, &deviceCount));
	CLDevice* devices = NULL;
	if (deviceCount == 0)
		return std::unique_ptr<CLDevice>(devices);
	std::vector<cl_device_id> deviceIDs(deviceCount);
	CHECKED_CALL(clGetDeviceIDs(platform.m_platform_id, type, deviceCount, deviceIDs.data(), NULL));
	devices = (CLDevice*)malloc(sizeof(CLDevice) * deviceCount);
	for (int index = 0; index < deviceCount; ++index)
		new (devices + index) CLDevice(deviceIDs[index]);
	return std::unique_ptr<CLDevice>(devices);
}

std::string CLDevice::info(cl_device_info name) const
{
	size_t valueLength;
	CHECKED_CALL(clGetDeviceInfo(m_device_id, name, 0, NULL, &valueLength));
	std::vector<char> valueBuffer(valueLength);
	CHECKED_CALL(clGetDeviceInfo(m_device_id, name, valueLength, valueBuffer.data(), NULL));
	std::string value(valueBuffer.data());
	return value;
}

