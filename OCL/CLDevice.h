#ifndef CLDEVICE_H__
#define CLDEVICE_H__

#include <vector>
#include <string>
#include <CL/cl.h>

#include "CLPlatform.h"

class CLDevice
{
	friend class CLCommandQueue;
	friend class CLContext;
	friend class CLProgram;
	
public:

	static CLDevice& get();

	static cl_device_type getPreferredType();
	static void setPreferredType(cl_device_type type);

	static std::unique_ptr<CLDevice> list(const CLPlatform& platform, cl_device_type type);
	std::string info(cl_device_info name) const;

	virtual ~CLDevice();
	
private:

	CLDevice(cl_device_id device_id);
	
	cl_device_id m_device_id;

	CLDevice();
	CLDevice(CLDevice const&);
	void operator=(CLDevice const&);
};

#endif
