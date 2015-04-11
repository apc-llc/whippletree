#ifndef CLPLATFORM_H__
#define CLPLATFORM_H__

#include <vector>
#include <memory>
#include <string>
#include <CL/cl.h>

class CLPlatform
{
	friend class CLDevice;
	
public:

	static CLPlatform& get();

	static std::unique_ptr<CLPlatform> list();
	std::string info(cl_platform_info name) const;

	virtual ~CLPlatform();	
private:

	CLPlatform(cl_platform_id platform_id);
	
	cl_platform_id m_platform_id;

	CLPlatform(CLPlatform const&);
	void operator=(CLPlatform const&);
};

#endif
