#ifndef CLPROGRAM_H__
#define CLPROGRAM_H__

#include "CLContext.h"
#include "CLDevice.h"

class CLProgram
{
	friend class CLKernel;
	
public:
	CLProgram(const char* source);
	CLProgram(const CLContext& context, const CLDevice& device, const char* source);
	virtual ~CLProgram();
	
private:
	cl_program m_program;
	
};

#endif
