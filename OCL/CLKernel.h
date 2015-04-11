#ifndef CLKERNEL_H__
#define CLKERNEL_H__

#include "CLProgram.h"
#include "CLBuffer.h"
#include "CLBufferShared.h"
#include "CLMem.h"

class CLKernel
{
    friend class CLCommandQueue;
    
public:
    CLKernel(const CLProgram& program, const char* name);
    virtual ~CLKernel();
    
    size_t maxWorkGroupSize() const;
    size_t preferredWorkGroupSizeMultiple() const;
    void setArg(cl_uint index, const CLBuffer& buffer) const;
    void setArg(cl_uint index, size_t size, const void* data) const;
    void setArg(cl_uint index, const CLBufferShared& buffer) const;
    void setArg(cl_uint index, const CLMem& buffer) const;
    
private:
    cl_kernel m_kernel;
    
};

#endif
