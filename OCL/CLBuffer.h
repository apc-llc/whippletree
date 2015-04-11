#ifndef CLBUFFER_H__
#define CLBUFFER_H__

#include "CLContext.h"

class CLBuffer
{
    friend class CLCommandQueue;
    friend class CLKernel;
    
public:
    CLBuffer(const CLBuffer& copy);
    CLBuffer(const CLContext& context, cl_mem_flags flags, size_t dataSz, void* data);
    virtual ~CLBuffer();
    
private:
    cl_mem m_buffer;
    
};

#endif
