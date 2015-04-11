#include "CLBuffer.h"
#include "tools/utils.h"

CLBuffer::CLBuffer(const CLBuffer& copy) : m_buffer(copy.m_buffer)
{
    CHECKED_CALL(clRetainMemObject(m_buffer));
}

CLBuffer::CLBuffer(const CLContext& context, cl_mem_flags flags, size_t dataSz, void* data)
{
    cl_int error;
    m_buffer = clCreateBuffer(context.m_context, flags, dataSz, data, &error);
    CHECKED_CALL(error);
}

CLBuffer::~CLBuffer()
{
    CHECKED_CALL(clReleaseMemObject(m_buffer));
}
