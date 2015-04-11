#include "CLBufferShared.h"
#include "tools/utils.h"

CLBufferShared::CLBufferShared(const CLBufferShared& copy) : m_buffer(copy.m_buffer)
{
    CHECKED_CALL(clRetainMemObject(m_buffer));
}

CLBufferShared::CLBufferShared(const CLContext& context, cl_mem_flags flags, cl_GLuint name)
{
    cl_int error;
    m_buffer = clCreateFromGLBuffer(context.m_context, flags, name, &error);
    CHECKED_CALL(error);
}

CLBufferShared::~CLBufferShared()
{
    CHECKED_CALL(clReleaseMemObject(m_buffer));
}
