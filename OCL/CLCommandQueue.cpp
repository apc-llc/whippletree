#include "CLCommandQueue.h"
#include "tools/utils.h"

CLCommandQueue::CLCommandQueue(cl_command_queue_properties properties) : CLCommandQueue(CLContext::get(), CLDevice::get(), properties)
{
}

CLCommandQueue::CLCommandQueue(const CLContext& context, const CLDevice& device,
                               cl_command_queue_properties properties)
{
    cl_int error;
    m_command_queue = clCreateCommandQueueWithProperties(context.m_context, device.m_device_id, &properties, &error);
    CHECKED_CALL(error);
}

CLCommandQueue::~CLCommandQueue()
{
    CHECKED_CALL(clFinish(m_command_queue));
    CHECKED_CALL(clReleaseCommandQueue(m_command_queue));
}

void CLCommandQueue::finish() const
{
    CHECKED_CALL(clFinish(m_command_queue));
}

void CLCommandQueue::flush() const
{
    CHECKED_CALL(clFlush(m_command_queue));
}

void CLCommandQueue::enqueueReadBuffer(const CLBuffer &buffer, size_t dataSz, void *data) const
{
    CHECKED_CALL(clEnqueueReadBuffer(m_command_queue, buffer.m_buffer, CL_TRUE, 0, dataSz, data, 0, NULL, NULL));
}

void CLCommandQueue::enqueueNDRangeKernel(const CLKernel& kernel,
                                          cl_uint dimensions,
                                          const size_t* globalOffset,
                                          const size_t* globalSize,
                                          const size_t* localSize,
                                          cl_event* event) const
{
    CHECKED_CALL(clEnqueueNDRangeKernel(m_command_queue, kernel.m_kernel,
                           dimensions, globalOffset, globalSize, localSize, 0, nullptr, event));
}

void CLCommandQueue::enqueueAcquireGLBuffer(const CLBufferShared& buffer) const
{
    CHECKED_CALL(clEnqueueAcquireGLObjects(m_command_queue, 1, &buffer.m_buffer, 0, nullptr, nullptr));
}

void CLCommandQueue::enqueueReleaseGLBuffer(const CLBufferShared& buffer) const
{
    CHECKED_CALL(clEnqueueReleaseGLObjects(m_command_queue, 1, &buffer.m_buffer, 0, nullptr, nullptr));
}

