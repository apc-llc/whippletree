#include "tools/cl_memory.h"
#include <cstdio>

int main(int argc, char* argv[])
{
	CLDevice::setPreferredType(CL_DEVICE_TYPE_GPU);

	std::unique_ptr<GPUMem<int> > scalarPtr = deviceAlloc<int>();
	std::unique_ptr<GPUMem<int> > arrayPtr = deviceAllocArray<int>(1000);
	
	printf("scalar ptr = %p, array ptr = %p\n", scalarPtr->getDeviceAddress(), arrayPtr->getDeviceAddress());

	return 0;
}

