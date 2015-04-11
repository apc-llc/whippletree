all: devaddr

devaddr: devaddr.cpp ../../OCL/*.cpp
	g++ -g -O0 -D_OPENCL -std=c++11 -I../.. -I/opt/AMDAPPSDK-3.0-0-Beta/include $^ -o $@ -L/opt/AMDAPPSDK-3.0-0-Beta/lib/x86_64/ -lOpenCL

clean:
	rm -rf devaddr

