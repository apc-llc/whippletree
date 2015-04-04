//  Project Whippletree
//  http://www.icg.tugraz.at/project/parallel
//
//  Copyright (C) 2014 Institute for Computer Graphics and Vision,
//                     Graz University of Technology
//
//  Author(s):  Markus Steinberger - steinberger ( at ) icg.tugraz.at
//              Michael Kenzel - kenzel ( at ) icg.tugraz.at
//              Pedro Boechat - boechat ( at ) icg.tugraz.at
//              Bernhard Kerbl - kerbl ( at ) icg.tugraz.at
//              Mark Dokter - dokter ( at ) icg.tugraz.at
//              Dieter Schmalstieg - schmalstieg ( at ) icg.tugraz.at
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE.
//

#ifndef TOOLS_UTILS_INCLUDED
#define TOOLS_UTILS_INCLUDED

#include <cstdio>
#if defined(_OPENCL)
#include <CL/cl.h>
#endif

#define CUDA_CHECKED_CALL(x) do { cudaError_t err = x; if (( err ) != cudaSuccess ) { \
	printf ("Error \"%s\" at %s :%d \n" , cudaGetErrorString(err), \
		__FILE__ , __LINE__ ) ; exit(-1); \
}} while (0)

#define CL_CHECKED_CALL(x) do { cl_int err = x; if (( err ) != CL_SUCCESS ) { \
	printf ("Error \"%d\" at %s :%d \n" , err, \
		__FILE__ , __LINE__ ) ; exit(-1); \
}} while (0)

#if defined(_CUDA)
#define CHECKED_CALL CUDA_CHECKED_CALL
#elif defined(_OPENCL)
#define CHECKED_CALL CL_CHECKED_CALL
#endif

#endif  // TOOLS_UTILS_INCLUDED
