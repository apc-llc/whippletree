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

#ifndef INCLUDED_CL_PTR
#define INCLUDED_CL_PTR

#pragma once

#include <utility>

template <typename T>
class device_ptr
{
private:
	device_ptr(const device_ptr& p);
	device_ptr& operator =(const device_ptr& p);

	T* ptr;

	static void release(T* ptr)
	{
		if (ptr != nullptr)
			cudaFree(ptr);
	}

public:
	explicit device_ptr(T* ptr = nullptr)
		: ptr(ptr)
	{
	}

	device_ptr(device_ptr&& p)
		: ptr(p.ptr)
	{
		p.ptr = nullptr;
	}

	~device_ptr()
	{
		release(ptr);
	}

	device_ptr& operator =(device_ptr&& p)
	{
		std::swap(ptr, p.ptr);
		return *this;
	}

	void release()
	{
		release(ptr);
		ptr = nullptr;
	}

	T** bind()
	{
		release(ptr);
		return &ptr;
	}

	T* unbind()
	{
		T* temp = ptr;
		ptr = nullptr;
		return temp;
	}

	T* operator ->() const { return ptr; }

	T& operator *() const { return *ptr; }

	operator T*() const { return ptr; }

};


#include "OCL/CLMem.h"
#include <map>
#include <memory>
#include "utils.h"

static std::map<void*, CLMem*> allocations;

template<typename T = void>
class GPUMem
{
	std::unique_ptr<CLMem> m_clMem;

public :

	T* getDeviceAddress() { return static_cast<T*>(m_clMem->getDeviceAddress()); }

	GPUMem(CLMem* clMem) : m_clMem(clMem) { }
};

template<typename T>
inline std::unique_ptr<GPUMem<T> > deviceAlloc()
{
	printf("trying to allocate %.2f MB cuda buffer (%zu bytes)\n", sizeof(T) * 1.0 / (1024.0 * 1024.0), sizeof(T));
	CLMem* clMem = CLMem::create(CL_MEM_READ_WRITE, sizeof(T));
	return std::unique_ptr<GPUMem<T> >(new GPUMem<T>(clMem));
}

template<typename T>
inline std::unique_ptr<GPUMem<T> > deviceAllocArray(size_t N)
{
	printf("trying to allocate %.2f MB cuda buffer (%zu * %zu bytes)\n", N * sizeof(T) * 1.0 / (1024.0 * 1024.0), N, sizeof(T));
	CLMem* clMem = CLMem::create(CL_MEM_READ_WRITE, N * sizeof(T));
	return std::unique_ptr<GPUMem<T> >(new GPUMem<T>(clMem));
}

#endif	// INCLUDED_CL_PTR

