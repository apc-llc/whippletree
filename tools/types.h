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


#ifndef TOOLS_TYPES_INCLUDED
#define TOOLS_TYPES_INCLUDED

typedef unsigned int uint;
typedef unsigned short ushort;

namespace Tools
{
	struct dim
	{
		union
		{
			struct
			{
				uint x, y, z;
			};
			uint d[3];
		};
		dim(uint _x, uint _y = 1, uint _z = 1) :
			x(_x), y(_y), z(_z)
		{
		}
	};
}

#if defined(_OPENCL)
typedef unsigned char uchar;
struct uchar1 { char x; };
struct uchar2 { char x, y; };
struct uchar3 { char x, y, z; };

static uchar1 make_uchar1(uchar x)
{
	uchar1 result;
	result.x = x;
}

static uchar2 make_uchar2(uchar x, uchar y)
{
	uchar2 result;
	result.x = x; result.y = y;
}

static uchar3 make_uchar3(uchar x, uchar y, uchar z)
{
	uchar3 result;
	result.x = x; result.y = y; result.z = z;
}

struct int1 { int x; };
struct int2 { int x, y; };
struct int4 { int x, y, z, w; };

static int1 make_int1(int x)
{
	int1 result;
	result.x = x;
}

static int2 make_int2(int x, int y)
{
	int2 result;
	result.x = x; result.y = y;
}

static int4 make_int4(int x, int y, int z, int w)
{
	int4 result;
	result.x = x; result.y = y; result.z = z; result.w = w;
}

struct uint1 { int x; };
struct uint2 { int x, y; };
struct uint4 { int x, y, z, w; };

static uint1 make_uint1(int x)
{
	uint1 result;
	result.x = x;
}

static uint2 make_uint2(int x, int y)
{
	uint2 result;
	result.x = x; result.y = y;
}

static uint4 make_uint4(int x, int y, int z, int w)
{
	uint4 result;
	result.x = x; result.y = y; result.z = z; result.w = w;
}
#endif

#endif //TOOLS_TYPES_INCLUDED
