## Overview

We present Whippletree, a novel approach to scheduling dynamic, irregular workloads on the GPU.
We introduce a new programming model which offers the simplicity and expressiveness of task-based
parallelism while retaining all aspects of the multi-level execution hierarchy essential to 
unlocking the full potential of a modern GPU. At the same time, our programming model lends 
itself to efficient implementation on the SIMD-based architecture typical of a current GPU. 
We demonstrate the practical utility of our model by providing a reference implementation on top 
of current CUDA hardware. Furthermore, we show that our model compares favorably to traditional 
approaches in terms of both performance as well as the range of applications that can be covered. 
We demonstrate the benefits of our model for recursive Reyes rendering, procedural geometry 
generation and volume rendering with concurrent irradiance caching.

## Paper reference

Markus Steinberger, Michael Kenzel, Pedro Boechat, Bernhard Kerbl, Mark Dokter, and Dieter Schmalstieg.
Whippletree: Task-based Scheduling of Dynamic Workloads on the GPU.
ACM Transactions on Graphics (Proc. SIGGRAPH Asia 2014), December 2014. To appear. [I'm an inline-style link with title](`http://data.icg.tugraz.at/~dieter/publications/Schmalstieg_286.pdf ".pdf")

## Getting started

Clone the source tree and build basic examples:

```
$ git clone https://github.com/apc-llc/whippletree.git
$ cd whippletree/example
$ mkdir build
$ cd build
$ cmake ..
```

Three different procedures are defined using `proc0.cuh`, `proc1.cuh` and `proc2.cuh`. The host control logic is found in `test.cu`.

