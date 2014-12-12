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

To start the example run cmake in the example folder. Three different procedures are defined
using proc0.cuh, proc1.cuh and proc2.cuh. The host control logic is found in test.cu.