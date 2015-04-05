#include <cstdlib>
#include <cublas_v2.h>
#include <iostream>
#include <time.h>
#include <tools/utils.h>

#include "queueDistLocks.cuh"
#include "queueShared.cuh"
#include "queuingPerProc.cuh"
#include "techniqueMegakernel.cuh"
#include "techniqueKernels.cuh"
#include "techniqueDynamicParallelism.cuh"
#include "segmentedStorage.cuh"

#include "procedureInterface.cuh"
#include "procinfoTemplate.cuh"
#include "random.cuh"

namespace Tools
{
	class CublasError : public std::runtime_error
	{
	private:
		static __host__ std::string genErrorString(cublasStatus_t error, const char* file, int line)
		{
			std::string strerror;
			switch (error)
			{
			case CUBLAS_STATUS_NOT_INITIALIZED :
				strerror = "CUBLAS_STATUS_NOT_INITIALIZED";
				break;
			case CUBLAS_STATUS_ALLOC_FAILED :
				strerror = "CUBLAS_STATUS_ALLOC_FAILED";
				break;
			case CUBLAS_STATUS_INVALID_VALUE :
				strerror = "CUBLAS_STATUS_INVALID_VALUE";
				break;
			case CUBLAS_STATUS_ARCH_MISMATCH :
				strerror = "CUBLAS_STATUS_ARCH_MISMATCH";
				break;
			case CUBLAS_STATUS_MAPPING_ERROR :
				strerror = "CUBLAS_STATUS_MAPPING_ERROR";
				break;
			case CUBLAS_STATUS_EXECUTION_FAILED :
				strerror = "CUBLAS_STATUS_EXECUTION_FAILED";
				break;
			case CUBLAS_STATUS_INTERNAL_ERROR :
				strerror = "CUBLAS_STATUS_INTERNAL_ERROR";
				break;
			case CUBLAS_STATUS_NOT_SUPPORTED :
				strerror = "CUBLAS_STATUS_NOT_SUPPORTED";
				break;
			case CUBLAS_STATUS_LICENSE_ERROR :
				strerror = "CUBLAS_STATUS_LICENSE_ERROR";
				break;
			}
		
			return std::string(file) + '(' + std::to_string(static_cast<long long>(line)) + "): error: " + strerror;
		}
	public:
		__host__ CublasError(cublasStatus_t error, const char* file, int line)
		: runtime_error(genErrorString(error, file, line))
		{
		}
	};

	inline __host__ void cublasError(cublasStatus_t error, const char* file, int line)
	{
		if (error != CUBLAS_STATUS_SUCCESS)
			throw CublasError(error, file, line);
	}
}

#define CUBLAS_CHECKED_CALL(call) Tools::cublasError(call, __FILE__, __LINE__)

struct dim2 { uint x, y; };

struct MatmulConfig
{
	float *A, *B, *C;
	size_t n;
	dim2 gridDim_;
};

__constant__ MatmulConfig config;

class MatmulTask : public ::Procedure
{
public:
	static const int NumThreads = BLOCK_SIZE * BLOCK_SIZE;
	static const bool ItemInput = false; // false results in a lvl 1	task
	static const int sharedMemory = 2 * sizeof(float) * NumThreads;	// shared memory requirements 
	
	typedef uint ExpectedData;

	template<class Q, class Context>
	static __device__ __inline__ void execute(int threadId, int numThreads, Q* queue, ExpectedData* ptaskid, volatile uint* shared) 
	{
		float*& A = config.A;
		float*& B = config.B;
		float*& C = config.C;
		size_t& n = config.n;
		dim2& gridDim_ = config.gridDim_;
	
		const uint taskid = *ptaskid;
	
		struct { uint x, y; } blockDim;
		blockDim.x = BLOCK_SIZE;
		blockDim.y = BLOCK_SIZE;
		
		struct { uint x, y; } blockIdx;
		blockIdx_x = taskid % gridDim_.x;
		blockIdx.y = taskid / gridDim_.x;
		
		struct { uint x, y; } threadIdx;
		threadIdx_x = threadId % BLOCK_SIZE;
		threadIdx.y = threadId / BLOCK_SIZE;

		float sum = 0.0f;

#ifndef MATMUL_USE_SHARED
		int ia = (blockDim.y * blockIdx.y + threadIdx.y) * n;
		int ib = blockDim.x * blockIdx_x + threadIdx_x;
		int ic = ia + ib;

		// Multiply two matrices
		for (int k = 0; k < n; k++)
			sum += A [ia + k] * B [ib + k * n];
#else
		// Base indexes inside A and B
		int ia = (blockDim.y * blockIdx.y) * n;
		int ib = blockDim.x * blockIdx_x;
	
		// Subindex inside a "tile"
		int tileidx = n * threadIdx.y + threadIdx_x;
	
		// Index in C
		int ic = ia + ib + tileidx;

		// Shared memory for the "tile" sub-matrix of A and B
		float* As = (float*)shared;
		float* Bs = (float*)shared + BLOCK_SIZE * BLOCK_SIZE;

		// Go through "tiles" of size blockDim.x * blockDim.y
		for (uint aoff = 0, boff = 0; aoff < n; aoff += blockDim.x, boff += blockDim.y * n)
		{
			// Load the "tile" matrices from global memory to shared memory
			As [threadIdx.y * BLOCK_SIZE + threadIdx_x] = A [ia + aoff + tileidx];
			Bs [threadIdx.y * BLOCK_SIZE + threadIdx_x] = B [ib + boff + tileidx];

			// Synchronize to make sure the matrices are loaded
			Context::sync();

			// Multiply the two matrices
			for (int k = 0; k < BLOCK_SIZE; k++)
				sum += As [threadIdx.y * BLOCK_SIZE + k] * Bs [k * BLOCK_SIZE + threadIdx_x];

			// Synchronize to make sure that the preceding
			// computation is done before loading two new
			// sub-matrices of A and B in the next iteration
			Context::sync();
		}
#endif
		// Write the block sub-matrix to global memory
		// each thread writes one element
		C [ic] = sum;
	}

	template<class Q>
	__device__ __inline__ static void init(Q* q, int id)
	{
		q->template enqueueInitial<MatmulTask>(id);
	}
};

enum MatmulVersion
{
	CUBLAS,
	CUDA,
	WHIPPLETREE
};

__global__ void cuda_matmul(float* A, float* B, float* C, size_t n)
{
    float sum = 0.0f;

#ifndef MATMUL_USE_SHARED
	int ia = (blockDim.y * blockIdx.y + threadIdx.y) * n;
	int ib = blockDim.x * blockIdx_x + threadIdx_x;
	int ic = ia + ib;

	// Multiply two matrices
	for (int k = 0; k < n; k++)
		sum += A [ia + k] * B [ib + k * n];
#else
    // Base indexes inside A and B
    int ia = (blockDim.y * blockIdx.y) * n;
    int ib = blockDim.x * blockIdx_x;
    
    // Subindex inside a "tile"
    int tileidx = n * threadIdx.y + threadIdx_x;
    
    // Index in C
    int ic = ia + ib + tileidx;

    int aoff = 0, boff = 0;

    // Shared memory for the "tile" sub-matrix of A and B
    __shared__ float As [BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs [BLOCK_SIZE][BLOCK_SIZE];

    // Go through "tiles" of size blockDim.x * blockDim.y
    for (; aoff < n; aoff += blockDim.x, boff += blockDim.y * n)
    {
        // Load the "tile" matrices from global memory to shared memory
        As [threadIdx.y][threadIdx_x] = A [ia + aoff + tileidx];
        Bs [threadIdx.y][threadIdx_x] = B [ib + boff + tileidx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices
        for (int k = 0; k < BLOCK_SIZE; k++)
            sum += As [threadIdx.y][k] * Bs [k][threadIdx_x];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
#endif
    // Write the block sub-matrix to global memory
    // each thread writes one element
    C [ic] = sum;
}

class Matmul
{
public :
	//lets use a dist locks queue for each procedure, which can hold 12k elements
	template<class ProcInfo>
	class MyQueue : public PerProcedureQueueTyping<QueueDistLocksOpt_t, 96 * 1024, false>::Type<ProcInfo> { };

	//and lets use a Megakernel which can execute multiple workpackages concurrently (dynamic)
	//and offers a maximum of 16k shared memory
	typedef Megakernel::SimplePointed16336<MyQueue, ProcInfo<MatmulTask> > MyTechnique;

	Matmul(float* Ah, float* Bh, float* Ch, size_t n, MatmulVersion version, float* time = NULL)
	{
		MatmulConfig hconfig;
		float*& A = hconfig.A;
		float*& B = hconfig.B;
		float*& C = hconfig.C;
		hconfig.n = n;
		hconfig.gridDim_.x = n / BLOCK_SIZE;
		hconfig.gridDim_.y = n / BLOCK_SIZE;
	
		CHECKED_CALL(cudaMalloc(&A, sizeof(float) * n * n));
		CHECKED_CALL(cudaMalloc(&B, sizeof(float) * n * n));
		CHECKED_CALL(cudaMalloc(&C, sizeof(float) * n * n));

		CHECKED_CALL(cudaMemcpyToSymbol(config, &hconfig, sizeof(MatmulConfig)));

		CHECKED_CALL(cudaMemcpy(A, Ah, sizeof(float) * n * n, cudaMemcpyHostToDevice));
		CHECKED_CALL(cudaMemcpy(B, Bh, sizeof(float) * n * n, cudaMemcpyHostToDevice));

		if (version == MatmulVersion::CUBLAS)
		{		
			cublasHandle_t handle;
			CUBLAS_CHECKED_CALL(cublasCreate(&handle));

			volatile struct timespec start;
			clock_gettime(CLOCK_REALTIME, (struct timespec*)&start);

			float fone = 1.0f, fzero = 0.0f;
			CUBLAS_CHECKED_CALL(cublasSgemm(handle,
				cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_T,
				n, n, n, &fone, A, n, B, n, &fzero, C, n));
			
			CHECKED_CALL(cudaDeviceSynchronize());

			volatile struct timespec finish;
			clock_gettime(CLOCK_REALTIME, (struct timespec*)&finish);

			cublasDestroy(handle);
			
			if (time)
				*time = (float)((double)0.000000001 * (finish.tv_nsec - start.tv_nsec) +
					finish.tv_sec - start.tv_sec);

		}
		if (version == MatmulVersion::CUDA)
		{
			volatile struct timespec start;
			clock_gettime(CLOCK_REALTIME, (struct timespec*)&start);

		    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    		dim3 blocks( n / threads.x, n / threads.y);
			cuda_matmul<<<blocks, threads>>>(A, B, C, n);
			CHECKED_CALL(cudaGetLastError());
			CHECKED_CALL(cudaDeviceSynchronize());

			volatile struct timespec finish;
			clock_gettime(CLOCK_REALTIME, (struct timespec*)&finish);

			if (time)
				*time = (float)((double)0.000000001 * (finish.tv_nsec - start.tv_nsec) +
					finish.tv_sec - start.tv_sec);
		}
		if (version == MatmulVersion::WHIPPLETREE)
		{
			MyTechnique technique;
			technique.init();

			technique.insertIntoQueue<MatmulTask>(hconfig.gridDim_.x * hconfig.gridDim_.y);

			volatile struct timespec start;
			clock_gettime(CLOCK_REALTIME, (struct timespec*)&start);

			technique.execute(0);
			CHECKED_CALL(cudaDeviceSynchronize());

			volatile struct timespec finish;
			clock_gettime(CLOCK_REALTIME, (struct timespec*)&finish);

			if (time)
				*time = (float)((double)0.000000001 * (finish.tv_nsec - start.tv_nsec) +
					finish.tv_sec - start.tv_sec);
		}

		CHECKED_CALL(cudaMemcpy(Ch, C, sizeof(float) * n * n, cudaMemcpyDeviceToHost));

		CHECKED_CALL(cudaFree(A));
		CHECKED_CALL(cudaFree(B));
		CHECKED_CALL(cudaFree(C));
	}
};

int main(int argc, char** argv)
{
	using namespace std;

	if (argc != 2)
	{
		cout << "Usage: " << argv[0] << " <n>" << endl;
		return 1;
	}

	int count;
	CHECKED_CALL(cudaGetDeviceCount(&count));
	if (!count)
	{
		cerr << "No CUDA devices available" << endl;
		return -1;
	}
	cudaDeviceProp deviceProp;
	CHECKED_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	cout << "Using device: " << deviceProp.name << endl;

	size_t n = (size_t)strtoull(argv[1], NULL, 0);
	if (n % BLOCK_SIZE)
	{
		cerr << "For simplisity, we require n (" << n <<
			") to be exact multiplier of BLOCK_SIZE (" <<
			std::to_string(static_cast<long long>(BLOCK_SIZE)) << ")" << endl;
		return -1;
	}

	float *A1 = new float[n * n], *A2 = new float[n * n], *A3 = new float[n * n];
	float *B1 = new float[n * n], *B2 = new float[n * n], *B3 = new float[n * n];
	float *C1 = new float[n * n], *C2 = new float[n * n], *C3 = new float[n * n];

	// Generate random input matrices.
	double dinvrandmax = (double)1.0 / RAND_MAX;
	for (size_t i = 0, length = n * n; i < length; i++)
	{
		A1[i] = rand() * dinvrandmax; A2[i] = A1[i]; A3[i] = A1[i];
		B1[i] = rand() * dinvrandmax; B2[i] = B1[i]; B3[i] = B1[i];
	}
	memset(C1, 0, sizeof(float) * n * n);
	memset(C2, 0, sizeof(float) * n * n);
	memset(C3, 0, sizeof(float) * n * n);

	float time;
	Matmul(A1, B1, C1, n, MatmulVersion::CUBLAS, &time);
	cout << "CUBLAS      version completed in " << time << " sec" << endl;

	Matmul(A2, B2, C2, n, MatmulVersion::CUDA, &time);
	cout << "CUDA        version completed in " << time << " sec" << endl;

	Matmul(A3, B3, C3, n, MatmulVersion::WHIPPLETREE, &time);
	cout << "WHIPPLETREE version completed in " << time << " sec" << endl;

	// Compare C1 and C2 results.
	int status = 0;
	for (int j = 0; j < n; j++)
	{
		for (int i = 0; i < n; i++)
		{
			float c1 = C1[i + j * n];
			float c2 = C2[i * n + j];
			if (fabsf(c1 - c2) > 0.1f)
			{
				cerr << "Mismatching C2 result @ [" << i << "][" << j << "]: " << c1 << " != " << c2 << endl;
				status = -1;
				break;
			}
		}
		if (status == -1) break;
	}

	// Compare C1 and C3 results.
	for (int j = 0; j < n; j++)
	{
		for (int i = 0; i < n; i++)
		{
			float c1 = C1[i + j * n];
			float c3 = C3[i * n + j];
			if (fabsf(c1 - c3) > 0.1f)
			{
				cerr << "Mismatching C3 result @ [" << i << "][" << j << "]: " << c1 << " != " << c3 << endl;
				status = -1;
				break;
			}
		}
		if (status == -1) break;
	}

	delete[] A1; delete[] A2; delete[] A3;
	delete[] B1; delete[] B2; delete[] B3;
	delete[] C1; delete[] C2; delete[] C3;

	return status;
}

