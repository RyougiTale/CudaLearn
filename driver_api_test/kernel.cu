#include <cuda_runtime.h>

// 防止 Name Mangling
extern "C" __global__ void vecAdd_kernel(float* C, const float* A, const float* B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}


// nvcc -ptx kernel.cu -o vecAdd.ptx