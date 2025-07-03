#include <iostream>
#include <vector>

// Define the matrix size (N x N)
// NOTE: Must be a multiple of the block size (16) for this simple example.
#define N 1024

// CUDA Kernel to perform matrix addition
__global__ void MatAdd(float *A, float *B, float *C)
{
    // Calculate the global row and column index for the thread
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds to avoid writing outside the matrix
    if (i < N && j < N) {
        // Flatten the 2D index to a 1D index for memory access
        int idx = i * N + j;
        C[idx] = A[idx] + B[idx];
    }
}

// Function to verify the results on the CPU
void verifyResult(float *h_C, float *h_A, float *h_B) {
    std::cout << "Verifying result..." << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            if (abs(h_C[idx] - (h_A[idx] + h_B[idx])) > 1e-5) {
                std::cout << "Error at index (" << i << ", " << j << ")" << std::endl;
                std::cout << "GPU Result: " << h_C[idx] << ", CPU Expected: " << (h_A[idx] + h_B[idx]) << std::endl;
                return;
            }
        }
    }
    std::cout << "Verification Successful!" << std::endl;
}

// Main CPU function
int main()
{
    // 1. Host (CPU) memory allocation and initialization
    size_t bytes = N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 2. Device (GPU) memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // 3. Copy data from Host to Device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 4. Set up grid and block dimensions and launch the kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    std::cout << "Launching kernel..." << std::endl;
    MatAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);
    
    // Synchronize to ensure the kernel has finished
    cudaDeviceSynchronize();
    std::cout << "Kernel execution finished." << std::endl;

    // 5. Copy result from Device to Host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // 6. Verify the result
    verifyResult(h_C, h_A, h_B);

    // 7. Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}