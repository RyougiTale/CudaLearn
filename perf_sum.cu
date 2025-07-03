#include <iostream>
#include <vector>
#include <chrono> // For CPU timing

// Define the matrix size (N x N)
#define N 2048*4 // Increased size to better show performance difference

// CUDA Kernel to perform matrix addition on the GPU
__global__ void MatAddGPU(float *A, float *B, float *C)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N) {
        int idx = i * N + j;
        C[idx] = A[idx] + B[idx];
    }
}

// C++ function to perform matrix addition on the CPU
void MatAddCPU(float *A, float *B, float *C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            C[idx] = A[idx] + B[idx];
        }
    }
}

// Function to verify the results
void verifyResult(float *gpu_result, float *cpu_result) {
    std::cout << "Verifying result..." << std::endl;
    for (int i = 0; i < N * N; ++i) {
        if (abs(gpu_result[i] - cpu_result[i]) > 1e-5) {
            std::cout << "Error at index " << i << "!" << std::endl;
            std::cout << "GPU Result: " << gpu_result[i] << ", CPU Result: " << cpu_result[i] << std::endl;
            return;
        }
    }
    std::cout << "Verification Successful! Results match." << std::endl;
}

int main()
{
    // 1. Host (CPU) memory allocation and initialization
    size_t bytes = N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_gpu = (float*)malloc(bytes); // To store GPU result
    float *h_C_cpu = (float*)malloc(bytes); // To store CPU result

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.5f;
        h_B[i] = 2.5f;
    }

    // --- GPU Calculation ---

    // 2. Device (GPU) memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // 3. Copy data from Host to Device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 4. Set up kernel launch configuration
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // --- GPU TIMING START ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // Start timer

    // 4.1 Launch the kernel
    MatAddGPU<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

    cudaEventRecord(stop); // Stop timer
    cudaEventSynchronize(stop); // Wait for the kernel to finish

    float gpu_elapsed_time;
    cudaEventElapsedTime(&gpu_elapsed_time, start, stop); // Get elapsed time in milliseconds
    // --- GPU TIMING END ---

    // 5. Copy result from Device to Host
    cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);

    // --- CPU Calculation ---

    // --- CPU TIMING START ---
    auto start_cpu = std::chrono::high_resolution_clock::now();
    
    MatAddCPU(h_A, h_B, h_C_cpu);

    auto stop_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu - start_cpu);
    // --- CPU TIMING END ---


    // --- Results and Verification ---
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Matrix Size: " << N << "x" << N << std::endl;
    std::cout << "GPU Execution Time: " << gpu_elapsed_time << " ms" << std::endl;
    std::cout << "CPU Execution Time: " << cpu_duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    verifyResult(h_C_gpu, h_C_cpu);

    // --- Free all memory ---
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}