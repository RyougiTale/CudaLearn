#include <iostream>
#include <vector>

// CUDA Kernel Function
__global__ void addKernel(int *c, const int *a, const int *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Main function
int main() {
    const int N = 256;
    std::cout << "Starting vector addition for N = " << N << std::endl;

    std::vector<int> h_a(N);
    std::vector<int> h_b(N);
    std::vector<int> h_c(N);

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    size_t size = N * sizeof(int);

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Copy data from CPU to GPU
    cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice);

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_c, d_a, d_b, N);

    // Copy result back from GPU to CPU
    cudaMemcpy(h_c.data(), d_c, size, cudaMemcpyDeviceToHost);

    // Verification
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            std::cout << "Error at index " << i << "!" << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Calculation successful!" << std::endl;
    }

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}