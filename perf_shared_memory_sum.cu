#include <iostream>
#include <vector>
#include <chrono>

// Utility to check for CUDA errors
void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(result) << std::endl;
        exit(1);
    }
}

// --- Kernels ---

/**
 * Kernel 1: Naive Reduction (Baseline)
 * This kernel performs reduction directly in global memory.
 * It is very inefficient due to repeated global memory access.
 */
__global__ void reduction_naive(const float* g_input, float* g_output, int n) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // A very simple (and slow) way to do reduction
    // This is just for performance comparison.
    if (i < n) {
        // atomicAdd is needed to avoid race conditions on global memory
        atomicAdd(g_output, g_input[i]);
    }
}


/**
 * Kernel 2: Shared Memory Optimized Reduction
 */
__global__ void reduction_shared_mem(const float* g_input, float* g_output, int n) {
    // 1. Allocate shared memory for this block
    // extern is used for dynamically sized shared memory
    extern __shared__ float s_data[];

    // 2. Load data from global memory into shared memory
    // Each thread loads one element.
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        s_data[tid] = g_input[i];
    } else {
        s_data[tid] = 0;
    }

    // 3. Synchronize all threads in the block to ensure all data is loaded
    __syncthreads();

    // 4. Perform reduction in shared memory
    // Each step halves the number of active threads
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        // Synchronize after each reduction step to avoid race conditions
        __syncthreads();
    }

    // 5. Write the final result of this block back to global memory
    // Only the first thread in each block needs to do this.
    if (tid == 0) {
        g_output[blockIdx.x] = s_data[0];
    }
}


int main() {
    const int N = 1024 * 1024 * 16; // 16 million elements
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // --- Host Data Setup ---
    std::vector<float> h_input(N);
    double cpu_sum = 0.0;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < N; ++i) {
        h_input[i] = 1.0f;
        cpu_sum += h_input[i];
    }
    auto stop_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu - start_cpu);
    

    // --- Device Data Setup ---
    float *d_input, *d_output_naive, *d_output_shared;
    checkCuda(cudaMalloc(&d_input, N * sizeof(float)));
    checkCuda(cudaMalloc(&d_output_naive, sizeof(float))); // Naive version sums to one spot
    checkCuda(cudaMalloc(&d_output_shared, GRID_SIZE * sizeof(float))); // Shared mem version has partial sums

    checkCuda(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    // --- CUDA Events for Timing ---
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    // --- Run Naive Kernel ---
    // Note: The naive kernel is extremely slow and only for comparison.
    // It also uses atomicAdd which has its own performance characteristics.
    
    float h_output_naive = 0.0f;
    checkCuda(cudaMemset(d_output_naive, 0, sizeof(float)));
    checkCuda(cudaEventRecord(start));
    reduction_naive<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_output_naive, N);
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));
    float time_naive;
    checkCuda(cudaEventElapsedTime(&time_naive, start, stop));
    checkCuda(cudaMemcpy(&h_output_naive, d_output_naive, sizeof(float), cudaMemcpyDeviceToHost));
    

    // --- Run Shared Memory Kernel ---
    std::vector<float> h_output_shared(GRID_SIZE);
    size_t shared_mem_bytes = BLOCK_SIZE * sizeof(float);
    
    checkCuda(cudaEventRecord(start));
    // The third argument to <<<...>>> is the dynamic shared memory size
    reduction_shared_mem<<<GRID_SIZE, BLOCK_SIZE, shared_mem_bytes>>>(d_input, d_output_shared, N);
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));
    float time_shared;
    checkCuda(cudaEventElapsedTime(&time_shared, start, stop));
    checkCuda(cudaMemcpy(h_output_shared.data(), d_output_shared, GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // Final reduction of partial sums on CPU
    double gpu_sum_shared = 0.0;
    for (int i = 0; i < GRID_SIZE; ++i) {
        gpu_sum_shared += h_output_shared[i];
    }

    // --- Print Results ---
    std::cout << "--- Performance Comparison ---" << std::endl;
    std::cout << "Array Size: " << N << " elements" << std::endl;
    std::cout << "CPU Execution Time: " << cpu_duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "Naive Kernel Time: " << time_naive << " ms" << std::endl;
    std::cout << "Shared Memory Kernel Time: " << time_shared << " ms" << std::endl;
    std::cout << "\n--- Verification ---" << std::endl;
    std::cout << "CPU Sum: " << cpu_sum << std::endl;
    std::cout << "GPU (Shared Mem) Sum: " << gpu_sum_shared << std::endl;
    std::cout << "GPU (Naive) Sum: " << h_output_naive << std::endl;

    // --- Cleanup ---
    checkCuda(cudaFree(d_input));
    checkCuda(cudaFree(d_output_naive));
    checkCuda(cudaFree(d_output_shared));
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));

    return 0;
}


// extern __shared__ float s_data[];
// __shared__: 这个关键字告诉编译器，s_data数组将被分配在共享内存中。
// extern ... []: 这是一种声明动态大小共享内存数组的技巧。数组的实际大小将在内核启动时通过<<<...>>>的第三个参数传入
// 数据加载
// s_data[tid] = g_input[i];
// 每个线程负责从全局内存中读取一个元素，并将其存入共享内存中自己对应的位置。
// __syncthreads();
// 这是共享内存编程的生命线！ 它是一个同步屏障(barrier)，一个块内的所有线程都必须到达这个点，然后才能继续执行。
// 第一个 __syncthreads() 的作用是：确保所有线程都已经把自己负责的数据从全局内存加载到了共享内存中，然后才能开始下一步的计算。否则，有些快线程可能会去计算还没被慢线程加载进来的数据，导致结果错误。
// 共享内存中的归约循环
// for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
// 这是一个高效的并行归约循环。在每一步迭代中，我们将需要处理的数据量减半。
// if (tid < s)：只让前半部分的线程工作，将后半部分对应的数据加到自己身上。
// 第二个 __syncthreads()：它位于循环内部，同样至关重要。它确保了在一轮相加（例如，步长为128）完成之后，所有线程都完成了自己的加法，然后才能进入下一轮步长更小（例如64）的相加。
// 写回结果
// if (tid == 0)
// 当循环结束时，整个块的局部和就存储在共享内存的第一个元素s_data[0]中。
// 我们只需要让块内的0号线程负责将这个最终的局部和写回到全局内存的g_output数组中。这就把BLOCK_SIZE次全局内存写操作优化为了1次。