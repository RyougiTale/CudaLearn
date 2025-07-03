#include <cooperative_groups.h>
#include <iostream>
#include <vector>

namespace cg = cooperative_groups;

// The size of our data array.
// For simplicity, make it a multiple of block size and cluster size.
constexpr int NUM_ELEMENTS = 1024 * 1024;
constexpr int BLOCK_SIZE = 256;
constexpr int CLUSTER_SIZE_X = 4; // We'll have 4 blocks per cluster.

// A simple utility to check for CUDA errors
void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(result) << std::endl;
        exit(1);
    }
}

/**
 * Kernel to perform parallel reduction using Thread Block Clusters.
 * * Each block first reduces a chunk of the input array into its standard
 * shared memory. Then, the blocks within the cluster cooperate using
 * Distributed Shared Memory (DSM) to perform a final reduction.
 *
 * We define a cluster of size {4, 1, 1} at compile time.
 */
__global__ void __cluster_dims__(CLUSTER_SIZE_X, 1, 1)
    reduction_cluster_kernel(const float* g_input, float* g_output) {

    // Get the cluster handle for synchronization and coordination
    cg::cluster_group cluster = cg::this_cluster();

    // Allocate standard shared memory, private to each thread block
    __shared__ float s_partials[BLOCK_SIZE];

    // --- Stage 1: Intra-Block Reduction (Same as a standard reduction) ---

    // Each thread loads an element from global memory to shared memory
    // This uses a grid-stride loop to handle large inputs.
    unsigned int tid = threadIdx.x;
    unsigned int global_start_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int grid_stride = gridDim.x * blockDim.x;

    float my_sum = 0.0f;
    for (unsigned int i = global_start_idx; i < NUM_ELEMENTS; i += grid_stride) {
        my_sum += g_input[i];
    }
    s_partials[tid] = my_sum;
    __syncthreads(); // Sync within the block

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_partials[tid] += s_partials[tid + s];
        }
        __syncthreads();
    }

    // --- Stage 2: Inter-Block Reduction using Distributed Shared Memory ---

    // The partial sum for this block is in s_partials[0]
    // We now use DSM to share these partial sums across the cluster.
    // cudaGetClusterDsmHandle() gets a pointer to the DSM region.
    void* dsm_handle;
    cudaGetClusterDsmHandle(&dsm_handle);
    float* dsm_partials = (float*)dsm_handle;

    // Block 0 of each cluster will write its partial sum into DSM.
    // The location in DSM is determined by the block's rank in the cluster.
    if (tid == 0) {
        dsm_partials[cluster.block_rank()] = s_partials[0];
    }

    // Synchronize all blocks within the cluster. This ensures that all
    // blocks have written their partial sums to DSM before we proceed.
    cluster.sync();

    // --- Stage 3: Final Reduction and Write-out ---

    // Block 0 of the entire cluster (block_rank() == 0) is responsible
    // for summing the results from DSM and writing the final answer.
    if (cluster.block_rank() == 0 && tid == 0) {
        float final_sum = 0.0f;
        // The cluster size can be queried at runtime.
        for (unsigned int i = 0; i < cluster.num_blocks(); ++i) {
            final_sum += dsm_partials[i];
        }
        *g_output = final_sum;
    }
}

int main() {
    std::cout << "Starting Cluster Reduction Demo..." << std::endl;

    // Verify the device supports clusters (Compute Capability 9.0+)
    int device_id;
    checkCuda(cudaGetDevice(&device_id));
    cudaDeviceProp props;
    checkCuda(cudaGetDeviceProperties(&props, device_id));
    if (props.major < 9) {
        std::cout << "This program requires a GPU with Compute Capability 9.0 or higher." << std::endl;
        return 0;
    }
    std::cout << "GPU: " << props.name << " (Compute Capability " << props.major << "." << props.minor << ")" << std::endl;


    // 1. Allocate and initialize host memory
    std::vector<float> h_input(NUM_ELEMENTS);
    float h_output = 0.0f;
    double cpu_sum = 0.0;
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        h_input[i] = 1.0f; // Each element is 1.0, so the sum should be NUM_ELEMENTS
        cpu_sum += h_input[i];
    }
    
    // 2. Allocate device memory
    float *d_input, *d_output;
    checkCuda(cudaMalloc(&d_input, NUM_ELEMENTS * sizeof(float)));
    checkCuda(cudaMalloc(&d_output, sizeof(float)));

    // 3. Copy data from host to device
    checkCuda(cudaMemcpy(d_input, h_input.data(), NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));

    // 4. Launch the kernel
    dim3 threadsPerBlock(BLOCK_SIZE);
    // Grid dimension must be a multiple of the cluster size.
    dim3 numBlocks(NUM_ELEMENTS / BLOCK_SIZE);

    std::cout << "Launching kernel on a grid of " << numBlocks.x << " blocks, with " << CLUSTER_SIZE_X << " blocks per cluster." << std::endl;

    // Launch the kernel with its compile-time cluster dimensions.
    reduction_cluster_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    // 5. Copy result back to host
    checkCuda(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    // 6. Verify the result
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "GPU Sum: " << h_output << std::endl;
    std::cout << "CPU Sum: " << cpu_sum << std::endl;
    if (abs(h_output - cpu_sum) < 1e-5) {
        std::cout << "Result is CORRECT." << std::endl;
    } else {
        std::cout << "Result is INCORRECT." << std::endl;
    }
    std::cout << "----------------------------------------" << std::endl;

    // 7. Free device memory
    checkCuda(cudaFree(d_input));
    checkCuda(cudaFree(d_output));

    return 0;
}