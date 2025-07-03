#include <iostream>
#include <vector>

// Utility to check for CUDA errors
void checkCuda(cudaError_t result, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << " code=" << result
                  << " \"" << cudaGetErrorString(result) << "\"" << std::endl;
        exit(99);
    }
}
#define checkCudaErrors(val) checkCuda((val), __FILE__, __LINE__)

// A simple kernel to perform an element-wise operation
__global__ void process_chunk(float* data, int chunkSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < chunkSize) {
        data[i] = data[i] * 2.0f; // Example operation: double the value
    }
}

int main() {
    // --- Verify Hardware Support ---
    // This code works on all CUDA-enabled GPUs, including your MX250.

    // --- Setup ---
    const int N = 1024 * 1024 * 8; // 8 Million elements
    const int BLOCK_SIZE = 256;
    
    // We will process the data in chunks to demonstrate overlap
    const int NUM_CHUNKS = 8;
    const int CHUNK_SIZE = N / NUM_CHUNKS;
    size_t chunkBytes = CHUNK_SIZE * sizeof(float);

    // --- Host Memory ---
    // Use Pinned Memory for true asynchronous transfers
    float *h_input, *h_output;
    checkCudaErrors(cudaHostAlloc(&h_input, N * sizeof(float), cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc(&h_output, N * sizeof(float), cudaHostAllocDefault));

    // Initialize host data
    for(int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    // --- Device Memory ---
    float *d_data;
    checkCudaErrors(cudaMalloc(&d_data, N * sizeof(float)));

    // 1. Create a CUDA Stream
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    std::cout << "Processing " << NUM_CHUNKS << " chunks using a CUDA Stream..." << std::endl;
    
    // 2. Process data chunk by chunk on the stream
    for (int i = 0; i < NUM_CHUNKS; ++i) {
        int offset = i * CHUNK_SIZE;

        // 2a. Asynchronously copy chunk from Host to Device on the stream
        checkCudaErrors(cudaMemcpyAsync(d_data + offset, h_input + offset, chunkBytes, cudaMemcpyHostToDevice, stream));
        
        // 2b. Launch the kernel to process the chunk on the same stream
        dim3 grid(CHUNK_SIZE / BLOCK_SIZE, 1, 1);
        dim3 block(BLOCK_SIZE, 1, 1);
        process_chunk<<<grid, block, 0, stream>>>(d_data + offset, CHUNK_SIZE);

        // 2c. Asynchronously copy the processed chunk from Device to Host on the stream
        checkCudaErrors(cudaMemcpyAsync(h_output + offset, d_data + offset, chunkBytes, cudaMemcpyDeviceToHost, stream));
    }

    // 3. Synchronize the stream
    // The CPU host code will wait here until all operations issued to the stream are complete.
    checkCudaErrors(cudaStreamSynchronize(stream));
    std::cout << "All stream operations complete." << std::endl;

    // --- Verification ---
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (abs(h_output[i] - (h_input[i] * 2.0f)) > 1e-5) {
            std::cout << "Error at index " << i << "!" << std::endl;
            success = false;
            break;
        }
    }
    if (success) {
        std::cout << "Verification Successful!" << std::endl;
    }

    // --- Cleanup ---
    checkCudaErrors(cudaStreamDestroy(stream));
    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaFreeHost(h_input));
    checkCudaErrors(cudaFreeHost(h_output));

    return 0;
}