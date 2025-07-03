#include <iostream>
#include <vector>

// Define a small stencil kernel (weights)
// This data will be stored in Constant Memory
__constant__ float const_weights[3];

// NOTE: The old global texture reference is removed.
// texture<float, 1, cudaReadModeElementType> tex_input; // DELETED

/**
 * @brief Kernel demonstrating use of different memory types.
 *
 * @param g_output Pointer to the output array in Global Memory.
 * @param tex_obj The CUDA Texture Object for reading input.
 * @param N The total number of elements.
 */
__global__ void stencil_kernel(float* g_output, cudaTextureObject_t tex_obj, int N) {
    // 1. Declare Shared Memory
    const int BLOCK_SIZE = 256;
    __shared__ float s_data[BLOCK_SIZE + 2];

    // 2. Use Local Memory
    int i_global = blockIdx.x * blockDim.x + threadIdx.x;
    int i_shared = threadIdx.x + 1;

    // 3. Load from Global Memory (via Texture Object) into Shared Memory
    // Note the new tex1Dfetch syntax
    if (threadIdx.x == 0) {
        s_data[0] = tex1Dfetch<float>(tex_obj, i_global - 1); // Left ghost cell
    }
    s_data[i_shared] = tex1Dfetch<float>(tex_obj, i_global); // This thread's data
    if (threadIdx.x == blockDim.x - 1) {
        s_data[BLOCK_SIZE + 1] = tex1Dfetch<float>(tex_obj, i_global + 1); // Right ghost cell
    }

    __syncthreads();

    // 4. Perform computation
    if (i_global < N) {
        float left   = s_data[i_shared - 1];
        float middle = s_data[i_shared];
        float right  = s_data[i_shared + 1];

        float result = left   * const_weights[0] +
                       middle * const_weights[1] +
                       right  * const_weights[2];

        // 5. Write final result to Global Memory
        g_output[i_global] = result;
    }
}

int main() {
    // --- Setup ---
    const int N = 1024 * 1024;
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t bytes = N * sizeof(float);

    // --- Host Data ---
    std::vector<float> h_input(N);
    std::vector<float> h_output(N);
    float h_weights[3] = {0.25f, 0.5f, 0.25f};

    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    // --- Device Data (Global Memory) ---
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);

    // --- Constant Memory Setup ---
    cudaMemcpyToSymbol(const_weights, h_weights, sizeof(h_weights));

    // --- NEW: Texture Object Setup ---
    // 1. Define resource description (points to the device memory)
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_input;
    resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
    resDesc.res.linear.sizeInBytes = bytes;

    // 2. Define texture description (how to read the texture)
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.addressMode[0] = cudaAddressModeClamp; // Clamp out-of-bounds access

    // 3. Create the texture object
    cudaTextureObject_t tex_obj = 0;
    cudaCreateTextureObject(&tex_obj, &resDesc, &texDesc, NULL);


    // --- Kernel Launch ---
    std::cout << "Launching stencil kernel..." << std::endl;
    // Pass the texture object as a kernel argument
    stencil_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_output, tex_obj, N);
    cudaDeviceSynchronize();
    std::cout << "Kernel finished." << std::endl;

    // --- Copy result back and verify ---
    cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

    std::cout << "Verifying a few results..." << std::endl;
    for (int i = 10; i < 15; ++i) {
        float expected = h_input[i-1] * h_weights[0] + h_input[i] * h_weights[1] + h_input[i+1] * h_weights[2];
        std::cout << "GPU result at index " << i << ": " << h_output[i]
                  << ", CPU expected: " << expected << std::endl;
    }

    // --- Cleanup ---
    cudaDestroyTextureObject(tex_obj); // Destroy the texture object
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}


// 1. 全局内存 (Global Memory) - 主要仓库 🌍
// 角色：d_input 和 d_output 数组。
// 它是所有数据的起点和终点。我们的原始数据和最终结果都存放在这里。它的特点是容量巨大但速度最慢。所有与CPU的数据交换都必须通过它。在我们的“舞台剧”中，它是存放所有原材料和最终成品的巨大仓库。

// 2. 常量内存 (Constant Memory) - 不变的蓝图 📜
// 角色：const_weights 数组。
// 它用来存储那些在整个计算过程中固定不变、且所有线程都需要读取的数据。在我们的例子中，就是那三个权重值 {0.25, 0.5, 0.25}。
// 为什么用它？ 常量内存有专门的缓存，并且可以高效地将同一个值**广播(Broadcast)**给一个Warp中的所有线程。当所有线程都需要读取同一个“常数”时，使用常量内存比让每个线程都去全局内存读取要快得多。它就像一张下发给所有工人的、永不改变的“施工蓝图”。

// 3. 纹理内存 (Texture Memory) - 聪明的图书管理员 🖼️
// 角色：通过 tex_obj 访问 d_input。
// 它不是一种新的内存，而是一种特殊的、带缓存的访问全局内存的方式。在这个Demo中，它展现了两个核心优势：

// 硬件级边界处理：我们设置了 cudaAddressModeClamp。这意味着当一个线程（比如在数组边缘的0号线程）试图读取它左边的邻居（索引为-1）时，程序不会崩溃。纹理硬件会自动“钳位”，并安全地返回边界值（即索引为0的值）。这极大地简化了内核代码，我们无需在代码里写 if (i > 0) 这样的边界检查。
// 空间局部性缓存：纹理缓存对2D/3D等空间上邻近的数据访问有优化。
// 它就像一个聪明的图书管理员，你问他要一本不存在的书（越界访问），他不会报错，而是会给你一本最相关的书（边界值）。

// 4. 共享内存 (Shared Memory) - 高速本地工作台 ⚡
// 角色：s_data 数组。
// 这是整个优化的核心。它的速度极快，但容量小，且只对一个块内的线程可见。

// 优化策略：
// 昂贵的操作只做一次：每个线程从慢速的全局内存（通过纹理）中读取一个数据，并把它存入飞快的共享内存工作台。
// 协同工作：通过 __syncthreads() 确保工作台上的材料都准备好了。
// 重复利用：当一个线程需要计算 Output[i] 时，它需要 Input[i-1], Input[i], Input[i+1]。它从工作台（共享内存）上拿来这三个值。它的邻居线程也做同样的事。注意，Input[i] 这个值会被它的左邻居和右邻居重复使用。通过共享内存，这个值只需从全局内存读一次，就能被多个线程重复使用。
// 这个策略将原本需要 3 * N 次的全局内存读取，锐减到了大约 N 次，极大地减少了对慢速内存的访问，从而实现了巨大的性能提升。

// 5. 局部内存/寄存器 (Local Memory) - 私人草稿纸 ✍️
// 角色：i_global, result, left 等变量。
// 这是每个线程私有的内存，速度最快（寄存器）。它用来存放线程在计算过程中的临时变量。它就像每个工人自己手中的笔和草稿纸，用于最终的计算。

