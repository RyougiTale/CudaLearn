// cl main.cpp /I"%CUDA_PATH%\include" /link /LIBPATH:"%CUDA_PATH%\lib\x64" cuda.lib
// cl main.cpp /EHsc /I"%CUDA_PATH%\include" /link /LIBPATH:"%CUDA_PATH%\lib\x64" cuda.lib
#include <iostream>
#include <vector>
#include "cuda.h" // 注意！这里是 cuda.h 而不是 cuda_runtime.h

// 一个简单的宏，用于检查Driver API调用的返回值
#define CHECK_CUDA_DRIVER(err) \
    if (err != CUDA_SUCCESS) { \
        const char* err_name; \
        cuGetErrorName(err, &err_name); \
        std::cerr << "CUDA Driver API Error: " << err_name << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

int main() {
    // ---- 1. 初始化和上下文管理 ----
    // 这是Driver API的第一步，Runtime API中是隐式完成的
    cuInit(0);
    CUdevice cuDevice;
    CHECK_CUDA_DRIVER(cuDeviceGet(&cuDevice, 0)); // 获取第一个GPU设备
    CUcontext cuContext;
    CHECK_CUDA_DRIVER(cuCtxCreate(&cuContext, 0, cuDevice)); // 为设备创建上下文

    std::cout << "1. CUDA Driver API 初始化成功，上下文已创建。" << std::endl;

    // ---- 2. 加载模块和核函数 ----
    // Runtime API在编译时就链接好了核函数，Driver API需要手动加载
    CUmodule cuModule;
    CHECK_CUDA_DRIVER(cuModuleLoad(&cuModule, "vecAdd.ptx")); // 加载编译好的PTX文件
    CUfunction vecAddFunc;
    CHECK_CUDA_DRIVER(cuModuleGetFunction(&vecAddFunc, cuModule, "vecAdd_kernel")); // 从模块中获取核函数句柄

    std::cout << "2. PTX 模块已加载，核函数句柄已获取。" << std::endl;

    // ---- 3. 内存分配与数据传输 ----
    const int N = 1024;
    const size_t bytes = N * sizeof(float);

    std::vector<float> h_A(N), h_B(N), h_C(N);
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    CUdeviceptr d_A, d_B, d_C;
    CHECK_CUDA_DRIVER(cuMemAlloc(&d_A, bytes)); // 注意函数名是 cuMemAlloc
    CHECK_CUDA_DRIVER(cuMemAlloc(&d_B, bytes));
    CHECK_CUDA_DRIVER(cuMemAlloc(&d_C, bytes));

    CHECK_CUDA_DRIVER(cuMemcpyHtoD(d_A, h_A.data(), bytes)); // Host to Device
    CHECK_CUDA_DRIVER(cuMemcpyHtoD(d_B, h_B.data(), bytes));

    std::cout << "3. 设备内存已分配，数据已从主机拷贝到设备。" << std::endl;

    // ---- 4. 设置参数并启动核函数 ----
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Driver API 启动核函数需要将参数打包成一个指针数组
    void* kernel_args[] = { &d_C, &d_A, &d_B, (void*)&N };

    CHECK_CUDA_DRIVER(
        cuLaunchKernel(
            vecAddFunc,      // 要启动的核函数句柄
            blocksPerGrid, 1, 1,    // Grid 维度
            threadsPerBlock, 1, 1,  // Block 维度
            0,               // 共享内存大小 (bytes)
            NULL,            // Stream ID, NULL表示默认流
            kernel_args,     // 核函数参数
            NULL             // 额外的启动选项
        )
    );
    std::cout << "4. 核函数已启动。" << std::endl;
    
    // ---- 5. 同步与结果验证 ----
    CHECK_CUDA_DRIVER(cuCtxSynchronize()); // 等待上下文(所有流)中的操作完成
    CHECK_CUDA_DRIVER(cuMemcpyDtoH(h_C.data(), d_C, bytes)); // Device to Host

    std::cout << "5. 计算完成，结果已拷贝回主机。" << std::endl;

    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (std::abs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            std::cout << "验证失败于索引 " << i << "! " << h_C[i] << " != " << h_A[i] + h_B[i] << std::endl;
            success = false;
            break;
        }
    }
    if(success) {
        std::cout << "结果验证成功！" << std::endl;
    }

    // ---- 6. 清理资源 ----
    CHECK_CUDA_DRIVER(cuMemFree(d_A));
    CHECK_CUDA_DRIVER(cuMemFree(d_B));
    CHECK_CUDA_DRIVER(cuMemFree(d_C));
    CHECK_CUDA_DRIVER(cuModuleUnload(cuModule));
    CHECK_CUDA_DRIVER(cuCtxDestroy(cuContext)); // 必须手动销毁上下文

    std::cout << "6. 所有资源已释放。" << std::endl;

    return 0;
}