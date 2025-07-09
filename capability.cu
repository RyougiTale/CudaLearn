#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(error_id) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cout << "未找到可用的CUDA设备 (NVIDIA GPU)。" << std::endl;
    } else {
        std::cout << "找到 " << deviceCount << " 个可用的GPU。" << std::endl;
        for (int dev = 0; dev < deviceCount; ++dev) {
            cudaSetDevice(dev);
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, dev);

            std::cout << "\n--- GPU " << dev << " ---" << std::endl;
            std::cout << "设备名称: " << deviceProp.name << std::endl;
            std::cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        }
    }

    return 0;
}