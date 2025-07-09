import torch

import torch

# 打印PyTorch版本信息
print(f"PyTorch Version: {torch.__version__}")

# 打印PyTorch是为哪个CUDA版本编译的
print(f"PyTorch built with CUDA version: {torch.version.cuda}")

# 再次检查CUDA是否可用
print(f"Is CUDA available? {torch.cuda.is_available()}")

# 检查CUDA是否可用
if torch.cuda.is_available():
    # 获取当前使用的GPU设备数量
    num_gpus = torch.cuda.device_count()
    print(f"找到 {num_gpus} 个可用的GPU。")

    # 遍历所有可用的GPU
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        # 获取计算能力 (返回一个元组, 例如 (8, 6))
        capability = torch.cuda.get_device_capability(i)
        
        print(f"GPU {i}: {gpu_name}")
        print(f"  Compute Capability: {capability[0]}.{capability[1]}")
else:
    print("未找到可用的CUDA设备 (NVIDIA GPU)。")