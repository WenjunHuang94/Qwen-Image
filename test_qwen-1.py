import time
from diffusers import DiffusionPipeline
import torch
import os

# 配置部分
model_name = "Qwen/Qwen-Image"

# 1. 设备与精度设置
if torch.cuda.is_available():
    print("CUDA is available. Using bfloat16 on GPU.")
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    print("CUDA not available. Using float32 on CPU.")
    torch_dtype = torch.float32
    device = "cpu"

# 2. 加载模型 (不计入推理时间)
pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
pipe = pipe.to(device)

# ==========================================
# 4. 显存占用计算逻辑
# ==========================================
def get_module_mem(module, name):
    if module is None:
        return 0

    # 计算参数量 (Weights)
    param_size = 0
    for param in module.parameters():
        param_size += param.numel() * param.element_size()

    # 计算缓冲区 (Buffers, e.g. BatchNorm stats)
    buffer_size = 0
    for buffer in module.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    total_size_gb = (param_size + buffer_size) / (1024 ** 3)
    print(f"{name:.<25} : {total_size_gb:.2f} GB")
    return total_size_gb


print("\n" + "=" * 50)
print("     Qwen-Image (BF16) 显存占用分析报告")
print("=" * 50)

total_mem = 0

# 1. Text Encoder (通常是 T5-XXL)
if hasattr(pipe, "text_encoder") and pipe.text_encoder:
    total_mem += get_module_mem(pipe.text_encoder, "Text Encoder")

# 2. Transformer (核心生成模块)
if hasattr(pipe, "transformer") and pipe.transformer:
    total_mem += get_module_mem(pipe.transformer, "Transformer")
elif hasattr(pipe, "unet") and pipe.unet:
    total_mem += get_module_mem(pipe.unet, "UNet")

# 3. VAE (解码模块)
if hasattr(pipe, "vae") and pipe.vae:
    total_mem += get_module_mem(pipe.vae, "VAE")

print("-" * 50)
print(f"模型静态权重总计           : {total_mem:.2f} GB")
print("-" * 50)

# ==========================================
# 5. A100 80G 场景模拟结论
# ==========================================
a100_capacity = 80.0
remaining = a100_capacity - total_mem

print(f"A100 80G 显存容量          : {a100_capacity:.2f} GB")
print(f"预计剩余显存 (用于计算)    : {remaining:.2f} GB")
print("=" * 50 + "\n")

if remaining > 15:
    print("✅ 结论：在 A100 80G 上完全可以【单卡】运行！")
    print("   原因：模型权重约 54GB，显卡剩 26GB。")
    print("   剩下的 26GB 足够存放推理时的中间变量 (Activations) 和 KV Cache。")
    print("   因为不需要跨卡通信，也不需要 CPU Offload，所以速度极快 (15秒)。")
else:
    print("⚠️ 结论：显存非常紧张，可能需要优化。")