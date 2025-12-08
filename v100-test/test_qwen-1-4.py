import os
import torch
from diffusers import DiffusionPipeline

# 路径配置
local_model_path = "/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image/"

print("正在分析模型组件大小 (FP16)...")

try:
    # 我们加载到 CPU (meta device) 仅仅为了读取结构，不会爆显存
    # low_cpu_mem_usage=True 是关键，它不会把完整权重读入 RAM
    pipe = DiffusionPipeline.from_pretrained(
        local_model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
        use_safetensors=True
    )
except Exception as e:
    print(f"加载失败: {e}")
    exit()


def get_module_size(module, name):
    param_size = 0
    for param in module.parameters():
        param_size += param.numel() * param.element_size()

    gb_size = param_size / (1024 ** 3)
    print(f"{name:.<20} : {gb_size:.2f} GB")
    return gb_size


print("\n" + "=" * 40)
print("     Qwen-Image 显存占用分析表")
print("=" * 40)

total_size = 0

if hasattr(pipe, "text_encoder") and pipe.text_encoder:
    total_size += get_module_size(pipe.text_encoder, "Text Encoder")

if hasattr(pipe, "transformer") and pipe.transformer:
    # 這是核心
    total_size += get_module_size(pipe.transformer, "Transformer")
elif hasattr(pipe, "unet") and pipe.unet:
    total_size += get_module_size(pipe.unet, "UNet")

if hasattr(pipe, "vae") and pipe.vae:
    total_size += get_module_size(pipe.vae, "VAE")

print("-" * 40)
print(f"模型权重总计 (FP16)  : {total_size:.2f} GB")
print("注意：推理时还需要额外 2GB~5GB 用于中间计算(Activation)")
print("=" * 40 + "\n")