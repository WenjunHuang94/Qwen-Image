import os
import torch
import numpy as np
from diffusers import DiffusionPipeline

# 1. 显卡配置
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# 2. 路径
local_model_path = "/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image/"

# 显存分配
max_memory_config = {
    0: "20GB",
    1: "30GB",
    2: "30GB",
    3: "30GB",
}

print("🛠️ 正在加载模型进行故障诊断 (FP16)...")

# 加载 Pipeline
pipe = DiffusionPipeline.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16,  # 保持 FP16 以复现问题
    device_map="balanced",
    max_memory=max_memory_config,
    use_safetensors=True,
    local_files_only=True,
    trust_remote_code=True
)


# 辅助函数：检查 Tensor 是否正常
def check_tensor(name, tensor):
    if tensor is None:
        return
    # 转为 float32 计算统计量，防止统计时溢出
    t_float = tensor.float()
    max_val = t_float.max().item()
    min_val = t_float.min().item()
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    status = "✅ 正常"
    if has_nan or has_inf:
        status = "❌ 溢出 (NaN/Inf)"
    elif abs(max_val) > 60000:  # 接近 FP16 上限
        status = "⚠️ 危险 (接近 FP16 上限)"

    print(f"[{name}] 状态: {status} | Max: {max_val:.2f} | Min: {min_val:.2f}")
    return has_nan or has_inf


# ==============================================================================
# 第一步：检查 Text Encoder (文本编码器)
# ==============================================================================
print("\n🔍 步骤 1: 检查 Text Encoder 输出...")
prompt = "A cute cat"
try:
    # 手动调用 encode_prompt
    prompt_embeds, prompt_masks = pipe.encode_prompt(
        prompt=prompt,
        device=pipe.device,
        num_images_per_prompt=1,
        max_sequence_length=512
    )
    is_text_broken = check_tensor("Text Embeddings", prompt_embeds)

    if is_text_broken:
        print("\n🚨 诊断结论：【Text Encoder】在 FP16 下溢出！")
        print("💡 解决方案：必须将 Text Encoder 转为 FP32。")
        print("   代码：pipe.text_encoder.to(dtype=torch.float32)")
        # 如果这里就挂了，为了测试后面，我们强行修复一下继续跑
        # pipe.text_encoder.to(dtype=torch.float32)
        # (但在诊断脚本里我们先暂停，让你看清楚结果)
        exit()
    else:
        print("✅ Text Encoder 看起来没问题，继续检查下一步...")

except Exception as e:
    print(f"Text Encoder 运行出错: {e}")

# ==============================================================================
# 第二步：检查 DiT (Transformer) 推理过程
# ==============================================================================
print("\n🔍 步骤 2: 检查 DiT 逐步去噪过程...")


# 定义一个回调函数，监控每一步的 Latents
def callback_monitor(pipe, step, timestep, callback_kwargs):
    latents = callback_kwargs.get("latents")
    # 只检查第一步和中间几步，避免刷屏
    if step % 5 == 0:
        is_broken = check_tensor(f"DiT Step {step}", latents)
        if is_broken:
            print(f"\n🚨 诊断结论：【DiT (Transformer)】在第 {step} 步溢出！")
            print("💡 原因：FP16 范围不够，或者输入数据已经是 NaN。")
            print("💡 解决方案：如果 Text Encoder 没问题，那说明 DiT 也必须转 FP32 (但这在 V100 上可能显存不够)。")
            raise ValueError("DiT Output is NaN, stopping generation.")
    return callback_kwargs


# 准备数据
width, height = 512, 512
generator = torch.Generator(device="cpu").manual_seed(42)

try:
    # 强制 VAE 用 FP32 (我们已知 VAE 肯定有问题，先排除它，专心测 DiT)
    pipe.vae.to(dtype=torch.float32)

    print("   (注：已临时将 VAE 设为 FP32 以排除干扰，专心监测 DiT)")

    image = pipe(
        prompt=prompt,
        negative_prompt=" ",
        width=width,
        height=height,
        num_inference_steps=10,  # 只跑10步快速测试
        true_cfg_scale=4.0,
        generator=generator,
        callback_on_step_end=callback_monitor  # <--- 插入探针
    ).images[0]

    print("\n✅ 诊断结论：DiT (Transformer) 在 FP16 下运行正常！")
    print("🎉 如果最终图片还是黑的，那是 VAE 的问题 (但我们已经修了 VAE)。")

except ValueError as e:
    print("推理被中断。")
except Exception as e:
    print(f"推理出错: {e}")