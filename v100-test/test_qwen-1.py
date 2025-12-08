import os
import time
import torch
from diffusers import DiffusionPipeline

# ==========================================
# 1. 环境与路径配置
# ==========================================

# 强制使用前4张卡 (根据你的V100环境)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# 本地模型路径
local_model_path = "/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image/"

if not os.path.exists(local_model_path):
    raise FileNotFoundError(f"找不到模型路径: {local_model_path}")

# ==========================================
# 2. 显存分配策略 (仿照你的 Edit 脚本)
# ==========================================
# V100 32GB x 4
max_memory_config = {
    0: "24GB", # 预留显存给系统开销
    1: "30GB",
    2: "30GB",
    3: "30GB",
}

print(f"Loading Qwen-Image pipeline from {local_model_path}...")

# ==========================================
# 3. 加载模型 (核心修改：多卡 + BF16)
# ==========================================
try:
    # 按照你的要求：在 V100 上强行使用 bfloat16
    dtype_config = torch.bfloat16
    print(">>> ⚠️ 注意: 当前正在 V100 上强制使用 bfloat16，预计会报错！")

    pipe = DiffusionPipeline.from_pretrained(
        local_model_path,
        torch_dtype=dtype_config,        # 你的要求：bf16
        device_map="balanced",           # 自动分配到多卡
        max_memory=max_memory_config,    # 限制每张卡的显存
        trust_remote_code=True,
        local_files_only=True
    )
    print("Pipeline loaded successfully (Model weights loaded, but computation might fail later).")

except Exception as e:
    print(f"模型加载阶段报错: {e}")
    exit()

# 4. 准备参数
positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.",
    "zh": ", 超清，4K，电影级构图."
}

prompt = '''ohwhwj man, wearing a red helmet and orange life jacket, smiling with water droplets on his face.'''
negative_prompt = " "
width, height = (512, 512) # 为了测试速度，先用 512，Qwen-Image通常支持更大

# ==========================================
# 5. 推理与计时
# ==========================================
print("Starting image generation...")

try:
    # A. 强制等待 GPU 初始化
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # B. 记录开始时间
    start_time = time.time()

    # C. 执行推理 (Generator 使用 CPU 以避免多卡随机数问题)
    # 注意：Qwen-Image 的参数可能需要 true_cfg_scale，视具体 pipeline 实现而定
    image = pipe(
        prompt=prompt + positive_magic["en"],
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=torch.Generator(device="cpu").manual_seed(42)
    ).images[0]

    # D. 强制等待 GPU 结束
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # E. 记录结束时间
    end_time = time.time()

    # 计算耗时
    inference_time = end_time - start_time
    print(f"纯推理耗时: {inference_time:.4f} 秒")
    print(f"速度: {50 / inference_time:.2f} it/s")

    # 6. 保存图片
    save_path = "qwen_t2i_v100_bf16_test.png"
    image.save(save_path)
    print(f"Image saved as {save_path}")

except RuntimeError as e:
    print("\n" + "="*50)
    print("!!! 捕获到运行时错误 (预期内) !!!")
    if "BFloat16" in str(e) or "cudnn" in str(e):
        print("原因: V100 架构不支持 bfloat16 指令集。")
        print("建议: 请将 torch_dtype 改为 torch.float16。")
    print(f"详细报错信息: {e}")
    print("="*50 + "\n")
except Exception as e:
    print(f"发生其他错误: {e}")