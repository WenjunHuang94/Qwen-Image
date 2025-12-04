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

# 3. 准备参数
positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.",
    "zh": ", 超清，4K，电影级构图."
}

# 生成图像
# prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197".'''
prompt = '''ohwhwj man, wearing a red helmet and orange life jacket, smiling with water droplets on his face.'''
negative_prompt = " "
width, height = (512, 512)

print("Starting image generation... (This may take a moment)")

# ==========================================
# 核心修改：精确计算推理时间 (Inference Time)
# ==========================================

# A. 强制等待 GPU 完成之前的初始化任务
if torch.cuda.is_available():
    torch.cuda.synchronize()

# B. 记录开始时间
start_time = time.time()

# C. 执行推理
image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]

# D. 强制等待 GPU 推理任务完全结束
if torch.cuda.is_available():
    torch.cuda.synchronize()

# E. 记录结束时间
end_time = time.time()

# ==========================================

# 计算耗时
inference_time = end_time - start_time
print(f"纯推理耗时: {inference_time:.4f} 秒")

# 如果想看每秒生成多少步 (Steps per Second)
print(f"速度: {50 / inference_time:.2f} it/s")

# 4. 保存图片 (IO 操作，不计入推理时间)
save_path = "example2.png"
image.save(save_path)
print(f"Image saved as {save_path}")