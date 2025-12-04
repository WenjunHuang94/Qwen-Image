import os
import time
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from io import BytesIO
import requests

# 1. 加载模型
pipeline = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
print("pipeline loaded")
pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=None)
print("pipeline loaded")

# 2. 准备输入数据 (网络请求属于IO，不应计入推理时间)
print("Downloading images...")
image1 = Image.open(BytesIO(requests.get("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/edit2509_1.jpg").content))
image2 = Image.open(BytesIO(requests.get("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/edit2509_2.jpg").content))
print("Images downloaded.")

prompt = "The magician bear is on the left, the alchemist bear is on the right, facing each other in the central park square."

# 定义正式参数
inputs = {
    "image": [image1, image2],
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}

# ============================
# 3. 预热 (Warmup)
# ============================
print("-" * 30)
print("正在预热 (Warmup)...")
# 跑一次低步数的推理，完成显存分配和编译
with torch.inference_mode():
    _ = pipeline(
        image=[image1, image2],
        prompt=prompt,
        negative_prompt=" ",
        num_inference_steps=2 # 仅跑2步用于热身
    )
print("预热完成。")

# ============================
# 4. 正式推理与计时
# ============================
print("-" * 30)
print("开始正式推理 (Benchmarking)...")

# A. 强制等待 GPU 之前的预热任务完成
if torch.cuda.is_available():
    torch.cuda.synchronize()

# B. 记录开始时间
start_time = time.time()

# C. 执行推理
with torch.inference_mode():
    output = pipeline(**inputs)

# D. 强制等待 GPU 推理任务完全结束
if torch.cuda.is_available():
    torch.cuda.synchronize()

# E. 记录结束时间
end_time = time.time()

# 计算耗时
inference_time = end_time - start_time
print(f"纯推理耗时: {inference_time:.4f} 秒")
print(f"迭代速度: {inputs['num_inference_steps'] / inference_time:.2f} it/s")

# ============================
# 5. 保存结果
# ============================
output_image = output.images[0]
output_image.save("output_image_edit_plus.png")
print("-" * 30)
print("image saved at", os.path.abspath("output_image_edit_plus.png"))