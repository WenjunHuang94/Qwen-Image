import os
import time
from PIL import Image
import torch
from diffusers import QwenImageEditPipeline

# 1. 加载
print("Loading Qwen-Image-Edit pipeline...")
pipeline = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
pipeline.set_progress_bar_config(disable=None)

# 2. 准备输入
image = Image.open("./001.jpg").convert("RGB")
prompt = "Change the color of the person's clothing to purple with a flash as the background."
inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 50,
}

# ==========================================
# 3. 预热 (Warmup) - 关键步骤！
# ==========================================
print("正在预热 (Warmup)... (忽略这次的时间)")
with torch.inference_mode():
    # 跑一次很少步数的，或者完整步数的，让显卡热身
    _ = pipeline(image=image, prompt=prompt, num_inference_steps=2)

# ==========================================
# 4. 正式计时
# ==========================================
print("开始正式测试...")
if torch.cuda.is_available():
    torch.cuda.synchronize() # 确保显卡空闲

start_time = time.time()

with torch.inference_mode():
    output = pipeline(**inputs)

if torch.cuda.is_available():
    torch.cuda.synchronize() # 确保显卡跑完

end_time = time.time()
# ==========================================

inference_time = end_time - start_time
print(f"纯推理耗时: {inference_time:.4f} 秒")
print(f"迭代速度: {inputs['num_inference_steps'] / inference_time:.2f} it/s")

# 保存
output.images[0].save("output_image_edit3.png")
print("Image saved.")