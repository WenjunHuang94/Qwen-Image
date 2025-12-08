import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from io import BytesIO
import requests

# ==============================================================================
# 修改 1: 配置 V100 的显存分配策略 (假设你是 32GB 显存版本的 V100)
# ==============================================================================
# GPU 0: 预留更多空间给中间计算（因为要处理两张输入图），只放 20GB 权重
# GPU 1-7: 放满 30GB 权重
max_memory_config = {
    0: "20GB",
    1: "30GB", 2: "30GB", 3: "30GB", 4: "30GB",
    5: "30GB", 6: "30GB", 7: "30GB",
}

print("Loading Qwen-Image-Edit-Plus pipeline...")

# ==============================================================================
# 修改 2: 加载模型时的关键参数
# ==============================================================================
# 如果你已经下载到本地，把 "Qwen/Qwen-Image-Edit-2509" 换成你的本地绝对路径
model_path = "Qwen/Qwen-Image-Edit-2509"

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,    # <---【关键】V100 必须用 float16，不能用 bfloat16
    device_map="balanced",        # <---【关键】自动把模型切分到 8 张卡上
    max_memory=max_memory_config, # <---【关键】防止 0 号卡爆显存
    trust_remote_code=True
)

# 删除这就话！pipeline.to('cuda')  <--- 因为 device_map 已经自动处理了设备分配

print("pipeline loaded")
pipeline.set_progress_bar_config(disable=None)

# 准备输入图片
print("Downloading images...")
image1 = Image.open(BytesIO(requests.get("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/edit2509_1.jpg").content))
image2 = Image.open(BytesIO(requests.get("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/edit2509_2.jpg").content))

prompt = "The magician bear is on the left, the alchemist bear is on the right, facing each other in the central park square."

# ==============================================================================
# 修改 3: Generator 设置
# ==============================================================================
inputs = {
    "image": [image1, image2],
    "prompt": prompt,
    # 【关键】多卡环境下，Generator 最好放在 CPU 上，避免设备冲突报错
    "generator": torch.Generator(device="cpu").manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}

print("Starting inference on 8x V100...")

# 强制同步一下，确保之前的显存操作完成
if torch.cuda.is_available():
    torch.cuda.synchronize()

with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_image_edit_plus.png")
    print("image saved at", os.path.abspath("output_image_edit_plus.png"))