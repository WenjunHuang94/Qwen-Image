import os
import torch
from PIL import Image
from diffusers import QwenImageEditPipeline

# 1. 强制使用前4张卡 (V100 32G x 4 足够)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# 2. 你的本地模型路径 (根据你的报错信息修改的正确路径)
local_model_path = "/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image/my_hf_cache/Qwen-Image-Edit"

if not os.path.exists(local_model_path):
    raise FileNotFoundError(f"找不到模型路径: {local_model_path}")

print(f"Loading Qwen-Image-Edit pipeline from: {local_model_path}")

# 3. 核心修复配置
# 显存分配策略：
# GPU 0: 只给 20GB (放 Text Encoder + VAE)，防止被撑爆
# GPU 1-3: 给满 30GB (放 Transformer 权重)
max_memory_config = {
    0: "20GB",
    1: "30GB",
    2: "30GB",
    3: "30GB",
}

print("正在加载模型并分配显卡 (这可能需要几秒钟)...")

pipeline = QwenImageEditPipeline.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16,   # V100 必须用 float16
    device_map="balanced",           # 开启自动分配
    max_memory=max_memory_config,# <--- 【关键】强制限制显存，防止单卡 OOM
    trust_remote_code=True,
    local_files_only=True
)
print("Pipeline loaded successfully!")

pipeline.set_progress_bar_config(disable=None)

# 4. 准备测试图片
image_path = "./001.jpg"
if not os.path.exists(image_path):
    print(f"警告: 未找到 {image_path}，将生成一张纯色图用于测试。")
    # 创建一张 1024x1024 的测试图
    image = Image.new('RGB', (1024, 1024), color='blue')
else:
    image = Image.open(image_path).convert("RGB")

prompt = "Change the color of the person's clothing to purple with a flash as the background."

inputs = {
    "image": image,
    "prompt": prompt,
    # generator 放在 CPU 上以保证多卡环境下的随机性一致
    "generator": torch.Generator(device="cpu").manual_seed(42),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 30, # 测试时可以先设少一点，比如 30 步
}

print("Starting image editing...")
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_image_edit3.png")
    print("Success! Image saved at", os.path.abspath("output_image_edit3.png"))
