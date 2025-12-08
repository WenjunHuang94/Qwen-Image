import os
import time
import torch
from PIL import Image
from diffusers import QwenImageEditPipeline

# 1. 强制使用前4张卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

local_model_path = "/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image-Edit"

if not os.path.exists(local_model_path):
    raise FileNotFoundError(f"找不到模型路径: {local_model_path}")

print(f"Loading Qwen-Image-Edit pipeline...")

# 2. 显存分配策略
# V100 32GB x 4 应该足够。
# 这里的 key 0,1,2,3 对应的是 CUDA_VISIBLE_DEVICES 里的相对顺序
max_memory_config = {
    0: "24GB", # 稍微多给一点给 0 卡，V100 32G 还是比较充裕的
    1: "30GB",
    2: "30GB",
    3: "30GB",
}

# 3. 加载 Pipeline
pipeline = QwenImageEditPipeline.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16,   # TODO V100 关键设置：必须 float16,不能用A100上的bfloat16
    device_map="balanced",       # 开启自动分配
    max_memory=max_memory_config,# 强制限制显存
    trust_remote_code=True,
    local_files_only=True
)
print("Pipeline loaded successfully!")

pipeline.set_progress_bar_config(disable=None)

# 4. 准备测试图片
image_path = "./001.jpg"
if not os.path.exists(image_path):
    print(f"警告: 未找到 {image_path}，生成测试图。")
    image = Image.new('RGB', (1024, 1024), color='blue')
else:
    image = Image.open(image_path).convert("RGB")

prompt = "Change the color of the person's clothing to purple with a flash as the background."

inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.Generator(device="cpu").manual_seed(42), # Generator 用 CPU
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 30,
}

# ==========================================
# 5. 计时逻辑
# ==========================================
print("Starting image editing...")
if torch.cuda.is_available():
    torch.cuda.synchronize()

start_time = time.time()

with torch.inference_mode():
    output = pipeline(**inputs)

if torch.cuda.is_available():
    torch.cuda.synchronize()
end_time = time.time()

inference_time = end_time - start_time
print(f"纯推理耗时: {inference_time:.4f} 秒")
print(f"迭代速度: {inputs['num_inference_steps'] / inference_time:.2f} it/s")

output.images[0].save("output_image_edit3_v100.png")
print("Success!")