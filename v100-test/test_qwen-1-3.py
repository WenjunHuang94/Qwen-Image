import os
import time
import torch
from diffusers import DiffusionPipeline

# ==========================================
# 1. 使用 4 张卡 (32GB x 4 = 128GB)
# ==========================================
# 既然你有8张卡，咱们就大大方方用4张，确保显存绝对溢出
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

local_model_path = "/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image/"
print(f"Loading Qwen-Image pipeline from {local_model_path}...")

# ==========================================
# 2. 核心设置：4卡显存分配
# ==========================================
# 明确告诉它：这4张卡每张都有 30GB 可用！
# 这样它就绝对不会把模型层卸载到 CPU 上了。
max_memory_config = {
    0: "30GB",
    1: "30GB",
    2: "30GB",
    3: "30GB"
}

try:
    pipe = DiffusionPipeline.from_pretrained(
        local_model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
        use_safetensors=True,
        # 修正：使用 balanced (这是 Qwen Pipeline 支持的)
        device_map="balanced",
        max_memory=max_memory_config
    )

    print("Pipeline loaded successfully across 4 GPUs!")
    print(f"Device Map 分配情况: {pipe.hf_device_map}")  # 打印看看，确认没有 'cpu'

except Exception as e:
    print(f"模型加载失败: {e}")
    exit()

# 3. 准备参数
positive_magic = ", Ultra HD, 4K, cinematic composition."
prompt = "ohwhwj man, wearing a red helmet and orange life jacket, smiling with water droplets on his face."
negative_prompt = " "
width, height = (512, 512)

print("Starting image generation...")

# ==========================================
# 4. 推理与计时
# ==========================================
if torch.cuda.is_available():
    torch.cuda.synchronize()

start_time = time.time()

# 多卡环境 Generator 必须放 CPU
with torch.inference_mode():
    image = pipe(
        prompt=prompt + positive_magic,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=torch.Generator(device="cpu").manual_seed(42)
    ).images[0]

if torch.cuda.is_available():
    torch.cuda.synchronize()

end_time = time.time()
inference_time = end_time - start_time

print(f"纯推理耗时: {inference_time:.4f} 秒")
print(f"速度: {50 / inference_time:.2f} it/s")

image.save("qwen_t2i_4gpu_balanced.png")
print("Done.")