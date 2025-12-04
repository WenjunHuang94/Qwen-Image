import time
from diffusers import DiffusionPipeline
import torch
import os

# 记录开始时间
start_time = time.time()

# ------------------------------------------------------------------
# 修改点 1: 将 model_name 指向你手动下载好的文件夹绝对路径
# 假设你通过 git clone 下载到了 /home/disk2/hwj/my_hf_cache/Qwen-Image
# 文件夹内必须包含 model_index.json
# ------------------------------------------------------------------
local_model_path = "/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image/my_hf_cache/Qwen-Image/"

# 检查路径是否存在，避免报错
if not os.path.exists(local_model_path):
    raise FileNotFoundError(f"找不到模型路径: {local_model_path}。请确保你已经手动下载了模型文件夹。")

# 加载 pipeline
if torch.cuda.is_available():
    print("CUDA is available. Using bfloat16 on GPU.")
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    print("CUDA not available. Using float32 on CPU.")
    torch_dtype = torch.float32
    device = "cpu"

print(f"Loading model from local path: {local_model_path}")

# ------------------------------------------------------------------
# 修改点 2: 传入本地路径，并添加 local_files_only=True
# ------------------------------------------------------------------

# 注意：这里不需要再手动判断 cuda/cpu 了，accelerate 会自动处理
pipe = DiffusionPipeline.from_pretrained(
    local_model_path,
    torch_dtype=torch.bfloat16, # 建议保持 bf16
    local_files_only=True,
    use_safetensors=True,
    trust_remote_code=True,
    device_map="balanced"       # <--- 【核心修改】自动将模型分布到可见的 GPU 上
)

# ------------------------------------------------------------------
# 【非常重要】删除下面这行！！
# pipe = pipe.to(device)  <--- 删除这行，不要手动移动模型
# ------------------------------------------------------------------

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.",
    "zh": ", 超清，4K，电影级构图."
}

# 生成图像
prompt = '''ohwhwj man, wearing a red helmet and orange life jacket, smiling with water droplets on his face.'''

negative_prompt = " "

# 选择一个宽高比
width, height = (512, 512)

print("Starting image generation... (This may take a moment)")

image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0, # 注意：Qwen-Image 也可以使用 guidance_scale，如果不报错保留 true_cfg_scale 即可
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]

image.save("example2.png")

# 记录结束时间并计算总耗时
end_time = time.time()
total_time = end_time - start_time

print(f"Image saved as example2.png")
print(f"总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
