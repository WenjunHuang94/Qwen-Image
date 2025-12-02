import time
from diffusers import DiffusionPipeline
import torch

# 记录开始时间
start_time = time.time()

model_name = "Qwen/Qwen-Image"
# 1. 在这里写死你的缓存路径
my_cache_path = "/home/disk2/hwj/my_hf_cache"

# 加载 pipeline
if torch.cuda.is_available():
    print("CUDA is available. Using bfloat16 on GPU.")
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    print("CUDA not available. Using float32 on CPU.")
    torch_dtype = torch.float32
    device = "cpu"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
pipe = pipe.to(device)

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": ", 超清，4K，电影级构图." # for chinese prompt
}

# 生成图像
# prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197".'''
prompt = '''ohwhwj man, wearing a red helmet and orange life jacket, smiling with water droplets on his face.'''


negative_prompt = " " # Recommended if you don't use a negative prompt.

# 选择一个宽高比
# width, height = (1024, 1024) # 16:9
width, height = (512, 512) # 16:9

print("Starting image generation... (This may take a moment)")

image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]

image.save("example2.png")

# 记录结束时间并计算总耗时
end_time = time.time()
total_time = end_time - start_time

print(f"Image saved as example.png")
print(f"总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")