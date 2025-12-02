import os
from PIL import Image
import torch

from diffusers import QwenImageEditPipeline

# 确保你已经运行了 pip install accelerate
print("Loading Qwen-Image-Edit pipeline...")

pipeline = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    torch_dtype=torch.bfloat16,
    device_map="cuda"  # <--- 这是唯一的改动！
)
print("pipeline loaded")

# 和上次一样，用了 device_map 之后，就不需要再 .to("cuda") 了
pipeline.set_progress_bar_config(disable=None)

# 确保 ./input.png 文件存在
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

print("Starting image editing...")
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_image_edit2.png")
    print("image saved at", os.path.abspath("output_image_edit.png"))
