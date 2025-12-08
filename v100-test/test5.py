import os
import torch
import numpy as np
from PIL import Image
from diffusers import QwenImageEditPipeline, QwenImagePipeline
from accelerate import dispatch_model
from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from transformers import Qwen2_5_VLForConditionalGeneration

# 1. 显卡配置 (8卡)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


class EightGPUTransformerWrapper(torch.nn.Module):
    """
    专门为 8xV100 定制的 Transformer 包装器
    将 60 层 DiT 均匀切分到 GPU 1 ~ GPU 6
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        self.config = transformer.config
        self.setup_8gpu()

    def setup_8gpu(self):
        print("⚡️ Setting up 8-GPU Pipeline Parallelism...")
        device_map = {}

        # 1. 基础层 (Input Projections) 放 GPU 1
        for name, _ in self.transformer.named_children():
            if name != "transformer_blocks":
                device_map[name] = 1  # cuda:1

        # 2. 切分 60 个 Block 到 GPU 1-6 (每张卡 10 层)
        # GPU 1: 0-9
        # GPU 2: 10-19
        # ...
        # GPU 6: 50-59
        num_blocks = 60
        for i in range(num_blocks):
            # (0~9)//10 = 0 -> +1 = 1 (GPU 1)
            target_gpu = (i // 10) + 1
            # 只有 1-6 号卡负责 DiT，防止溢出
            if target_gpu > 6: target_gpu = 6

            device_map[f"transformer_blocks.{i}"] = target_gpu

        print(f"   - 切分策略: Blocks 均匀分布在 GPU 1 到 GPU 6")

        # 应用切分
        self.transformer = dispatch_model(self.transformer, device_map=device_map)

    # 代理属性访问，防止报错
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.transformer, name)

    def forward(self, *args, **kwargs):
        # 确保输入数据在 GPU 1 (DiT 的入口)
        # accelerate 会处理后续的流转
        new_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                new_kwargs[k] = v.to("cuda:1")
            else:
                new_kwargs[k] = v

        return self.transformer(*args, **new_kwargs)


# ==============================================================================
# 主程序
# ==============================================================================
local_model_path = "/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image/"
dtype = torch.float32  # FP32 防止黑图

print("1. Loading Models (FP32)...")

# 手动加载各部分以精确控制设备
vae = AutoencoderKLQwenImage.from_pretrained(local_model_path, subfolder="vae", torch_dtype=dtype)
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(local_model_path, subfolder="text_encoder",
                                                                  torch_dtype=dtype)
transformer = QwenImageTransformer2DModel.from_pretrained(local_model_path, subfolder="transformer", torch_dtype=dtype)

# --- 关键布局 ---
# GPU 0: Text Encoder (最占显存)
# GPU 7: VAE (解码专用)
text_encoder.to("cuda:0")
vae.to("cuda:7")
vae.enable_tiling()  # 必须开启

# --- 封装 Transformer ---
# 这会自动把 DiT 切分到 GPU 1-6
wrapped_transformer = EightGPUTransformerWrapper(transformer)

print("2. Assembling Pipeline...")
pipe = QwenImagePipeline.from_pretrained(
    local_model_path,
    text_encoder=text_encoder,
    vae=vae,
    transformer=wrapped_transformer,  # 传入包装后的对象
    torch_dtype=dtype,
    local_files_only=True
)

# 这里的 .to("cuda:0") 主要是为了告诉 pipeline 主设备在哪里
# 实际上各个组件已经在上面的步骤里去到了该去的地方
# pipe.to("cuda:0")

# ==============================================================================
# 推理
# ==============================================================================
prompt = "ohwhwj man, wearing a red helmet and orange life jacket, smiling with water droplets on his face."
negative_prompt = " "
width, height = 512, 512

print(f"3. Generating ({width}x{height})...")
import time

start = time.time()

# 强制 Generator 在 CPU
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=30,
    true_cfg_scale=4.0,
    generator=torch.Generator("cpu").manual_seed(42)
).images[0]

end = time.time()
print(f"✅ Success! Time: {end - start:.2f}s")
image.save("output_v100_8gpu_fast_wrapper.png")