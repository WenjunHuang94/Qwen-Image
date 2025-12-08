import os
import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import gc

from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel, FlowMatchEulerDiscreteScheduler
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
from accelerate import dispatch_model

# ==============================================================================
# 1. 基础配置
# ==============================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
local_model_path = "/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image/"
dtype = torch.float32  # FP32 防止黑图

print("🚀 启动 8xV100 终极修复方案 (Fixed Latents Packing)...")

# ==============================================================================
# 2. 加载并切分 Text Encoder (GPU 0 & 1)
# ==============================================================================
print("\n1. Loading & Splitting Text Encoder...")
text_enc_memory_map = {0: "18GB", 1: "18GB"}

text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    local_model_path,
    subfolder="text_encoder",
    torch_dtype=dtype,
    device_map="auto",
    max_memory=text_enc_memory_map
)
text_encoder.eval()

tokenizer = Qwen2Tokenizer.from_pretrained(local_model_path, subfolder="tokenizer")

# ==============================================================================
# 3. 加载 VAE (GPU 0)
# ==============================================================================
print("2. Loading VAE (GPU 0)...")
vae = AutoencoderKLQwenImage.from_pretrained(
    local_model_path, subfolder="vae", torch_dtype=dtype
).to("cuda:0")
vae.eval()
vae.enable_tiling()

# ==============================================================================
# 4. 加载并切分 Transformer (GPU 2 - 7)
# ==============================================================================
print("3. Loading & Sharding Transformer (GPU 2-7)...")
transformer = QwenImageTransformer2DModel.from_pretrained(
    local_model_path, subfolder="transformer", torch_dtype=dtype
)
transformer.eval()

# 切分策略: 60层 -> GPU 2,3,4,5,6,7 (每卡 10 层)
device_map_dit = {}
base_gpu_idx = 2

for name, _ in transformer.named_children():
    if name != "transformer_blocks":
        device_map_dit[name] = base_gpu_idx

for i in range(60):
    offset = i // 10
    target_gpu = base_gpu_idx + offset
    if target_gpu > 7: target_gpu = 7
    device_map_dit[f"transformer_blocks.{i}"] = target_gpu

transformer = dispatch_model(transformer, device_map=device_map_dit)

gc.collect()
torch.cuda.empty_cache()
print("✅ 模型加载完毕！")

# ==============================================================================
# 5. 手动推理循环
# ==============================================================================
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(local_model_path, subfolder="scheduler")
helper_pipe = QwenImagePipeline(
    vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
    transformer=None, scheduler=scheduler
)

prompt = "ohwhwj man, wearing a red helmet and orange life jacket, smiling with water droplets on his face."
negative_prompt = " "
width, height = 512, 512
num_inference_steps = 30
guidance_scale = 4.0
seed = 42

print(f"\n🎬 Starting Inference ({width}x{height})...")
torch.cuda.synchronize()
import time

start_time = time.time()

with torch.no_grad():
    # --- Step 1: Prompt ---
    print("   [1/4] Encoding Prompt...")
    prompt_embeds, prompt_mask = helper_pipe.encode_prompt(
        prompt=prompt, device=text_encoder.device, num_images_per_prompt=1
    )
    neg_embeds, neg_mask = helper_pipe.encode_prompt(
        prompt=negative_prompt, device=text_encoder.device, num_images_per_prompt=1
    )

    # --- Step 2: Latents (修复点) ---
    print("   [2/4] Preparing Latents...")

    # 1. 在 CPU 生成随机数
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(
        (1, transformer.config.in_channels // 4, height // 16, width // 16),
        device="cpu",
        dtype=dtype,
        generator=generator
    )

    # 2. 移到 GPU 2 (DiT 入口)
    latents = latents.to("cuda:2")

    # 3. 【核心修复】参数修正：传入 latent 的高宽 (32)，而不是错误的大尺寸
    # 512 // 16 = 32
    latents = helper_pipe._pack_latents(
        latents,
        1,
        transformer.config.in_channels // 4,
        height // 16,  # <--- 修正：这里应该是 32
        width // 16  # <--- 修正：这里应该是 32
    )

    scheduler.set_timesteps(num_inference_steps, device="cuda:2")
    timesteps = scheduler.timesteps

    # Condition 移到 GPU 2
    prompt_embeds = prompt_embeds.to("cuda:2")
    prompt_mask = prompt_mask.to("cuda:2")
    neg_embeds = neg_embeds.to("cuda:2")
    neg_mask = neg_mask.to("cuda:2")

    img_shapes = [(1, height // 16, width // 16)]
    txt_seq_lens = prompt_mask.sum(dim=1).tolist()
    neg_txt_seq_lens = neg_mask.sum(dim=1).tolist()

    # --- Step 3: Denoising ---
    print("   [3/4] Denoising (GPU 2 -> 7)...")
    for i, t in enumerate(tqdm(timesteps)):
        latent_model_input = latents.to("cuda:2")
        timestep = t.expand(latents.shape[0]).to(dtype).to("cuda:2")

        # Forward
        noise_pred_cond = transformer(
            hidden_states=latent_model_input,
            timestep=timestep / 1000,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_mask,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False
        )[0]

        noise_pred_uncond = transformer(
            hidden_states=latent_model_input,
            timestep=timestep / 1000,
            encoder_hidden_states=neg_embeds,
            encoder_hidden_states_mask=neg_mask,
            img_shapes=img_shapes,
            txt_seq_lens=neg_txt_seq_lens,
            return_dict=False
        )[0]

        # Step (GPU 7)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        cond_norm = torch.norm(noise_pred_cond, dim=-1, keepdim=True)
        noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
        noise_pred = noise_pred * (cond_norm / noise_norm)

        latents = latents.to("cuda:7")
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # --- Step 4: Decode ---
    print("   [4/4] Decoding (GPU 0)...")

    latents = latents.to("cuda:0")
    latents = helper_pipe._unpack_latents(latents, height, width, helper_pipe.vae_scale_factor)
    latents = latents.to(dtype)

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1).to("cuda:0", dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, 16, 1, 1).to("cuda:0", dtype)
    latents = latents / latents_std + latents_mean

    image = vae.decode(latents, return_dict=False)[0]

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")
    image = Image.fromarray(image[0])

torch.cuda.synchronize()
end_time = time.time()

# ==============================================================================
# 7. 保存
# ==============================================================================
save_path = "output_v100_final_success.png"
image.save(save_path)
print("-" * 30)
print(f"🎉 成功！图片已保存至: {save_path}")
print(f"⏱️ 耗时: {end_time - start_time:.2f}s")