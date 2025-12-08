import os
import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import gc

# 引入必要的库
from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel, FlowMatchEulerDiscreteScheduler
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
from accelerate import dispatch_model

# ==============================================================================
# 1. 基础配置
# ==============================================================================
# 使用全部 8 张卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# 你的本地路径
local_model_path = "/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image/"
dtype = torch.float32  # 【绝对核心】FP32 防止 V100 黑图

print("🚀 启动 8xV100 手动流水线方案 (Manual Pipeline)...")
print("   - 策略: 手动将模型拆解到 8 张卡，彻底绕过自动分配的 OOM 坑。")

# ==============================================================================
# 2. 手动加载组件 (逐个加载，精准控制)
# ==============================================================================

# --- A. 加载 Text Encoder (GPU 0) ---
print("\n1. [GPU 0] Loading Text Encoder (~28GB)...")
# Qwen2.5-VL-7B 在 FP32 下很大，我们让 GPU 0 只干这一件事
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    local_model_path, subfolder="text_encoder", torch_dtype=dtype
).to("cuda:0")
text_encoder.eval()

tokenizer = Qwen2Tokenizer.from_pretrained(local_model_path, subfolder="tokenizer")

# --- B. 加载 VAE (GPU 7) ---
print("2. [GPU 7] Loading VAE...")
vae = AutoencoderKLQwenImage.from_pretrained(
    local_model_path, subfolder="vae", torch_dtype=dtype
).to("cuda:7")
vae.eval()
vae.enable_tiling()  # 必须开启，FP32解码极吃显存

# --- C. 加载 Transformer (GPU 1-6) ---
print("3. [GPU 1-6] Loading & Sharding Transformer (~80GB)...")
# 先加载到 CPU，避免初始化时爆显存
transformer = QwenImageTransformer2DModel.from_pretrained(
    local_model_path, subfolder="transformer", torch_dtype=dtype
)
transformer.eval()

# --- 手动切分 Transformer ---
# 这是一个 60 层的深层网络。我们把它切成 6 段，每段 10 层，分别放在 GPU 1 到 6。
# 这样每张卡只占 ~13GB 显存，极其安全。
device_map = {}

# 1. 基础层 (Input Projections) 放 GPU 1
for name, _ in transformer.named_children():
    if name != "transformer_blocks":
        device_map[name] = 1  # cuda:1

# 2. 切分 60 个 Block
num_blocks = 60
cards = [1, 2, 3, 4, 5, 6]  # 使用这6张卡
layers_per_card = 10  # 60 / 6 = 10

for i in range(num_blocks):
    # 计算当前层应该去哪张卡 (0-9->card[0], 10-19->card[1]...)
    card_idx = i // layers_per_card
    if card_idx >= len(cards): card_idx = len(cards) - 1
    target_device = cards[card_idx]

    device_map[f"transformer_blocks.{i}"] = target_device

print(f"   - 切分表生成完毕: 60层 Block 平均分配到 GPU 1 ~ GPU 6")

# 应用切分 (物理移动权重)
transformer = dispatch_model(transformer, device_map=device_map)

# 强制清理一下 CPU 内存
gc.collect()
torch.cuda.empty_cache()
print("✅ 所有模型加载完毕！")

# ==============================================================================
# 3. 准备辅助 Helper
# ==============================================================================
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(local_model_path, subfolder="scheduler")
# 这个 pipeline 只是为了借用它的 encode_prompt 和 pack_latents 方法，不占用显存
helper_pipe = QwenImagePipeline(
    vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
    transformer=None, scheduler=scheduler
)

# ==============================================================================
# 4. 手动推理循环 (Manual Inference Loop)
# ==============================================================================
prompt = "ohwhwj man, wearing a red helmet and orange life jacket, smiling with water droplets on his face."
negative_prompt = " "
width, height = 512, 512  # 你要求的 512
num_inference_steps = 30
guidance_scale = 4.0
seed = 42

print(f"\n🎬 Starting Inference ({width}x{height}) - 30 Steps...")
torch.cuda.synchronize()
import time

start_time = time.time()

with torch.no_grad():
    # --- Step 1: Encode Prompt (在 GPU 0) ---
    print("   [1/4] Encoding Prompt on GPU 0...")
    prompt_embeds, prompt_mask = helper_pipe.encode_prompt(
        prompt=prompt, device="cuda:0", num_images_per_prompt=1
    )
    neg_embeds, neg_mask = helper_pipe.encode_prompt(
        prompt=negative_prompt, device="cuda:0", num_images_per_prompt=1
    )

    # --- Step 2: Prepare Latents (在 GPU 1) ---
    print("   [2/4] Preparing Latents on GPU 1...")
    # 初始噪声放在 Transformer 的入口 (GPU 1)
    latents = torch.randn(
        (1, transformer.config.in_channels // 4, height // 16, width // 16),
        device="cuda:1", dtype=dtype, generator=torch.Generator().manual_seed(seed)
    )
    # Pack latents
    latents = helper_pipe._pack_latents(latents, 1, transformer.config.in_channels // 4, height // 16 * 8,
                                        width // 16 * 8)

    # --- Step 3: Denoising Loop (数据流: GPU 1 -> ... -> GPU 6) ---
    print("   [3/4] Denoising (Pipeline: GPU 1 -> 2 -> 3 -> 4 -> 5 -> 6)...")

    scheduler.set_timesteps(num_inference_steps, device="cuda:1")
    timesteps = scheduler.timesteps

    # 准备辅助变量 (移到 Transformer 入口 GPU 1)
    # accelerate 会自动处理层级间的传输，但我们需要把入口数据喂给 GPU 1
    prompt_embeds = prompt_embeds.to("cuda:1")
    prompt_mask = prompt_mask.to("cuda:1")
    neg_embeds = neg_embeds.to("cuda:1")
    neg_mask = neg_mask.to("cuda:1")

    img_shapes = [(1, height // 16, width // 16)]
    txt_seq_lens = prompt_mask.sum(dim=1).tolist()
    neg_txt_seq_lens = neg_mask.sum(dim=1).tolist()

    for i, t in enumerate(tqdm(timesteps)):
        # 1. 准备 Step 输入 (确保在 GPU 1)
        # 上一步的输出在 GPU 6，需要拉回 GPU 1
        latent_model_input = latents.to("cuda:1")
        timestep = t.expand(latents.shape[0]).to(dtype).to("cuda:1")

        # 2. Transformer Forward
        # 此时数据会自动流转：GPU 1 -> 2 -> 3 -> 4 -> 5 -> 6

        # Positive
        noise_pred_cond = transformer(
            hidden_states=latent_model_input,
            timestep=timestep / 1000,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_mask,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False
        )[0]  # 最终结果在 GPU 6

        # Negative
        noise_pred_uncond = transformer(
            hidden_states=latent_model_input,
            timestep=timestep / 1000,
            encoder_hidden_states=neg_embeds,
            encoder_hidden_states_mask=neg_mask,
            img_shapes=img_shapes,
            txt_seq_lens=neg_txt_seq_lens,
            return_dict=False
        )[0]  # 最终结果在 GPU 6

        # 3. Guidance & Step (在 GPU 6 进行)
        # 这些计算量很小，就地解决
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        cond_norm = torch.norm(noise_pred_cond, dim=-1, keepdim=True)
        noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
        noise_pred = noise_pred * (cond_norm / noise_norm)

        # Update latents
        # 结果保留在 GPU 6，准备下一轮拉回 GPU 1
        latents = latents.to("cuda:6")
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # --- Step 4: Decode (在 GPU 7) ---
    print("   [4/4] Decoding on GPU 7...")

    # 移动到 VAE 所在的卡
    latents = latents.to("cuda:7")

    # Unpack & Denormalize
    latents = helper_pipe._unpack_latents(latents, height, width, helper_pipe.vae_scale_factor)
    latents = latents.to(dtype)

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1).to("cuda:7", dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, 16, 1, 1).to("cuda:7", dtype)
    latents = latents / latents_std + latents_mean

    # Decode
    image = vae.decode(latents, return_dict=False)[0]

    # Post-process
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")
    image = Image.fromarray(image[0])

torch.cuda.synchronize()
end_time = time.time()

# ==============================================================================
# 5. 保存
# ==============================================================================
save_path = "output_v100_manual_512.png"
image.save(save_path)
print("-" * 30)
print(f"🎉 Success! Saved to {save_path}")
print(f"⏱️ Total Time: {end_time - start_time:.2f}s")
print("💡 验证：此图片应该色彩正常（非全黑），且速度比 Offload 快得多。")