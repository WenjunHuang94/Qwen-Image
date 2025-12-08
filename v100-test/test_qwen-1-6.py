import os
import time
import torch
from diffusers import DiffusionPipeline
from accelerate import dispatch_model, infer_auto_device_map

# ==========================================
# 1. 显卡准备
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

local_model_path = "/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image/"
print(f"Loading Qwen-Image pipeline from {local_model_path}...")

# ==========================================
# 2. 初始加载 (CPU)
# ==========================================
try:
    print(">>> [Step 1] 正在将完整模型加载到 CPU 内存...")
    pipe = DiffusionPipeline.from_pretrained(
        local_model_path,
        torch_dtype=torch.float16,  # 全局默认 FP16 (为了 Text Encoder 和 DiT)
        trust_remote_code=True,
        local_files_only=True,
        use_safetensors=True,
    )
    print(">>> 模型已加载到 CPU。")

except Exception as e:
    print(f"加载失败: {e}")
    exit()

# ==========================================
# 3. 手动分发 (关键：混合精度策略)
# ==========================================
try:
    # --- A. Text Encoder -> GPU 0 (FP16) ---
    print(">>> [Step 2] 搬运 Text Encoder -> GPU 0 (FP16)...")
    # T5 FP16 占用 15.5GB。
    # 绝大多数情况下，Text Encoder 用 FP16 是安全的，不会导致黑图。
    # 如果真的运气不好黑图了，那只能把它放到 CPU 上跑 FP32。
    pipe.text_encoder = pipe.text_encoder.to("cuda:0", dtype=torch.float16)

    # --- B. VAE -> GPU 0 (FP32) ---
    print(">>> [Step 2.1] 搬运 VAE -> GPU 0 (FP32)...")
    # 【核心！】VAE 必须 FP32，否则 100% 黑图/噪点。
    # VAE 很小，FP32 也才 0.6GB，GPU 0 完全放得下 (15.5 + 0.6 = 16.1GB)
    pipe.vae = pipe.vae.to("cuda:0", dtype=torch.float32)

    # --- C. 切分 Transformer (FP16) ---
    print(">>> [Step 3] 计算 Transformer 切分方案 (FP16)...")

    # 获取 Block 类名
    transformer_block_class = pipe.transformer.transformer_blocks[0].__class__

    # 显存预算
    transformer_memory_config = {1: "30GB", 2: "30GB", 3: "30GB"}

    # 计算切分表
    device_map = infer_auto_device_map(
        pipe.transformer,
        max_memory=transformer_memory_config,
        no_split_module_classes=[transformer_block_class.__name__],
        dtype=torch.float16
    )

    print(">>> [Step 4] 执行物理切分 (CPU -> GPUs)...")
    pipe.transformer = dispatch_model(pipe.transformer, device_map)

    print(">>> ✅ 模型分发完毕！")
    print(f"    Text Enc (FP16): {pipe.text_encoder.device}")
    print(f"    VAE (FP32):      {pipe.vae.device}")
    print(f"    DiT (FP16):      分散在 GPU 1, 2, 3")

except Exception as e:
    print(f"分发失败: {e}")
    exit()

# ==========================================
# 4. 推理
# ==========================================
positive_magic = ", Ultra HD, 4K, cinematic composition."
prompt = "ohwhwj man, wearing a red helmet and orange life jacket, smiling with water droplets on his face."
negative_prompt = " "
width, height = (512, 512)

print("\nStarting image generation...")

if torch.cuda.is_available():
    torch.cuda.synchronize()
start_time = time.time()

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

image.save("qwen_t2i_v100_perfect.png")
print("Done.")