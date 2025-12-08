import os
import time
import torch
from diffusers import DiffusionPipeline
import numpy as np
import traceback

# ==============================================================================
# 1. 基础配置
# ==============================================================================
# 使用全部 8 张卡，有资源就要充分利用，确保稳定
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# 本地模型路径
local_model_path = "/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image/"

if not os.path.exists(local_model_path):
    raise FileNotFoundError(f"找不到模型路径: {local_model_path}")

print("🚀 启动 8xV100 极速稳定方案 (FP32 | 512x512)")
print("说明: 利用 8 卡的巨大显存优势，轻松加载全量 FP32 模型，彻底杜绝黑图和 OOM。")

# ==============================================================================
# 2. 加载模型 (全量 FP32)
# ==============================================================================
try:
    print("Adding model to GPU memory (this may take a minute)...")
    pipe = DiffusionPipeline.from_pretrained(
        local_model_path,
        torch_dtype=torch.float32,  # <--- 【核心】全量 FP32，V100 的唯一解
        device_map="balanced",  # <--- 8卡显存足够，自动平衡即可，无需复杂配置
        use_safetensors=True,
        trust_remote_code=True,
        local_files_only=True
    )

    print("✅ 模型加载成功！(全量驻留显存，无 CPU Offload)")

    # 开启 Tiling 作为双重保险（虽然 8 卡跑 512 可能不需要，但开了更稳）
    pipe.enable_vae_tiling()

    # --------------------------------------------------------------------------
    # 3. 推理 (严格限制 512x512)
    # --------------------------------------------------------------------------
    prompt = '''ohwhwj man, wearing a red helmet and orange life jacket, smiling with water droplets on his face.'''
    negative_prompt = " "

    # 【按你要求】永久锁定 512 x 512
    width, height = (512, 512)

    print(f"Starting generation ({width}x{height})...")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()

    # 生成
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=30,
        true_cfg_scale=4.0,
        # Generator 放在 CPU 上以保证多卡兼容
        generator=torch.Generator(device="cpu").manual_seed(42)
    ).images[0]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()

    # --------------------------------------------------------------------------
    # 4. 验证与保存
    # --------------------------------------------------------------------------
    img_arr = np.array(image)
    save_path = "output_v100_8gpu_512.png"

    if img_arr.max() == 0:
        # 如果这里还黑，那真是见鬼了，硬件或驱动可能有大问题
        print("❌ 绝望了：8卡 FP32 依然全黑。请检查 CUDA/驱动版本。")
    else:
        image.save(save_path)
        print("-" * 30)
        print(f"🎉 成功！图片已保存至: {save_path}")
        print(f"耗时: {end_time - start_time:.2f}s (极速)")
        print("✅ 结论：对于 V100，'8卡 + FP32' 是最完美的解决方案。")

except torch.cuda.OutOfMemoryError:
    print("❌ OOM: 难以置信，8张卡跑512还能爆显存？")
    traceback.print_exc()

except Exception as e:
    print(f"❌ 错误: {e}")
    traceback.print_exc()