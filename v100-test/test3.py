import os
import time
import torch
from diffusers import DiffusionPipeline
import numpy as np
import traceback
import shutil

# 1. 显卡配置
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# 2. 路径
local_model_path = "/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image/"

# 3. 准备 Offload 文件夹
offload_folder = "./model_offload"
if not os.path.exists(offload_folder):
    os.makedirs(offload_folder, exist_ok=True)

# ==============================================================================
# 4. 【核心配置】FP32 + 显存限额 (关键修复)
# ==============================================================================
print("🚀 启动 8xV100 终极方案 (FP32 + 显存限额)")

# 【关键】V100 只有 32GB。
# Text Encoder FP32 约 28GB，直接放一张卡必爆。
# 我们限制每张卡只存 20GB 权重。
# 8张卡 x 20GB = 160GB 总容量，远大于模型总需的 ~110GB，足够了。
# 这样会强迫 Text Encoder 和 Transformer 被切碎均匀分布，不仅不爆显存，计算也更均衡。
max_memory_config = {
    0: "20GB", 1: "20GB", 2: "20GB", 3: "20GB",
    4: "20GB", 5: "20GB", 6: "20GB", 7: "20GB",
}

try:
    pipe = DiffusionPipeline.from_pretrained(
        local_model_path,
        torch_dtype=torch.float32,  # 保持 FP32 防黑图
        device_map="balanced",  # 配合 max_memory 使用
        max_memory=max_memory_config,  # <--- 【修复 OOM 的关键】
        offload_folder=offload_folder,
        use_safetensors=True,
        trust_remote_code=True,
        local_files_only=True
    )

    print("✅ 模型加载成功！(Text Encoder 被成功切分)")

    # 开启 Tiling
    pipe.enable_vae_tiling()

    # --------------------------------------------------------------------------
    # 5. 推理 (512x512)
    # --------------------------------------------------------------------------
    prompt = '''ohwhwj man, wearing a red helmet and orange life jacket, smiling with water droplets on his face.'''
    negative_prompt = " "
    width, height = (512, 512)

    print(f"Starting generation ({width}x{height})...")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=30,
        true_cfg_scale=4.0,
        generator=torch.Generator(device="cpu").manual_seed(42)
    ).images[0]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()

    # --------------------------------------------------------------------------
    # 6. 验证
    # --------------------------------------------------------------------------
    img_arr = np.array(image)
    save_path = "output_v100_8gpu_fp32_fix.png"

    if img_arr.max() == 0:
        print("❌ 依然全黑。")
    else:
        image.save(save_path)
        print(f"🎉 成功！图片已保存至: {save_path}")
        print(f"耗时: {end_time - start_time:.2f}s")

except torch.cuda.OutOfMemoryError:
    print("❌ 依然 OOM？")
    print("请尝试将 max_memory_config 进一步调低到 '18GB'。")
    traceback.print_exc()

except Exception as e:
    print(f"❌ 错误: {e}")
    # 如果这里报 'Expected all tensors on same device'，那是 diffusers 的自动切分 bug
    # 遇到那种情况必须换用我之前写的 '手动流水线脚本' (fast_inference_8gpu.py)
    traceback.print_exc()

finally:
    if os.path.exists(offload_folder):
        try:
            shutil.rmtree(offload_folder)
        except:
            pass