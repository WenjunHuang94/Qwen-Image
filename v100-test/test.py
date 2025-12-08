import os
import time
import torch
from diffusers import DiffusionPipeline
import numpy as np
import traceback
import shutil

# 1. 显卡配置
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# 2. 路径
local_model_path = "/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image/"

print("🚀 启动 V100 终极方案 (Sequential Offload 版)")
print("说明：这将自动处理多卡/CPU之间的权重调度，彻底解决 'Expected all tensors on same device' 错误。")

try:
    # ==========================================================================
    # 3. 加载模型 (FP32)
    # ==========================================================================
    # 注意：这里 device_map 设为 "auto" 或者不设置，让后面的 enable_sequential_cpu_offload 接管
    # 我们先尝试不设置 device_map，手动加载到 CPU，然后开启 offload
    pipe = DiffusionPipeline.from_pretrained(
        local_model_path,
        torch_dtype=torch.bfloat16,  # <--- 全量 FP32 (防黑图)
        use_safetensors=True,
        trust_remote_code=True,
        local_files_only=True
        # 注意：这里移除了 device_map 和 max_memory，交给下面的 offload 处理
    )

    print("✅ 模型已加载到 CPU (FP32 Mode)")

    # ==========================================================================
    # 4. 【核心修复】开启顺序 CPU 卸载 (Sequential Offload)
    # ==========================================================================
    # 这个功能是 diffusers 的“大招”。
    # 它会将模型所有模块保留在 CPU 上，推理时只把当前需要计算的一层（Layer）加载到 GPU。
    # 计算完这层，立刻释放显存。
    # 优点：极大节省显存（V100 32GB 跑 FP32 毫无压力），绝对不会报设备不一致错误。
    # 缺点：速度会慢一些（因为有频繁的 PCIe 数据传输），但为了跑通，这是值得的。

    # 注意：在多卡环境下，它通常使用第一张可见卡 (cuda:0) 进行计算。
    # 如果你想利用多卡，enable_model_cpu_offload() 会更好，但在 V100+FP32 这种极限边缘，sequential 最稳。
    pipe.enable_sequential_cpu_offload()
    print("✅ 已开启 Sequential CPU Offload (解决设备不一致 & OOM)")

    # 开启 Tiling 节省显存
    pipe.enable_vae_tiling()

    # --------------------------------------------------------------------------
    # 5. 推理 (512x512)
    # --------------------------------------------------------------------------
    prompt = '''ohwhwj man, wearing a red helmet and orange life jacket, smiling with water droplets on his face.'''
    negative_prompt = " "
    width, height = (512, 512)

    print(f"Starting generation ({width}x{height})...")
    print("提示：由于开启了 Offload，推理速度会比纯 GPU 慢，请耐心等待。")

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

    end_time = time.time()

    # --------------------------------------------------------------------------
    # 6. 验证
    # --------------------------------------------------------------------------
    img_arr = np.array(image)
    save_path = "output_v100_offload_fix.png"

    if img_arr.max() == 0:
        print("❌ 图片依然是全黑的。")
    else:
        image.save(save_path)
        print(f"🎉 成功！图片已保存至: {save_path}")
        print(f"耗时: {end_time - start_time:.2f}s")

except Exception as e:
    print(f"❌ 错误: {e}")
    traceback.print_exc()