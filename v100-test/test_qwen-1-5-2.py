import os
import time
import torch
from diffusers import DiffusionPipeline
from accelerate import dispatch_model, infer_auto_device_map
import gc

# ==========================================
# 0. 清理环境
# ==========================================
torch.cuda.empty_cache()
gc.collect()

# 设置显存分配策略，减少碎片化 (针对报错建议)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

local_model_path = "/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image/"
print(f"Loading Qwen-Image pipeline from {local_model_path}...")

# ==========================================
# 1. 初始加载 (CPU, FP32)
# ==========================================
try:
    print(">>> [Step 1] 正在将完整模型加载到 CPU 内存 (FP32)...")
    pipe = DiffusionPipeline.from_pretrained(
        local_model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        local_files_only=True,
        use_safetensors=True,
    )
    print(">>> 模型已加载到 CPU。")

except Exception as e:
    print(f"加载失败: {e}")
    exit()

# ==========================================
# 2. 自动分发策略 (8卡全开模式)
# ==========================================
try:
    # -------------------------------------------------
    # A. 切分 Text Encoder -> GPU 0, 1, 2
    #    原因：FP32下约31GB，2张卡太挤，3张卡(每张10GB)最稳
    # -------------------------------------------------
    print(">>> [Step 2] 计算 Text Encoder 切分方案 (GPU 0, 1, 2)...")

    # 限制每张卡只用 12GB，留出 20GB 给系统和推理计算
    te_memory_config = {0: "12GB", 1: "12GB", 2: "12GB"}

    te_device_map = infer_auto_device_map(
        pipe.text_encoder,
        max_memory=te_memory_config,
        dtype=torch.float32
    )

    print(f">>> Text Encoder 切分结果: {te_device_map}")
    print(">>> 执行 Text Encoder 物理分发...")
    pipe.text_encoder = dispatch_model(pipe.text_encoder, te_device_map)

    # -------------------------------------------------
    # B. 切分 Transformer -> GPU 3, 4, 5, 6
    #    原因：FP32下约76GB，4张卡(每张19GB)非常稳
    # -------------------------------------------------
    print(">>> [Step 3] 计算 Transformer 切分方案 (GPU 3, 4, 5, 6)...")

    transformer_block_class = pipe.transformer.transformer_blocks[0].__class__
    print(f">>> 锁定原子切分层类名: {transformer_block_class.__name__}")

    tr_memory_config = {
        3: "25GB",
        4: "25GB",
        5: "25GB",
        6: "25GB"
    }

    tr_device_map = infer_auto_device_map(
        pipe.transformer,
        max_memory=tr_memory_config,
        no_split_module_classes=[transformer_block_class.__name__],
        dtype=torch.float32
    )

    print(">>> Transformer 切分预览 (前5层):", {k: v for i, (k, v) in enumerate(tr_device_map.items()) if i < 5})
    print(">>> 执行 Transformer 物理分发...")
    pipe.transformer = dispatch_model(pipe.transformer, tr_device_map)

    # -------------------------------------------------
    # C. 搬运 VAE -> GPU 7
    # -------------------------------------------------
    print(">>> [Step 4] 搬运 VAE -> GPU 7 ...")
    pipe.vae = pipe.vae.to("cuda:7")

    print(">>> ✅ 所有模型分发完毕！")
    print(f"    Text Encoder: GPU 0, 1, 2")
    print(f"    Transformer:  GPU 3, 4, 5, 6")
    print(f"    VAE:          GPU 7")

except Exception as e:
    import traceback

    traceback.print_exc()
    print(f"分发失败: {e}")
    exit()

# ==========================================
# 4. 推理
# ==========================================
positive_magic = ", Ultra HD, 4K, cinematic composition."
prompt = "ohwhwj man, wearing a red helmet and orange life jacket, smiling with water droplets on his face."
negative_prompt = " "
width, height = (512, 512)

print("\nStarting image generation (FP32 on V100 x 8)...")

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

image.save("qwen_t2i_v100_fp32_sharded.png")
print("Done.")