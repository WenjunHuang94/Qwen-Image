import os
import time
import torch
from diffusers import DiffusionPipeline
from accelerate import dispatch_model, infer_auto_device_map

# ==========================================
# 1. 显卡准备
# ==========================================
# GPU 0: Text Encoder + VAE
# GPU 1, 2, 3: Transformer (分层切分)
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
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
        use_safetensors=True,
    )
    print(">>> 模型已加载到 CPU。")

except Exception as e:
    print(f"加载失败: {e}")
    exit()

# ==========================================
# 3. 手动分发 (根据源码修正)
# ==========================================
try:
    # --- A. 搬运小组件到 GPU 0 ---
    print(">>> [Step 2] 搬运 Text Encoder & VAE -> GPU 0 ...")
    pipe.text_encoder = pipe.text_encoder.to("cuda:0")
    pipe.vae = pipe.vae.to("cuda:0")

    # --- B. 切分 Transformer ---
    print(">>> [Step 3] 计算 Transformer 切分方案...")

    # 1. 【核心修改】正确获取 Block 的类名
    # 根据源码: self.transformer_blocks = nn.ModuleList(...)
    # 我们取第一个 block 的类型，防止 accelerate 把它从中间切开
    transformer_block_class = pipe.transformer.transformer_blocks[0].__class__
    print(f">>> 锁定原子切分层类名: {transformer_block_class.__name__}")  # 应该是 QwenImageTransformerBlock

    # 2. 显存预算：GPU 1, 2, 3 全力支持 Transformer
    transformer_memory_config = {
        1: "30GB",
        2: "30GB",
        3: "30GB"
    }

    # 3. 计算切分表
    # no_split_module_classes 保证了 GPU 1 算完一整层，再把数据传给 GPU 2
    device_map = infer_auto_device_map(
        pipe.transformer,
        max_memory=transformer_memory_config,
        no_split_module_classes=[transformer_block_class.__name__],
        dtype=torch.float16
    )

    print(">>> 切分方案预览:", {k: v for i, (k, v) in enumerate(device_map.items()) if i < 5})

    print(">>> [Step 4] 执行物理切分 (CPU -> GPUs)...")
    pipe.transformer = dispatch_model(pipe.transformer, device_map)

    print(">>> ✅ 模型分发完毕！")
    print(f"    Transformer 分布: {sorted(list(set(device_map.values())))}")

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

image.save("qwen_t2i_sharded_final.png")
print("Done.")