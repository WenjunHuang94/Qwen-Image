import os
import shutil  # 导入 shutil 用于复制文件
from PIL import Image
import torch
from diffusers import QwenImageEditPipeline
import time

# --- 1. 配置 ---

# 您的原始头像图片（“控制图”的来源）
source_image_path = "./001.jpg"

# 训练数据规范要求的目录
target_dir = "./images"  # 存放“目标图”和“指令txt”
control_dir = "./control_images"  # 存放“控制图”（原图）

# 确保目录存在
os.makedirs(target_dir, exist_ok=True)
os.makedirs(control_dir, exist_ok=True)

# 【关键】这是 10 个不同的编辑指令
edit_prompts = [
    # 第 1 个 (您已有的)
    "Change the color of the person's clothing to purple with a flash as the background.",
    # 第 2 个 (换发色)
    "Change the person's hair to bright blue.",
    # 第 3 个 (换风格)
    "Turn the entire image into a pencil sketch.",
    # 第 4 个 (换背景)
    "Change the background to a sunny beach.",
    # 第 5 个 (加配饰)
    "Add a pair of stylish sunglasses to the person.",
    # 第 6 个 (换风格)
    "Make the image in a pixel art style.",
    # 第 7 个 (换衣服)
    "Change the shirt to a red jacket.",
    # 第 8 个 (换背景)
    "Place the person on the moon, with Earth in the background.",
    # 第 9 个 (换风格)
    "Turn the image into a Van Gogh style painting.",
    # 第 10 个 (改细节)
    "Give the person bright green eyes and add freckles."
]

# --- 2. 加载模型 ---
print("Loading Qwen-Image-Edit pipeline (这可能需要一点时间)...")
pipeline = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
print("Pipeline loaded successfully.")
pipeline.set_progress_bar_config(disable=None)

# --- 3. 加载一次“控制图”原图 ---
try:
    source_image = Image.open(source_image_path).convert("RGB")
    print(f"Loaded source control image from: {source_image_path}")
except FileNotFoundError:
    print(f"错误: 找不到源文件 {source_image_path}。请检查文件路径。")
    exit()

print("=" * 60)
print(f"开始生成 {len(edit_prompts)} 组训练数据...")
print("=" * 60)

# --- 4. 循环生成数据 ---

total_start_time = time.time()

for i, prompt in enumerate(edit_prompts):

    file_index = i + 1
    # 格式化文件名为 "001", "002", ... "010"
    basename = f"{file_index:03d}"

    print(f"\n--- 正在处理第 {file_index}/{len(edit_prompts)} 组数据 ---")

    # --- A. 准备“控制图” (复制原图) ---
    control_image_path = os.path.join(control_dir, f"{basename}.jpg")
    # 从源文件复制，这是最快的方式
    shutil.copy(source_image_path, control_image_path)
    print(f"  [控制图] 已保存: {control_image_path}")

    # --- B. 准备“指令”和“目标图” (AI生成) ---
    target_image_path = os.path.join(target_dir, f"{basename}.jpg")
    target_txt_path = os.path.join(target_dir, f"{basename}.txt")

    print(f"  [指 令] {prompt}")

    # 保存“指令” .txt 文件
    try:
        with open(target_txt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"  [指令TXT] 已保存: {target_txt_path}")
    except Exception as e:
        print(f"  [错误] 保存TXT文件失败: {e}")
        continue  # 跳过这个循环

    # 设置 AI 输入
    inputs = {
        "image": source_image,  # 始终使用“原图”作为输入
        "prompt": prompt,  # 使用当前循环的指令
        "generator": torch.manual_seed(42),  # 固定种子以便复现, 您可以换个数字
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 50,
    }

    print("  [目标图] 正在生成 (请稍候)...")
    gen_start_time = time.time()

    # --- C. 运行 AI ---
    try:
        with torch.inference_mode():
            output = pipeline(**inputs)
            output_image = output.images[0]

            # 保存“目标图” .jpg 文件
            output_image.save(target_image_path)
            gen_time = time.time() - gen_start_time
            print(f"  [目标图] 已保存: {target_image_path} (耗时: {gen_time:.2f} 秒)")

    except Exception as e:
        print(f"  [错误] AI 生成失败: {e}")

total_time = time.time() - total_start_time
print("=" * 60)
print("=== 数据集生成完毕! ===")
print(f"总耗时: {total_time:.2f} 秒 ({total_time / 60:.2f} 分钟)")
print(f"请检查 '{target_dir}' 和 '{control_dir}' 文件夹。")