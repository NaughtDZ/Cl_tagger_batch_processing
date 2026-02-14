import os
import numpy as np
from PIL import Image
import onnxruntime as ort
import json
import time

# ==================== 配置区 ====================
MODEL_OPTIONS = {
    "cl_tagger_1_02": "cl_tagger_1_02",
    "cl_eva02_tagger_v1_250812": "cl_eva02_tagger_v1_250812",
    "cl_eva02_tagger_v1_250807": "cl_eva02_tagger_v1_250807",
}

DEFAULT_MODEL_KEY = "cl_tagger_1_02"
MODELS_BASE_DIR = "cl_tagger" 
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
# ===============================================

def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    return image

def pil_pad_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height: return image
    new_size = max(width, height)
    new_image = Image.new("RGB", (new_size, new_size), (255, 255, 255))
    paste_position = ((new_size - width) // 2, (new_size - height) // 2)
    new_image.paste(image, paste_position)
    return new_image

def preprocess_image(image: Image.Image, target_size=(448, 448)):
    image = pil_ensure_rgb(image)
    image = pil_pad_square(image)
    image_resized = image.resize(target_size, Image.BICUBIC)
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)  
    img_array = img_array[::-1, :, :]  # RGB -> BGR
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
    img_array = (img_array - mean) / std
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def load_tag_mapping(mapping_path: str):
    with open(mapping_path, 'r', encoding='utf-8') as f:
        tag_mapping_data = json.load(f)
    if isinstance(tag_mapping_data, dict) and "idx_to_tag" in tag_mapping_data:
        idx_to_tag = {int(k): v for k, v in tag_mapping_data["idx_to_tag"].items()}
        tag_to_category = tag_mapping_data["tag_to_category"]
    else:
        tag_mapping_data_int_keys = {int(k): v for k, v in tag_mapping_data.items()}
        idx_to_tag = {idx: data['tag'] for idx, data in tag_mapping_data_int_keys.items()}
        tag_to_category = {data['tag']: data['category'] for data in tag_mapping_data_int_keys.values()}

    names = [None] * (max(idx_to_tag.keys()) + 1)
    categories = {"rating": [], "general": [], "artist": [], "character": [], "copyright": [], "meta": [], "quality": [], "model": []}
    for idx, tag in idx_to_tag.items():
        if idx >= len(names): names.extend([None] * (idx - len(names) + 1))
        names[idx] = tag
        cat = tag_to_category.get(tag, "general").lower()
        if cat in categories: categories[cat].append(idx)

    for key in categories:
        categories[key] = np.array(categories[key], dtype=np.int64)
    return names, categories

def get_tags(probs, names, categories, gen_threshold, char_threshold):
    tags = []
    # 评分和质量标签
    for cat in ["rating", "quality"]:
        if len(categories[cat]) > 0:
            cat_probs = probs[categories[cat]]
            best_idx = np.argmax(cat_probs)
            global_idx = categories[cat][best_idx]
            if global_idx < len(names) and names[global_idx]:
                tags.append(names[global_idx].replace("_", " "))

    # 通用标签
    threshold_map = {
        "general": gen_threshold, "meta": gen_threshold, "model": gen_threshold,
        "character": char_threshold, "copyright": char_threshold, "artist": char_threshold
    }
    for cat, thresh in threshold_map.items():
        if len(categories[cat]) == 0: continue
        cat_probs = probs[categories[cat]]
        selected_local = np.where(cat_probs >= thresh)[0]
        for local_idx in selected_local:
            global_idx = categories[cat][local_idx]
            if global_idx < len(names) and names[global_idx]:
                tag = names[global_idx].replace("_", " ")
                if cat == "meta" and any(x in tag.lower() for x in ['id', 'commentary', 'request', 'mismatch']):
                    continue
                tags.append(tag)
    return ", ".join(tags)

def main():
    print("=== CL EVA02 ONNX 批量打标工具 (DirectML 版) ===")
    print("提示：此版本不依赖 cuDNN，适合 RTX 5090 快速部署\n")

    # 1. 模型选择
    print("可用模型：")
    for i, key in enumerate(MODEL_OPTIONS.keys(), 1):
        print(f"  {i}. {key}")
    choice = input(f"\n请选择模型 [默认 {DEFAULT_MODEL_KEY}]: ").strip()
    model_key = list(MODEL_OPTIONS.keys())[int(choice)-1] if choice.isdigit() and 0 < int(choice) <= len(MODEL_OPTIONS) else DEFAULT_MODEL_KEY

    model_folder = MODEL_OPTIONS[model_key]
    model_path = os.path.join(MODELS_BASE_DIR, model_folder, "model.onnx")
    mapping_path = os.path.join(MODELS_BASE_DIR, model_folder, "tag_mapping.json")

    if not os.path.exists(model_path):
        print(f"✗ 错误：找不到模型文件 {model_path}")
        return

    # 2. 加载标签
    names, categories = load_tag_mapping(mapping_path)

    # 3. 参数输入
    gen_thresh = float(input("\nGeneral 阈值 [默认 0.55]: ").strip() or "0.55")
    char_thresh = float(input("Character 阈值 [默认 0.60]: ").strip() or "0.60")
    folder = input("\n请输入图片文件夹路径: ").strip().strip('"\'')
    recursive = input("是否递归处理子文件夹？(y/N): ").strip().lower() == 'y'

    # 4. 收集图片
    image_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f.lower())[1] in IMAGE_EXTENSIONS:
                image_files.append(os.path.join(root, f))
        if not recursive: break
    
    if not image_files:
        print("✗ 文件夹中未找到支持的图片")
        return

    # 5. 初始化 DirectML Session
    # DirectML 会自动寻找系统中支持 DX12 的最强显卡 (RTX 5090)
    print("\n正在初始化推理引擎 (DirectML)...")
    try:
        # 强制指定使用 DirectML
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
        
        actual_provider = session.get_providers()[0]
        print(f"✓ 推理引擎加载成功: {actual_provider}")
    except Exception as e:
        print(f"✗ 无法启动 DirectML: {e}")
        return

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 6. 批量处理
    print(f"\n开始处理 {len(image_files)} 张图片...\n")
    success, failed = 0, 0
    start_time = time.time()

    for idx, img_path in enumerate(image_files, 1):
        try:
            image = Image.open(img_path)
            input_tensor = preprocess_image(image).astype(np.float32)
            
            # 执行推理
            outputs = session.run([output_name], {input_name: input_tensor})[0]
            
            # Sigmoid 处理
            probs = 1 / (1 + np.exp(-np.clip(outputs[0], -30, 30)))
            tags_text = get_tags(probs, names, categories, gen_thresh, char_thresh)
            
            # 保存结果
            with open(os.path.splitext(img_path)[0] + ".txt", "w", encoding="utf-8") as f:
                f.write(tags_text)
            
            print(f"[{idx}/{len(image_files)}] ✓ {os.path.basename(img_path)}")
            success += 1
        except Exception as e:
            print(f"[{idx}/{len(image_files)}] ✗ {os.path.basename(img_path)}: {e}")
            failed += 1

    total_time = time.time() - start_time
    print(f"\n=== 任务完成 ===")
    print(f"总计用时: {total_time:.2f}秒 (平均 {total_time/len(image_files):.2f}秒/张)")
    print(f"成功: {success}, 失败: {failed}")
    input("\n按 Enter 键退出...")

if __name__ == "__main__":
    main()
