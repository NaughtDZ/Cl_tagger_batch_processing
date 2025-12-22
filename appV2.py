import os
import numpy as np
from PIL import Image
import onnxruntime as ort
import json
import time

# ==================== 配置区 ====================
# 支持的模型列表（键是显示名称，值是相对于 models 的子文件夹名）
MODEL_OPTIONS = {
    "cl_tagger_1_02": "cl_tagger_1_02",
    "cl_eva02_tagger_v1_250812": "cl_eva02_tagger_v1_250812",
    "cl_eva02_tagger_v1_250807": "cl_eva02_tagger_v1_250807",
    # 添加更多版本时，在这里加一行即可
}

DEFAULT_MODEL_KEY = "cl_tagger_1_02"
MODELS_BASE_DIR = "cl_tagger"  # 模型根目录，相对于脚本所在位置

# 支持的图片扩展名
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
    if width == height:
        return image
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
    img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
    img_array = img_array[::-1, :, :]  # RGB -> BGR (根据原项目经验)
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
        if idx >= len(names):
            names.extend([None] * (idx - len(names) + 1))
        names[idx] = tag
        cat = tag_to_category.get(tag, "general")
        if cat.lower() in categories:
            categories[cat.lower()].append(idx)

    for key in categories:
        categories[key] = np.array(categories[key], dtype=np.int64)

    return names, categories

def get_tags(probs: np.ndarray, names: list, categories: dict, gen_threshold: float, char_threshold: float):
    tags = []
    # Rating & Quality: 取最高概率
    for cat in ["rating", "quality"]:
        if len(categories[cat]) > 0:
            cat_probs = probs[categories[cat]]
            best_idx = np.argmax(cat_probs)
            global_idx = categories[cat][best_idx]
            if global_idx < len(names) and names[global_idx] is not None:
                tags.append(names[global_idx].replace("_", " "))

    # 其他类别：阈值过滤
    threshold_map = {
        "general": gen_threshold, "meta": gen_threshold, "model": gen_threshold,
        "character": char_threshold, "copyright": char_threshold, "artist": char_threshold
    }
    for cat, thresh in threshold_map.items():
        if len(categories[cat]) == 0:
            continue
        cat_probs = probs[categories[cat]]
        mask = cat_probs >= thresh
        selected_local = np.where(mask)[0]
        for local_idx in selected_local:
            global_idx = categories[cat][local_idx]
            if global_idx < len(names) and names[global_idx] is not None:
                tag = names[global_idx].replace("_", " ")
                # 过滤无用 meta tag
                if cat == "meta" and any(x in tag.lower() for x in ['id', 'commentary', 'request', 'mismatch']):
                    continue
                tags.append(tag)

    return ", ".join(tags)

def main():
    print("=== CL EVA02 ONNX 批量打标工具 ===\n")

    # 1. 选择模型
    print("可用模型：")
    for i, key in enumerate(MODEL_OPTIONS.keys(), 1):
        print(f"  {i}. {key}")
    while True:
        choice = input(f"\n请选择模型 [默认 {DEFAULT_MODEL_KEY}]: ").strip()
        if choice == "":
            model_key = DEFAULT_MODEL_KEY
            break
        if choice in MODEL_OPTIONS:
            model_key = choice
            break
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(MODEL_OPTIONS):
                model_key = list(MODEL_OPTIONS.keys())[idx]
                break
        except:
            pass
        print("输入无效，请重新选择。")

    model_folder = MODEL_OPTIONS[model_key]
    model_path = os.path.join(MODELS_BASE_DIR, model_folder, "model.onnx")
    mapping_path = os.path.join(MODELS_BASE_DIR, model_folder, "tag_mapping.json")

    if not os.path.exists(model_path):
        print(f"✗ 模型文件不存在: {model_path}")
        return
    if not os.path.exists(mapping_path):
        print(f"✗ tag_mapping.json 不存在: {mapping_path}")
        return

    print(f"✓ 加载模型: {model_key}")

    # 2. 加载标签
    names, categories = load_tag_mapping(mapping_path)

    # 3. 设置阈值
    try:
        gen_thresh = float(input("\nGeneral/Meta/Model 阈值 [默认 0.55]: ").strip() or "0.55")
        char_thresh = float(input("Character/Copyright/Artist 阈值 [默认 0.60]: ").strip() or "0.60")
    except ValueError:
        print("阈值输入无效，使用默认值")
        gen_thresh, char_thresh = 0.55, 0.60

    # 4. 输入文件夹
    folder = input("\n请输入图片文件夹路径: ").strip().strip('"\'')
    if not os.path.isdir(folder):
        print("✗ 路径不存在或不是文件夹")
        return

    recursive = input("是否递归处理子文件夹？(y/N): ").strip().lower() == 'y'

    # 5. 收集图片文件
    image_files = []
    walk = os.walk(folder)
    for root, _, files in walk:
        for f in files:
            if os.path.splitext(f.lower())[1] in IMAGE_EXTENSIONS:
                image_files.append(os.path.join(root, f))
        if not recursive:
            break  # 只处理顶级文件夹

    if not image_files:
        print("✗ 指定文件夹中未找到支持的图片文件")
        return

    print(f"\n找到 {len(image_files)} 张图片，开始处理...\n")

    # 6. 加载 ONNX Session（只加载一次）
    providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    success = 0
    failed = 0

    for idx, img_path in enumerate(image_files, 1):
        try:
            image = Image.open(img_path).convert("RGB")
            input_tensor = preprocess_image(image)
            input_tensor = input_tensor.astype(np.float32)

            outputs = session.run([output_name], {input_name: input_tensor})[0]
            probs = 1 / (1 + np.exp(-np.clip(outputs[0], -30, 30)))  # stable sigmoid

            tags_text = get_tags(probs, names, categories, gen_thresh, char_thresh)

            txt_path = os.path.splitext(img_path)[0] + ".txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(tags_text)

            print(f"[{idx}/{len(image_files)}] ✓ {os.path.basename(img_path)}")
            success += 1

        except Exception as e:
            print(f"[{idx}/{len(image_files)}] ✗ {os.path.basename(img_path)} -> {e}")
            failed += 1

    print(f"\n=== 完成！成功: {success} 张，失败: {failed} 张 ===")
    input("\n按 Enter 键退出...")

if __name__ == "__main__":
    main()