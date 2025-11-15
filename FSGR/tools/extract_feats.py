# 文件路径： ~/autodl-tmp/FSGR/tools/extract_feats.py

import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from tqdm import tqdm

# 如果你使用 transformers + CLIPModel
from transformers import CLIPProcessor, CLIPModel

def main():
    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载 CLIP 模型和处理器
    model_name = "openai/clip-vit-base-patch16"
    print(f"Loading CLIP model {model_name} …")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    # 图像预处理方法
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    # 提取图像特征的函数
    def extract_image_features(image_path):
        image = Image.open(image_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]
        with torch.no_grad():
            img_features = model.get_image_features(pixel_values=img_tensor)
        return img_features.cpu().numpy()

    # 两个数据集分割：train2014 和 val2014
    dataset_splits = [
        ("train2014", "datasets/coco/images/train2014", "precomp_feats/vitb16/train2014"),
        ("val2014", "datasets/coco/images/val2014", "precomp_feats/vitb16/val2014"),
    ]

    # 遍历每个数据集
    for split_name, img_dir, out_dir in dataset_splits:
        print(f"\n*** Processing split {split_name} …")
        # 确保输出目录存在
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        # 获取所有图片文件
        image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg") or f.lower().endswith(".png")]
        print(f"Found {len(image_files)} images in {img_dir}")
        for img_file in tqdm(image_files, desc=f"Extracting {split_name}", ncols=80):
            img_path = os.path.join(img_dir, img_file)
            feat = extract_image_features(img_path)  # 获取图像特征
            # 保存特征文件
            out_path = os.path.join(out_dir, img_file + ".npy")
            np.save(out_path, feat)

    print("\nExtraction finished.")

if __name__ == "__main__":
    main()
