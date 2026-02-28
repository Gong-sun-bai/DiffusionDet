import os
import json
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def yolo_to_coco(data_dir, output_dir, train_ratio=0.8):
    """
    data_dir: 包含 jpg 和 txt 的大文件夹路径
    output_dir: 输出的 COCO 格式文件夹路径
    """
    # 创建目录结构
    train_img_dir = os.path.join(output_dir, "train2017")
    val_img_dir = os.path.join(output_dir, "val2017")
    anno_dir = os.path.join(output_dir, "annotations")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(anno_dir, exist_ok=True)

    # 获取所有图片文件
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    train_files, val_files = train_test_split(image_files, train_size=train_ratio, random_state=42)

    def process_files(files, subset_name, target_img_dir):
        categories = [{"id": 1, "name": "ship", "supercategory": "none"}]
        images = []
        annotations = []
        ann_id = 1

        for img_id, img_name in enumerate(tqdm(files, desc=f"Processing {subset_name}")):
            txt_name = img_name.replace('.jpg', '.txt')
            txt_path = os.path.join(data_dir, txt_name)
            img_path = os.path.join(data_dir, img_name)

            if not os.path.exists(txt_path):
                continue

            # 读取图片获取尺寸
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            
            # 复制图片到目标文件夹
            os.system(f"cp {img_path} {os.path.join(target_img_dir, img_name)}")

            images.append({
                "id": img_id,
                "file_name": img_name,
                "width": w,
                "height": h
            })

            # 读取 YOLO 格式标签: class x_center y_center width height (normalized)
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if not parts: continue
                    
                    cls_id = int(parts[0])
                    x_c, y_c, bw, bh = map(float, parts[1:])

                    # 转换为 COCO 格式: [x_min, y_min, width, height] (pixel)
                    abs_x = (x_c - bw / 2) * w
                    abs_y = (y_c - bh / 2) * h
                    abs_w = bw * w
                    abs_h = bh * h

                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1, # 强制设为1，因为SAR通常只有ship
                        "bbox": [abs_x, abs_y, abs_w, abs_h],
                        "area": abs_w * abs_h,
                        "iscrowd": 0
                    })
                    ann_id += 1

        dataset = {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }
        
        with open(os.path.join(anno_dir, f"instances_{subset_name}2017.json"), 'w') as f:
            json.dump(dataset, f)

    process_files(train_files, "train", train_img_dir)
    process_files(val_files, "val", val_img_dir)
    print(f"Done! Dataset saved to {output_dir}")

# 使用示例
ship_data_dir = "/home/bai/demo/DiffusionDet/ship_dataset_v0"
output_coco_dir = "/home/bai/demo/DiffusionDet/SAR_COCO"
yolo_to_coco(ship_data_dir, output_coco_dir)