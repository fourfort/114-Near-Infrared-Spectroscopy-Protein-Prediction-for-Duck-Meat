from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms

class CustomCocoDataset(CocoDetection):
    def __init__(self, image_dir, ann_file, transform=None):
        super().__init__(image_dir, ann_file)
        self.transform = transform
        # 創建類別 ID 映射
        self.cat_id_map = {cat['id']: i for i, cat in enumerate(self.coco.dataset['categories'])}
        print(f"Category ID map: {self.cat_id_map}")
        # 檢查類別數量是否與 num_classes 匹配
        if len(self.cat_id_map) != 3:
            raise ValueError(f"Expected 3 categories, got {len(self.cat_id_map)}: {self.coco.dataset['categories']}")

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if self.transform:
            img = self.transform(img)
        # 提取單一類別標籤並映射
        if not target:
            label = 0
            print(f"[Warning] Image {index} has no annotations, defaulting to label 0")
        else:
            original_id = target[0]['category_id']
            if original_id not in self.cat_id_map:
                raise ValueError(f"Invalid category_id {original_id} for image {index}, not in {self.cat_id_map}")
            label = self.cat_id_map[original_id]
            print(f"Image {index} category_id: {original_id}, mapped label: {label}")
        # 檢查圖片格式
        if not isinstance(img, torch.Tensor):
            print(f"[Warning] Image {index} is not a Tensor, type: {type(img)}")
        elif img.ndim != 3:
            print(f"[Warning] Image {index} shape mismatch, current shape: {img.shape}")
        elif img.shape[0] != 3:
            print(f"[Warning] Image {index} has {img.shape[0]} channels (expected 3)")
        return img, label

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels

def get_loader_from_yaml(yaml_path, split='train'):
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"YAML file {yaml_path} not found!")
    except yaml.YAMLError:
        raise ValueError("Invalid YAML file format!")

    root = config.get('root', '')
    image_dir = config[split]['images'].replace('${root}', root)
    ann_file = config[split]['annotations'].replace('${root}', root)
    batch_size = config[split].get('batch_size', config.get('batch_size', 16))
    img_size = config[split].get('image_size', config.get('image_size', [640, 640]))

    if not Path(image_dir).exists():
        raise FileNotFoundError(f"Image directory {image_dir} not found!")
    if not Path(ann_file).exists():
        raise FileNotFoundError(f"Annotation file {ann_file} not found!")

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CustomCocoDataset(image_dir, ann_file, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        collate_fn=collate_fn
    )

def check_dataset_dimensions(dataloader):
    print("🔍 Checking Tensor shapes:")
    for batch_idx, (images, labels) in enumerate(dataloader):
        for i, img in enumerate(images):
            print(f"Image {batch_idx * len(images) + i} shape: {img.shape}")
        print(f"Labels: {labels}")
    print("✅ Check complete")

if __name__ == '__main__':
    yaml_path = r"D:\Users\wedke\document\extra document\大三上\113專題研究之製作\論文人生\紅外線\red_coding main\data\mydata.yaml"
    loader = get_loader_from_yaml(yaml_path, split='train')
    check_dataset_dimensions(loader)