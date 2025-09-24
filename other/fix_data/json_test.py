import json

ann_file = r"D:\Users\wedke\document\extra document\大三上\113專題研究之製作\論文人生\紅外線\red_coding main\data\train\_annotations.coco.json"
with open(ann_file, 'r') as f:
    data = json.load(f)

# 檢查 categories
print("Categories:", data['categories'])

# 檢查所有 category_id
category_ids = set(ann['category_id'] for ann in data['annotations'])
print("Unique category_ids in annotations:", category_ids)