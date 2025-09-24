import json
from pathlib import Path

ann_file = r"D:\Users\wedke\document\extra document\大三上\113專題研究之製作\論文人生\紅外線\red_coding main\data\valid\_annotations.coco.json"
with open(ann_file, 'r') as f:
    data = json.load(f)

# Update categories to exclude id: 0
data['categories'] = [
    {"id": 0, "name": "broken", "supercategory": "none"},
    {"id": 1, "name": "dirty", "supercategory": "none"},
    {"id": 2, "name": "normal", "supercategory": "none"}
]

# Remap category_id in annotations from [1, 2, 3] to [0, 1, 2]
for ann in data['annotations']:
    if ann['category_id'] in [1, 2, 3]:
        ann['category_id'] -= 1
    else:
        print(f"Warning: Unexpected category_id {ann['category_id']} in annotation {ann}")

# Save the fixed file
output_file = ann_file.replace('.json', '_fixed.json')
with open(output_file, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Fixed annotations saved to {output_file}")