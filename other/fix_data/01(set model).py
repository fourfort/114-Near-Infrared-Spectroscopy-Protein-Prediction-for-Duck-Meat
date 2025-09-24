from draw_model import Model
from get_yaml import get_loader_from_yaml

yaml_path = r"D:\Users\wedke\document\extra document\大三上\113專題研究之製作\論文人生\紅外線\red_coding main\data\mydata.yaml"
model = Model("test.yaml")
model.eval()
loader = get_loader_from_yaml(yaml_path, split='train')
images, labels = next(iter(loader))
print(f"Input shape: {images.shape}")
output = model(images)
print(f"Output shape: {output.shape}")