import yaml
from pathlib import Path

    # 获得yaml文件名字
yaml_file = Path('test.yaml').name
with open(yaml_file,errors='ignore') as f:
        yaml_ = yaml.safe_load(f)
print(yaml_)