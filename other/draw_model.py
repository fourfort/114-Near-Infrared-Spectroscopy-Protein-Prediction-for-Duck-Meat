import torch
import torch.nn as nn
import yaml
from copy import deepcopy
from models import BaseConv, Bottleneck, Flatten

MODULES = {
    "BaseConv": BaseConv,
    "Bottleneck": Bottleneck,
    "AdaptiveAvgPool2d": nn.AdaptiveAvgPool2d,
    "Flatten": Flatten,
    "Linear": nn.Linear,
}

def parse_model(yaml_cfg, ch):
    layers = []
    out_channels = ch[-1]

    # 解析 backbone
    for i, (f, number, module_name, args) in enumerate(yaml_cfg.get('backbone', [])):
        m = MODULES.get(module_name)
        if m is None:
            raise ValueError(f"Module {module_name} not recognized!")
        args = [eval(a) if isinstance(a, str) else a for a in args]
        if module_name == "BaseConv":
            if len(args) != 3:
                raise ValueError(f"BaseConv expects [out_channels, kernel_size, stride], got {args}")
            out_channels = args[0]  # out_channels from YAML
            in_channels = ch[-1]  # in_channels from previous layer
            args = [in_channels, args[0], args[1], args[2]]  # [in_channels, out_channels, k, s]
        elif module_name == "Bottleneck":
            out_channels = args[0]
            args = [ch[-1], args[0]]  # [in_channels, out_channels]
        layer = nn.Sequential(*[m(*args) for _ in range(number)]) if number > 1 else m(*args)
        layers.append(layer)
        ch.append(out_channels)
        print(f"Layer {i}: {module_name}, in_channels: {ch[-2]}, out_channels: {out_channels}")

    # 解析 head
    for i, (f, number, module_name, args) in enumerate(yaml_cfg.get('head', [])):
        m = MODULES.get(module_name)
        if m is None:
            raise ValueError(f"Module {module_name} not recognized!")
        args = [eval(a) if isinstance(a, str) else a for a in args]
        if module_name == "AdaptiveAvgPool2d":
            args = [args[0]]
        elif module_name == "Linear":
            if args[0] == -1:
                args[0] = ch[-1]
        layer = nn.Sequential(*[m(*args) for _ in range(number)]) if number > 1 else m(*args)
        layers.append(layer)
        if module_name == "Linear":
            out_channels = args[1]
        ch.append(out_channels)
        print(f"Head Layer {i}: {module_name}, in_features: {ch[-2]}, out_features: {out_channels}")

    return nn.Sequential(*layers)

class Model(nn.Module):
    def __init__(self, cfg='model.yaml', ch=3):
        super().__init__()
        try:
            with open(cfg, 'r') as f:
                self.yaml = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"YAML file {cfg} not found!")
        except yaml.YAMLError:
            raise ValueError("Invalid YAML file format!")
        ch = [ch]
        self.model = parse_model(deepcopy(self.yaml), ch)
        print("Model structure:", self.model)

    def forward(self, x):
        return self.model(x)

# if __name__ == "__main__":
#     cfg = 'test.yaml'
#     model = Model(cfg=cfg)
#     model.eval()
#     x = torch.randn(1, 3, 640, 640)
#     output = model(x)
#     print("Output shape:", output.shape)