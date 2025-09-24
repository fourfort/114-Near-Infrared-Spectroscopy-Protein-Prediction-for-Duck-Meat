from copy import deepcopy
 
from models import BaseConv, Bottleneck
import torch.nn as nn
import os
 
path = os.getcwd()
from pathlib import Path
import torch
 
 
def parse_model(yaml_cfg, ch):
    """
    :param yaml_cfg: yaml file
    :param ch: init in_channels default is 3
    :return: model
    """
 
    layer, out_channels = [], ch[-1]
    for i, (f, number, Module_name, args) in enumerate(yaml_cfg['backbone']):
        """
        f:上一层输出通道
        number:该模块有几层，就是该模块要重复几次
        Mdule_name：卷积层名字
        args:参数，包含输出通道数，k,s,p等
        """
        # 通过eval，将str类型转自己定义的BaseConv
        m = eval(Module_name) if isinstance(Module_name, str) else Module_name
        for j, a in enumerate(args):
            # 通过eval，将str转int,获得输出通道数
            args[j] = eval(a) if isinstance(a, str) else a
        # 更新通道
        # args[0]是输出通道
        if m in [BaseConv, Bottleneck]:
            in_channels, out_channels = ch[f], args[0]
            args = [in_channels, out_channels, *args[1:]]  # args=[in_channels, out_channels, k, s, p]
 
        # 将参数传入模型
        model_ = nn.Sequential(*[m(*args) for _ in range(number)]) if number > 1 else m(*args)
        # 更新通道列表，每次获取输出通道
        ch.append(out_channels)
        layer.append(model_)
    return nn.Sequential(*layer)
 
 
class Model(nn.Module):
    def __init__(self, cfg='./test.yaml', ch=3, ):
        super().__init__()
        self.yaml = cfg
        import yaml
        yaml_file = Path(cfg).name
        with open(yaml_file, errors='ignore')as f:
            self.yaml = yaml.safe_load(f)
 
        ch = self.yaml["ch"] = self.yaml["ch"] = 3
        self.backbone = parse_model(deepcopy(self.yaml), ch=[ch])
 
    def forward(self, x):
        output = self.backbone(x)
        return output
 
 
if __name__ == "__main__":
    cfg = path + '/test.yaml'
    model = Model()
    model.eval()
    print(model)
    x = torch.ones(1, 3, 512, 512)
    output = model(x)
    torch.save(model, "model.pth")
 
 
 
    # model = torch.load('model.pth')
    # model.eval()
    # x = torch.ones(1,3,512,512)
    # input_name = ['input']
    # output_name = ['output']
    # torch.onnx.export(model, x, 'myonnx.onnx', verbose=True)
 