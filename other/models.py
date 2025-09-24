import torch
import torch.nn as nn

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, s=1, p=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=autopad(k, p))
        self.bn = nn.BatchNorm2d(out_channels)
        self.act_fn = nn.ReLU(inplace=False)
 
    def forward(self, x):
        return self.act_fn(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        self.conv1 = BaseConv(in_channels, out_channels, k=1, s=1)
        self.conv2 = BaseConv(out_channels, out_channels, k=3, s=1)
        self.add = shortcut and in_channels == out_channels
        if self.add and in_channels != out_channels:
            self.shortcut_conv = BaseConv(in_channels, out_channels, k=1, s=1)
        else:
            self.shortcut_conv = None

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        if self.add:
            if self.shortcut_conv is not None:
                x = self.shortcut_conv(x)
            return x + out
        return out

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ClassificationModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            BaseConv(3, 32, k=3, s=1),
            BaseConv(32, 64, k=1, s=1),
            Bottleneck(64, 64),
            Bottleneck(64, 128),
            nn.Dropout(0.5)  # 添加正則化
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# if __name__ == "__main__":
#     model = ClassificationModel(num_classes=3)
#     x = torch.randn(1, 3, 512, 512)
#     output = model(x)
#     print("Output shape:", output.shape)