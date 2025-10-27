import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class Channel_Attention(nn.Module):
    def __init__(self, in_channels=30):
        super().__init__()
        # 高度方向压缩：每个通道独立处理
        self.conv_h = nn.Conv2d(in_channels, in_channels,
                              kernel_size=(175, 1),    # 覆盖整个高度维度
                              groups=in_channels)      # 通道独立计算
        # 宽度方向压缩：每个通道独立处理
        self.conv_w = nn.Conv2d(in_channels, in_channels,
                              kernel_size=(1, 24),     # 覆盖整个宽度维度
                              groups=in_channels)
        self.activate = nn.LeakyReLU()

    def forward(self, x):
        identity = x
        x = self.conv_h(x)  # 形状变换 [64,30,175,24] → [64,30,1,24]
        x = self.conv_w(x)  # 形状变换 [64,30,1,24] → [64,30,1,1]
        return identity*self.activate(x) + identity  #残差连接F(x)+x

class DenseLayer(nn.Module):
    def __init__(self, in_channel,out_channel,h,w):
        super().__init__()
        self.conv3 = nn.Conv2d(in_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1)
        self.conv1 = nn.Conv2d(in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0)
        self.bn = nn.LayerNorm([h,w])
        self.activate = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        identity = self.conv1(x)
        x = self.conv3(x)
        x = self.bn(x)
        x = self.activate(x)
        return torch.cat((x, identity), 2)
class TransitionLayer(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.bn = nn.LayerNorm([350,24])
        self.activate = nn.LeakyReLU(inplace=True)
        self.conv_h = nn.Conv2d(in_channel,out_channels=12,kernel_size=3,stride=1,padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = self.bn(x)
        x = self.activate(x)
        x = self.conv_h(x)
        x = self.maxpool(x)
        return x
class CADCNN(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        num_class = 0
        if in_channels==30:
            num_class = 123
        elif in_channels==32:
            num_class = 32
        self.ca = Channel_Attention(in_channels)
        self.first_conv = nn.Conv2d(in_channels, 12, kernel_size=3, stride=1, padding=1)
        self.dense1 = DenseLayer(12,24,175,24)
        self.transition1 = TransitionLayer(24)
        self.dense2 = DenseLayer(12,24,175,12)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(24, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_class)
        )
    def forward(self, x):
        x = self.ca(x)
        x = self.first_conv(x)
        x = self.dense1(x)
        x = self.transition1(x)
        x = self.dense2(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
if __name__ == '__main__':
    test = torch.randn(64,30,175,24)
    model = CADCNN(30)
    out = model(test)
    print(out.shape)