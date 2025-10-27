import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict



class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class Channel_Attention(nn.Module):
    def __init__(self,input_channel,mid_ratio=8):
        super(Channel_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(input_channel, input_channel // mid_ratio, 1, bias=False)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.fc2   = nn.Conv2d(input_channel // mid_ratio, input_channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
#早期代号为SA，故一直沿用
class Spatial_Attention(nn.Module):
    def __init__(self,input_channel,isEA,isTA,mid_ratio=8):
        super(Spatial_Attention, self).__init__()
        self.isEA = isEA
        self.isTA = isTA
        if isEA:
            self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
            self.conv1_h = nn.Conv2d(input_channel, input_channel // mid_ratio, 1, bias=False)
            self.bn1_h = nn.LayerNorm(88)
            self.conv2_h = nn.Conv2d(input_channel // mid_ratio, input_channel, 1, bias=False)
        if isTA:
            self.pool_w = nn.AdaptiveAvgPool2d((1, None))
            self.conv1_w = nn.Conv2d(input_channel, input_channel // mid_ratio, 1, bias=False)
            self.bn1_w = nn.LayerNorm(12)
            self.conv2_w = nn.Conv2d(input_channel // mid_ratio, input_channel, 1, bias=False)

        self.act = h_swish()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_h = 1
        x_w = 1
        if self.isEA:
            x_h = self.act(self.bn1_h(self.conv1_h(self.pool_h(x)).squeeze(3)).unsqueeze(3))
            x_h = self.conv2_h(x_h)
            x_h = self.sigmoid(x_h)

        if self.isTA:

            x_w = self.act(self.bn1_w(self.conv1_w(self.pool_w(x))))
            x_w = self.conv2_w(x_w)
            x_w = self.sigmoid(x_w)

        out = x_h*x_w
        return out


class Attention(nn.Module):
    def __init__(self,input_channel,isCA,isEA,isTA,mid_ratio=8):
        super(Attention, self).__init__()
        self.isCA = isCA
        self.isEA = isEA
        self.isTA = isTA
        if isCA:
            self.CA = Channel_Attention(input_channel,mid_ratio)
        if isEA or isTA:
            self.SA = Spatial_Attention(input_channel,isEA,isTA,mid_ratio)
    def forward(self, x):
        identity = x
        CA_out = 1
        SA_out = 1
        if self.isCA:
            CA_out = self.CA(x)  # 通道注意力机制 输出结果为B*D*1*1,为权重系数

        if self.isEA or self.isTA:
            SA_out = self.SA(x)#空间注意力机制 输出结果为B*1*H*W，为权重系数
        output =  identity*(CA_out*SA_out) + identity

        return output


class _DenseLayer(nn.Sequential):
    def __init__(self,h_w, num_input_features, growth_rate, bn_size,
                 drop_rate):  # 第一个参数是输入的通道数，第二个是增长率是一个重要的超参数，它控制了每个密集块中特征图的维度增加量，
        super(_DenseLayer, self).__init__()  # 调用父类的构造方法，这句话的意思是在调用nn.Sequential的构造方法
        self.add_module('norm1', nn.LayerNorm(h_w)),  # 批量归一化
        self.add_module('relu1', nn.ReLU(inplace=True)),  # ReLU层
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),  # 表示其输出为4*k   其中bn_size等于4，growth_rate为k     不改变大小，只改变通道的个数
        self.add_module('norm2', nn.LayerNorm(h_w)),  # 归一化
        self.add_module('relu2', nn.ReLU(inplace=True)),  # 激活函数
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),  # 输出为growth_rate：表示输出通道数为k,增加特征图维度的大小
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        #因为drop_rate都是0，所以是没有实际效果的
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)  # 通道维度连接


class _DenseBlock(nn.Sequential):  # 构建稠密块
    def __init__(self,h_w, num_layers, num_input_features, bn_size, growth_rate, drop_rate):  # 密集块中密集层的数量，第二参数是输入通道数量
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(h_w,num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

#由于继承的是Sequential，将会自动创建forward，依次执行所有层级
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):  # 输入通道数 输出通道数
        super(_Transition, self).__init__()
        self.add_module('norm', nn.LayerNorm([88,12]))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


# DenseNet网络模型基本结构
class DenseNet(nn.Module):
    def __init__(self,isAttention,isCA,isEA,isTA,growth_rate=32, block_config=(3,6),
                 num_init_features=64, bn_size=4, drop_rate=0, ):

        super(DenseNet, self).__init__()
        # First convolution
        layers = [
            ('conv0', nn.Conv2d(30, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm0', nn.LayerNorm([175, 24])),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]
        self.features = nn.Sequential(OrderedDict(layers))
        # Each denseblock

        num_features = num_init_features
        block = _DenseBlock(h_w=[88,12],  num_layers=block_config[0], num_input_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.features.add_module('denseblock1', block)
        num_features = num_features + block_config[0] * growth_rate
        if isAttention:
            self.features.add_module('attention', Attention(num_features,isCA,isEA,isTA))

        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        self.features.add_module('transition1', trans)
        num_features = num_features // 2


        block = _DenseBlock(h_w=[44, 6], num_layers=block_config[1], num_input_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.features.add_module('denseblock2', block)
        num_features = num_features + block_config[1] * growth_rate

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))


        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(224, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 123)
        )

        # Official init from torch repo.
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal(m.weight.data)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.reshape(x.shape[0], -1)  # 将x展平，输入全连接层
        out = self.fc(out)
        return out


def densenet(isAttention=True,isCA=True,isEA=True,isTA=True):
    model = DenseNet(isAttention=isAttention,isCA=isCA,isEA=isEA,isTA=isTA,num_init_features=64, growth_rate=32, block_config=(4, 4) )
    return model

