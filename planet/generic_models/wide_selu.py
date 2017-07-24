#-*- coding: utf8 -*-
import math
import torch, torch.nn as nn, torch.nn.functional as F
from future_pytorch import SeluActivation as SELU, AlphaDropout


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.selu1 = SELU()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)

        self.selu2 = SELU()
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.drop_rate = drop_rate
        self.equal_in_out = (in_planes == out_planes)
        self.conv_shortcut = (not self.equal_in_out) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                   padding=0, bias=False) or None
        self.dropout = AlphaDropout(drop_rate)

    def forward(self, x):
        if not self.equal_in_out:
            x = self.selu1(x)
        else:
            out = self.selu1(x)
        out = self.selu2(self.conv1(out if self.equal_in_out else x))

        if self.drop_rate > 0:
            out = self.dropout(out)
        out = self.conv2(out)

        return torch.add(x if self.equal_in_out else self.conv_shortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideSeluResNet(nn.Module):
    def __init__(self, widen_factor, num_classes, layers=[2,2,2,2], input_channels=3, num_channel=16,
                 drop_rate=0.0):
        super(WideSeluResNet, self).__init__()
        block = BasicBlock
        n_channels = [num_channel] + [num_channel*widen_factor*2**i for i in xrange(5)]

        self.conv0 = nn.Conv2d(input_channels, n_channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.selu0 = SELU()
        self.maxpool = nn.MaxPool2d(3,2,1)

        self.block1 = NetworkBlock(layers[0], n_channels[0], n_channels[1], block, 1, drop_rate)
        self.block2 = NetworkBlock(layers[1], n_channels[1], n_channels[2], block, 2, drop_rate)
        self.block3 = NetworkBlock(layers[2], n_channels[2], n_channels[3], block, 2, drop_rate)
        self.block4 = NetworkBlock(layers[3], n_channels[3], n_channels[4], block, 2, drop_rate)

        self.avgpool = nn.AvgPool2d(8, 1)
        self.selu = SELU()
        self.fc = nn.Linear(n_channels[4], num_classes)
        self.n_channels = n_channels[4]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.selu0(self.conv0(x))
        out = self.maxpool(out)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        out = self.selu(out)
        out = self.avgpool(out)
        if out.size(2) != 1:
            out = F.avg_pool2d(out, out.size(2))
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
