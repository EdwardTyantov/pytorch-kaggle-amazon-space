#-*- coding: utf8 -*-
import sys, math
import torch, torch.nn as nn, torch.nn.functional as F


def init_modules(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)


def _make_transpose_layer(self, block, planes, blocks, stride=1):
    upsample = None
    if stride != 1:
        upsample = nn.Sequential(
            nn.ConvTranspose2d(self.inplanes, planes * block.expansion,
                               kernel_size=1, stride=2, padding=0, output_padding=1),
            nn.BatchNorm2d(planes * block.expansion),
        )
    elif self.inplanes != planes * block.expansion:
        upsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(self.inplanes, planes, stride, upsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicTransposeBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(BasicTransposeBlock, self).__init__()
        if stride == 1:
            # resolution doesn't change
            self.conv1 = conv3x3(inplanes, planes, stride)
        else:
            self.conv1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)
        out += residual
        out = self.relu(out)

        return out


def main():
    torch.manual_seed(0)
    # model = Encoder(3, BasicBlock, [2, 2, 2, 2])
    model = Decoder(3,)

    model = model.cuda()
    model.eval()

    x = torch.FloatTensor(torch.zeros((1, 256, 8, 8)))
    # x = torch.FloatTensor(torch.zeros((1, 64)))
    input = torch.autograd.Variable(x.cuda())
    res = model(input)
    print res

if __name__ == '__main__':
    sys.exit(main())