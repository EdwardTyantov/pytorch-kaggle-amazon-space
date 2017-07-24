#-*- coding: utf8 -*-
import os, sys, math
import torch, torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock
from generic_models.resnet import init_modules, _make_layer
from __init__ import NUM_CLASSES, PRE_TRAINED_TIF_RESNET18


class ResNet(nn.Module):
    def __init__(self, input_dim, block, layers, num_classes=NUM_CLASSES, iplane=32):
        self.block = block
        self.inplanes = self.multiplier = iplane
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, iplane, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(iplane)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = _make_layer(self, block, iplane, layers[0])
        self.layer2 = _make_layer(self, block, iplane*2, layers[1], stride=2)
        self.layer3 = _make_layer(self, block, iplane*4, layers[2], stride=2)
        self.layer4 = _make_layer(self, block, iplane*8, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(8, 1)
        self.fc = nn.Linear(iplane * 8 * block.expansion, num_classes)

        init_modules(self.modules())

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if x.size(2) != 1:
            x = F.avg_pool2d(x, x.size(2))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def tif_resnet18(num_classes=NUM_CLASSES, pretrained=False):
    if pretrained:
        raise ValueError, 'Pretrained for tif is not available'
    model = ResNet(4, BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    layers = None

    return model, layers


def tif_index_resnet18(num_classes=NUM_CLASSES, pretrained=False):
    if pretrained:
        raise ValueError, 'Pretrained for tif is not available'
    model = ResNet(6, BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    layers = None

    return model, layers


def tif_resnet18_fcbn(num_classes=NUM_CLASSES, embedding_len=256, pretrained=True):
    model, _ = tif_resnet18(num_classes)
    if True: #pretrained:
        model = torch.nn.DataParallel(model)  # due to saving parallel table
        checkpoint = torch.load(PRE_TRAINED_TIF_RESNET18)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.module

    model.fc1 = nn.Linear(model.multiplier * 8 * model.block.expansion, embedding_len)
    model.relu_fc1 = nn.ReLU(inplace=True)
    model.fc_bn = nn.BatchNorm1d(embedding_len)
    model.fc = nn.Linear(embedding_len, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu_fc1(self.fc1(x))
        x = self.fc_bn(x)
        x = self.fc(x)

        return x

    setattr(model.__class__, 'forward', forward)

    layers = [(model.fc1, 1), (model.fc, 1), (model.layer4, 1), (model.layer3, 1), (model.layer2, 0.1), (model.layer1, 0.1)]

    return model, layers


def main():
    torch.manual_seed(0)
    #import torch.nn.parameter.Parameter
    #model, _ = tif_index_resnet18()

    #from model import ResNet
    import torch.utils.model_zoo as model_zoo
    model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',}
    model = ResNet(1, BasicBlock, [2, 2, 2, 2], iplane=64, num_classes=1000)
    state_dict = model_zoo.load_url(model_urls['resnet18'])
    st = model.state_dict()
    state_dict['conv1.weight'] = state_dict['conv1.weight'].data[:, :1, :, :]
    state_dict['bn1.weight'] = state_dict['bn1.weight'].data/3.0
    state_dict['bn1.bias'] = state_dict['bn1.bias'].data/3.0
    model.load_state_dict(state_dict)


    print model
    model = model.cuda()
    model.eval()

    x = torch.FloatTensor(torch.zeros((1, 1, 256, 256)))
    input = torch.autograd.Variable(x.cuda())
    res = model(input)
    print res.data.cpu().numpy().mean()
    print res.data.cpu().numpy().std()

if __name__ == '__main__':
    sys.exit(main())