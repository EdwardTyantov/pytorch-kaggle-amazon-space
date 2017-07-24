#-*- coding: utf8 -*-
import os, sys, logging
import torch, torch.nn as nn
import math, urllib2
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchvision.models import densenet121, densenet169, densenet201
from __init__ import PRE_TRAINED_RESNET18, NUM_CLASSES, MODEL_FOLDER
import tif_model
import mix_model


logger = logging.getLogger('app')
logger.setLevel(logging.DEBUG)

__all__ = ['ResNet', 'resnet18']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}

model_cache_path = '/tmp/model_cache_%s.pth'


curry = lambda func, *args, **kw:\
            lambda *p, **n:\
                 func(*args + p, **dict(kw.items() + n.items()))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.__block = block
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(8, 1)
        #adapt gradient
        #def hook_a(module, grad_input, grad_output):
        #    return tuple(map(lambda x: x.mul(0.5), grad_input))
        #self.avgpool.register_backward_hook(hook_a)
        #
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def reinit_fc(self, num_classes):
        self.fc = nn.Linear(512 * self.__block.expansion, num_classes)

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


def try_load_model(model_name):
    try:
        _model_loaded = model_zoo.load_url(model_urls[model_name])
    except urllib2.URLError as _:
        logger.error('Urllib failes to get through SSL, downloading using wget')
        fname = model_cache_path % model_name
        if not os.path.exists(fname):
            command_name = 'wget -O - %s > %s' % (model_urls[model_name], fname)
            os.system(command_name)
        _model_loaded = torch.load(fname)
    return _model_loaded


def resnet18(num_classes, pretrained=False):
    """Constructs a ResNet-18 model.
    0.088924 loss
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1000)
    if pretrained:
        logger.info('Resnet18: Loading pretrained')
        _model_loaded = try_load_model('resnet18')
        model.load_state_dict(_model_loaded)
    if num_classes != 1000:
        model.reinit_fc(num_classes)

    layers = [model.fc, model.layer4, model.layer3]

    return model, layers


def resnet18_warmup(num_classes, pretrained=False):
    model, _ = resnet18(num_classes, pretrained)
    layers = [model.fc, (model.layer4, 0, 0.5), (model.layer3, 0, 0.5), (model.layer2, 0, 0.1), (model.layer1, 0, 0.1)]

    return model, layers


def resnet18_warmup_v2(num_classes, pretrained=False):
    model, _ = resnet18(num_classes, pretrained)
    layers = [model.fc, (model.layer4, 0, 1), (model.layer3, 0, 1), (model.layer2, 0, 0.1), (model.layer1, 0, 0.1)]

    return model, layers


def resnet18_conv_only(pretrained=True):
    "pretrained just for code reuse"
    model, _ = resnet18(NUM_CLASSES, pretrained=(not pretrained))
    if pretrained:
        logger.info('Loading resnet18 planet')
        model = torch.nn.DataParallel(model)  # due to saving parallel table
        checkpoint = torch.load(PRE_TRAINED_RESNET18)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.module
    model.fc = None

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
        return x

    setattr(model.__class__, 'forward', forward)
    return model


def resnet18_fc(num_classes, pretrained, embedding_len, bn, dp=False):
    "resnet18 + FC layer embedding"
    model, _ = resnet18(num_classes, pretrained=(not pretrained))
    if pretrained:
        logger.info('Loading resnet18 planet')
        model = torch.nn.DataParallel(model)  # due to saving parallel table
        checkpoint = torch.load(PRE_TRAINED_RESNET18)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.module

    model.fc1 = nn.Linear(512, embedding_len)
    model.relu_fc1 = nn.ReLU(inplace=True) #new
    model.fc = nn.Linear(embedding_len, num_classes)
    if bn:
        model.fc_bn = nn.BatchNorm1d(embedding_len)

    if dp:
        model.dp = nn.Dropout2d(p=0.5)

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
        x = self.relu_fc1(self.fc1(x))

        if getattr(self, 'fc_bn', None) is not None:
            x = self.fc_bn(x)
        if getattr(self, 'dp', None) is not None:
            x = self.dp(x)
        x = self.fc(x)

        return x

    setattr(model.__class__, 'forward', forward)

    layers = [model.fc1, model.fc, model.avgpool, model.layer4, model.layer3]

    return model, layers


def resnet18_fc128(num_classes=NUM_CLASSES, pretrained=True, bn=False):
    "0.08693 loss, LB: 0.92047"
    return resnet18_fc(num_classes, pretrained, embedding_len=128, bn=False)


def resnet18_fcbn_128(num_classes=NUM_CLASSES, pretrained=True):
    "0.08571 loss, LB: 0.92460"
    return resnet18_fc(num_classes, pretrained, embedding_len=128, bn=True)


def resnet18_fcbn_256(num_classes=NUM_CLASSES, pretrained=True):
    """0.08466 loss, LB: 0.92531
       0.08369 loss, LB: 0.92644 - another learning rate schedule,
       args to train: lr=0.7, b=128"""
    return resnet18_fc(num_classes, pretrained, embedding_len=256, bn=True)


def resnet18_fcbn_256_warmup(num_classes, pretrained=False):
    model, _ = resnet18_fc(num_classes, pretrained, embedding_len=256, bn=True)
    layers = [model.fc1, model.fc, (model.layer4, 0, 0.5), (model.layer3, 0, 0.5), ]
    return model, layers


def resnet18_fcbn_256_v2(num_classes=NUM_CLASSES, pretrained=True):
    ""
    model, _ = resnet18_fc(num_classes, pretrained, embedding_len=256, bn=True)
    layers = [(model.fc1, 1), (model.fc, 1), (model.layer4, 0.1), (model.layer3, 0.1),]

    return model, layers


def resnet18_fcbn_256_v3(num_classes=NUM_CLASSES, pretrained=True):
    ""
    model, _ = resnet18_fc(num_classes, pretrained, embedding_len=256, bn=True)
    layers = [(model.fc1, 1), (model.fc, 1), (model.layer4, 0.1), (model.layer3, 0.1),
              (model.layer2, 0.1), (model.layer1, 0.1)]

    return model, layers


def resnet18_fcbn_256_v4(num_classes=NUM_CLASSES, pretrained=True):
    ""
    model, _ = resnet18_fc(num_classes, pretrained, embedding_len=256, bn=True)
    layers = [model.fc1, model.fc_bn, model.fc, model.avgpool, model.layer4, model.layer3]

    return model, layers


def resnet18_fcbn_256_dp(num_classes=NUM_CLASSES, pretrained=True):
    ""
    return resnet18_fc(num_classes, pretrained, embedding_len=256, bn=True, dp=True)


def resnet18_fcbn_512(num_classes=NUM_CLASSES, pretrained=True):
    "0.08466 loss, LB: 0.92531"
    return resnet18_fc(num_classes, pretrained, embedding_len=512, bn=True)


def resnet34_warmup(num_classes, pretrained=False):
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    model.reinit_fc(num_classes)
    layers = [model.fc, (model.layer4, 0, 0.5), (model.layer3, 0, 0.5), (model.layer2, 0, 0.1), (model.layer1, 0, 0.1)]

    return model, layers


def resnet34_warmup_v2(num_classes, pretrained=False):
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    model.reinit_fc(num_classes)
    layers = [model.fc, (model.layer4, 0, 1), (model.layer3, 0, 1), (model.layer2, 0, 0.1), (model.layer1, 0, 0.1)]

    return model, layers


def resnet34_fcbn(num_classes=NUM_CLASSES, embedding_len=256, pretrained=True):
    """loss 0.083676, LB: 0.92645
    lr=0.7, 2 hooks 0.5 each, 2layers only"""
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))

    model.fc1 = nn.Linear(512, embedding_len)
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
        if x.size(2) != 1:
            x = F.avg_pool2d(x, x.size(2))
        x = x.view(x.size(0), -1)
        x = self.relu_fc1(self.fc1(x))
        x = self.fc_bn(x)
        x = self.fc(x)

        return x

    setattr(model.__class__, 'forward', forward)

    layers = [(model.fc1, 1), (model.fc, 1), (model.layer4, 1), (model.layer3, 1)]

    return model, layers


def densenet121_(num_classes=NUM_CLASSES, pretrained=True):
    """loss: 0.07939, LB: 0.92758, epoch=38 adaptive LR
    args to train: lr=0.1/0.01, b=80"""
    model = densenet121(pretrained=pretrained)
    model.classifier = nn.Linear(1024, num_classes)

    layers = [(model.classifier, 1), (model.features, 0.1)]

    return model, layers


def densenet121_frozen(num_classes=NUM_CLASSES, pretrained=True):
    "loss: 0.08230, lr=0.1, b=80"
    model = densenet121(pretrained=pretrained)
    model.classifier = nn.Linear(1024, num_classes)

    layers = [(model.classifier, 1), (model.features.denseblock3, 0.1), (model.features.denseblock4, 0.1)]

    return model, layers


def densenet169_(num_classes=NUM_CLASSES, pretrained=True):
    """loss: 0.07959, LB: 0.92793
    args to train: -l 0.1 -b 64 --lr_schedule adaptive, epoch=30th"""
    model = densenet169(pretrained=pretrained)
    model.classifier = nn.Linear(1664, num_classes)

    layers = [(model.classifier, 1), (model.features, 0.1)]

    return model, layers


def densenet201_(num_classes=NUM_CLASSES, pretrained=True):
    """
    lr=0.1, b=52
    """
    model = densenet201(pretrained=pretrained)
    model.classifier = nn.Linear(1920, num_classes)

    layers = [(model.classifier, 1), (model.features, 0.1)]

    return model, layers


def resnext_101(num_classes=NUM_CLASSES, pretrained=True):
    from legacy import resnext_pth
    model = resnext_pth.resnext_101_32x4d
    model.load_state_dict(torch.load(os.path.join(MODEL_FOLDER, 'resnext.pth')))
    model._modules['10']._modules['1'] = nn.Linear(2048, num_classes)

    layers = [(model._modules['10'], 1), (model._modules['6'], 0.1), (model._modules['7'], 0.1)]

    return model, None


tif_resnet18 = tif_model.tif_resnet18
tif_resnet18_fcbn = tif_model.tif_resnet18_fcbn
tif_index_resnet18 = tif_model.tif_index_resnet18


mix_net_v1 = curry(mix_model.mix_net_v1, resnet18_fcbn_256=resnet18_fcbn_256)
mix_net_v2 = curry(mix_model.mix_net_v2, resnet18_fcbn_256=resnet18_fcbn_256)
mix_net_v3 = curry(mix_model.mix_net_v3, resnet18_fcbn_256=resnet18_fcbn_256)
mix_net_v4 = curry(mix_model.mix_net_v4, resnet18_fcbn_256=resnet18_fcbn_256)
mix_net_v5 = curry(mix_model.mix_net_v5, resnet18_fcbn_256=resnet18_fcbn_256)
mix_net_v6 = curry(mix_model.mix_net_v6, resnet18_fcbn_256=resnet18_fcbn_256)
mix_net_v7 = curry(mix_model.mix_net_v7, resnet18_fcbn_256=resnet18_fcbn_256)
mix_wide_v1 = curry(mix_model.mix_wide_v1, resnet18_fcbn_256=resnet18_fcbn_256)
mix_wide_v2 = curry(mix_model.mix_wide_v2, resnet18_fcbn_256=resnet18_fcbn_256)
mix_wideselu_v1 = curry(mix_model.mix_wideselu_v1, resnet18_fcbn_256=resnet18_fcbn_256)
mix_netmr_v1 = curry(mix_model.mix_netmr_v1, resnet18_fcbn_256=resnet18_fcbn_256)
mix_nir_v1 = curry(mix_model.mix_nir_v1, resnet18_fcbn_256=resnet18_fcbn_256)
mixnet18_128 = curry(mix_model.mixnet18_128, resnet18_conv_only=resnet18_conv_only)
mixnet18_256 = curry(mix_model.mixnet18_256, resnet18_conv_only=resnet18_conv_only)
mixnet18_256_v2 = curry(mix_model.mixnet18_256_v2, resnet18_conv_only=resnet18_conv_only)


def main():
    model, _ = mix_net_v6()
    #model = ResNet(BasicBlock, [2,2,2,2], num_classes=17)

    print model
    #print dir(model.classifier)
    model = model.cuda()
    model.eval()

    x = torch.FloatTensor(torch.zeros((1, 6, 256, 256)))
    input = torch.autograd.Variable(x.cuda())
    res = model(input)
    print 'res', res.size()

if __name__ == '__main__':
    sys.exit(main())
