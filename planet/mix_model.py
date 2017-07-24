#-*- coding: utf8 -*-
import sys, logging
import torch, torch.nn as nn
from torchvision.models.resnet import BasicBlock
import torch.utils.model_zoo as model_zoo
from __init__ import NUM_CLASSES, PRE_TRAINED_RESNET18_FCBN_256, PRE_TRAINED_RESNET18
from tif_model import ResNet
from generic_models.wide_resnet import WideResNet
from generic_models.wide_selu import WideSeluResNet
import torch.nn.functional as F


logger = logging.getLogger('app')
logger.setLevel(logging.DEBUG)


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}


class MixNet(nn.Module):
    def __init__(self, resnet18_fcbn_256, tif_layers=[2, 2, 1, 1], num_classes=NUM_CLASSES, pretrained=True,
                 tif_embedding_len=128, load_tif=True, tif_pretrained=False, iplane=32):
        super(MixNet, self).__init__()
        self.__tif_layers = tif_layers
        self.__iplane = iplane
        self.jpg_model, _ = resnet18_fcbn_256(num_classes, pretrained=False) # resnet18 planet loads here
        self.jpg_model = self.__load_pretrained_jpg(self.jpg_model, pretrained)

        if load_tif:
            self.tif_model = self._load_tif_model(tif_embedding_len, tif_pretrained)
        self.fc_final = nn.Linear(256 + tif_embedding_len, num_classes)

    def _load_tif_model(self, embedding_len, tif_pretrained):
        tif_model = ResNet(3, BasicBlock, self.__tif_layers, num_classes=embedding_len, iplane=self.__iplane)
        if tif_pretrained:
            state_dict = model_zoo.load_url(model_urls['resnet18'])
            fc_w, fc_b = list(tif_model.fc.parameters())
            state_dict['fc.weight'] = fc_w
            state_dict['fc.bias'] = fc_b
            tif_model.load_state_dict(state_dict)

        tif_model.relu_fc1 = nn.ReLU(inplace=True)
        tif_model.fc_bn = nn.BatchNorm1d(embedding_len)

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
            x = self.relu_fc1(x)
            x = self.fc_bn(x)

            return x

        setattr(tif_model.__class__, 'forward', forward)

        return tif_model

    def __load_pretrained_jpg(self, model, pretrained):
        if pretrained:
            model = torch.nn.DataParallel(model)  # due to saving parallel table
            checkpoint = torch.load(PRE_TRAINED_RESNET18_FCBN_256)
            logger.info('Loading from %s', PRE_TRAINED_RESNET18_FCBN_256)
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
            x = self.relu_fc1(self.fc1(x))
            x = self.fc_bn(x)

            return x

        setattr(model.__class__, 'forward', forward)

        return model

    def forward(self, x):
        jpg_x = x[:,0:3,:,:]
        tif_x = x[:,3:6,:,:]

        jpg_out = self.jpg_model(jpg_x)
        tif_out = self.tif_model(tif_x)

        out = torch.cat([jpg_out, tif_out],1)
        out = self.fc_final(out)

        return out


class MixWideNet(MixNet):
    def __init__(self, resnet18_fcbn_256, widen_factor=2, tif_layers=[2, 2, 2, 2], num_classes=NUM_CLASSES,
                 pretrained=True, tif_embedding_len=128):
        MixNet.__init__(self, resnet18_fcbn_256, tif_layers, num_classes, pretrained, tif_embedding_len, load_tif=False)
        self.tif_model = self._load_tif_model(widen_factor, embedding_len=tif_embedding_len)

    def _load_tif_model(self, widen_factor, embedding_len):
        tif_model = WideResNet(widen_factor, embedding_len, num_channel=32, drop_rate=0.5)
        tif_model.relu_fc1 = nn.ReLU(inplace=True)
        tif_model.fc_bn = nn.BatchNorm1d(embedding_len)

        def forward(self, x):
            out = self.relu0(self.bn0(self.conv0(x)))
            out = self.maxpool(out)

            out = self.block1(out)
            out = self.block2(out)
            out = self.block3(out)
            out = self.block4(out)

            out = self.relu(self.bn1(out))
            out = self.avgpool(out)
            if out.size(2) != 1:
                out = F.avg_pool2d(out, out.size(2))
            out = out.view(out.size(0), -1)
            out = self.fc(out)

            out = self.relu_fc1(out)
            out = self.fc_bn(out)

            return out

        setattr(tif_model.__class__, 'forward', forward)

        return tif_model


class MixWideSeluNet(MixNet):
    def __init__(self, resnet18_fcbn_256, widen_factor=2, tif_layers=[2, 2, 2, 2], num_classes=NUM_CLASSES,
                 pretrained=True, tif_embedding_len=128):
        MixNet.__init__(self, resnet18_fcbn_256, tif_layers, num_classes, pretrained, tif_embedding_len, load_tif=False)
        self.tif_model = self._load_tif_model(widen_factor, embedding_len=tif_embedding_len)

    def _load_tif_model(self, widen_factor, embedding_len):
        tif_model = WideSeluResNet(widen_factor, embedding_len, num_channel=32, drop_rate=0.5)
        tif_model.relu_fc1 = nn.ReLU(inplace=True)
        tif_model.fc_bn = nn.BatchNorm1d(embedding_len)

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

            out = self.relu_fc1(out)
            out = self.fc_bn(out)

            return out

        setattr(tif_model.__class__, 'forward', forward)

        return tif_model


class MixNet18(nn.Module):
    def __init__(self, resnet18_conv_only, num_classes=NUM_CLASSES, pretrained=True,
                 embedding_len=128):
        super(MixNet18, self).__init__()
        self.jpg_model = resnet18_conv_only(pretrained=pretrained)
        self.tif_model = resnet18_conv_only(pretrained=False)

        self.embedding = nn.Sequential(nn.Linear(1024, embedding_len),
                                        nn.BatchNorm1d(embedding_len),
                                        nn.ReLU(inplace=True))
        self.classifier = nn.Linear(embedding_len, num_classes)

    def forward(self, x):
        jpg_x = x[:,0:3,:,:]
        tif_x = x[:,3:6,:,:]

        jpg_out = self.jpg_model(jpg_x)
        tif_out = self.tif_model(tif_x)

        out = torch.cat([jpg_out, tif_out],1)
        out = self.embedding(out)
        out = self.classifier(out)

        return out


class MixNetMirror(nn.Module):
    def __init__(self, resnet18_fcbn_256, num_classes=NUM_CLASSES, pretrained=True):
        super(MixNetMirror, self).__init__()
        self.jpg_model, _ = resnet18_fcbn_256(num_classes, pretrained=False) # resnet18 planet loads here
        self.jpg_model = self.__load_pretrained(self.jpg_model, pretrained)
        self.tif_model, _ = resnet18_fcbn_256(num_classes, pretrained=False)  # resnet18 planet loads here
        self.tif_model = self.__load_pretrained(self.tif_model, pretrained)
        self.fc_final = nn.Linear(256*2, num_classes)

    def __load_pretrained(self, model, pretrained):
        if pretrained:
            model = torch.nn.DataParallel(model)  # due to saving parallel table
            checkpoint = torch.load(PRE_TRAINED_RESNET18_FCBN_256)
            logger.info('Loading from %s', PRE_TRAINED_RESNET18_FCBN_256)
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
            x = self.relu_fc1(self.fc1(x))
            x = self.fc_bn(x)

            return x

        setattr(model.__class__, 'forward', forward)

        return model

    def forward(self, x):
        jpg_x = x[:,0:3,:,:]
        tif_x = x[:,3:6,:,:]

        jpg_out = self.jpg_model(jpg_x)
        tif_out = self.tif_model(tif_x)

        out = torch.cat([jpg_out, tif_out],1)
        out = self.fc_final(out)

        return out


class MixNirNet(nn.Module):
    def __init__(self, resnet18_fcbn_256, num_classes=NUM_CLASSES, pretrained=True):
        super(MixNirNet, self).__init__()
        self.jpg_model, _ = resnet18_fcbn_256(num_classes, pretrained=False) # resnet18 planet loads here
        self.jpg_model = self.__load_pretrained_jpg(self.jpg_model, pretrained)

        self.nir_model = self._load_nir_model()
        self.fc_final = nn.Linear(256*2, num_classes)

    def _load_nir_model(self, embedding_len=256):
        nir_model = ResNet(1, BasicBlock, [2,2,2,2], iplane=64, num_classes=embedding_len)
        # Load resnet18
        state_dict = model_zoo.load_url(model_urls['resnet18'])
        # Patch FC
        fc_w, fc_b = list(nir_model.fc.parameters())
        state_dict['fc.weight'] = fc_w
        state_dict['fc.bias'] = fc_b
        # Patch first conv layer
        state_dict['conv1.weight'] = state_dict['conv1.weight'].data[:, :1, :, :]
        nir_model.load_state_dict(state_dict)

        nir_model.relu_fc1 = nn.ReLU(inplace=True)
        nir_model.fc_bn = nn.BatchNorm1d(embedding_len)

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
            x = self.relu_fc1(x)
            x = self.fc_bn(x)

            return x

        setattr(nir_model.__class__, 'forward', forward)

        return nir_model

    def __load_pretrained_jpg(self, model, pretrained):
        if pretrained:
            model = torch.nn.DataParallel(model)  # due to saving parallel table
            checkpoint = torch.load(PRE_TRAINED_RESNET18_FCBN_256)
            logger.info('Loading from %s', PRE_TRAINED_RESNET18_FCBN_256)
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
            x = self.relu_fc1(self.fc1(x))
            x = self.fc_bn(x)

            return x

        setattr(model.__class__, 'forward', forward)

        return model

    def forward(self, x):
        jpg_x = x[:,0:3,:,:]
        nir_x = x[:,3:,:,:]

        jpg_out = self.jpg_model(jpg_x)
        nir_out = self.nir_model(nir_x)

        out = torch.cat([jpg_out, nir_out],1)
        out = self.fc_final(out)

        return out


def mix_nir_v1(resnet18_fcbn_256, num_classes=NUM_CLASSES, pretrained=True):
    model = MixNirNet(resnet18_fcbn_256, num_classes, pretrained)

    tm = model.nir_model
    layers = [(tm.fc, 1), (tm.layer4, 1), (tm.layer3, 1), (tm.layer2, 0.1), (tm.layer1, 0.1), (tm.bn1, 0.1),
              (model.jpg_model, 0.05), (model.fc_final, 1)]
    return model, layers


def mix_netmr_v1(resnet18_fcbn_256, num_classes=NUM_CLASSES, pretrained=True):
    model = MixNetMirror(resnet18_fcbn_256, num_classes, pretrained=True)
    tm = model.tif_model
    layers = [(tm.fc1, 1), (tm.fc_bn, 1), (tm.layer4, 1), (tm.layer3, 1), (tm.layer2, 0.1), (tm.layer1, 0.1),
              (model.jpg_model, 0.05), (model.fc_final, 1)]
    return model, layers


def mixnet18_128(resnet18_conv_only, num_classes=NUM_CLASSES, pretrained=True):
    model = MixNet18(resnet18_conv_only, num_classes, pretrained, embedding_len=128)
    tm = model.tif_model
    layers = [ (tm.layer4, 0, 1), (tm.layer3, 0, 1), (tm.layer2, 0, 0.1), (tm.layer1, 0, 0.1), (tm.conv1, 0, 0.1),
               (tm.bn1, 0, 0.1), (model.jpg_model, 0, 0.05), (model.embedding, 1), (model.classifier, 1)]

    return model, layers


def mixnet18_256(resnet18_conv_only, num_classes=NUM_CLASSES, pretrained=True):
    model = MixNet18(resnet18_conv_only, num_classes, pretrained, embedding_len=256)
    tm = model.tif_model
    layers = [(tm.layer4, 0, 1), (tm.layer3, 0, 1), (tm.layer2, 0, 0.1), (tm.layer1, 0, 0.1), (tm.conv1, 0, 0.1),
              (tm.bn1, 0, 0.1), (model.jpg_model, 0, 0.05), (model.embedding, 1), (model.classifier, 1)]

    return model, layers


def mixnet18_256_v2(resnet18_conv_only, num_classes=NUM_CLASSES, pretrained=True):
    "no better 92.9"
    model = MixNet18(resnet18_conv_only, num_classes, pretrained, embedding_len=256)
    tm = model.tif_model
    layers = [(tm.layer4, 0, 1), (tm.layer3, 0, 1), (tm.layer2, 0, 1), (tm.layer1, 0, 1), (tm.conv1, 0, 1),
              (tm.bn1, 0, 1), (model.jpg_model, 0, 1), (model.embedding, 1), (model.classifier, 1)]

    return model, layers


def mix_net_v1(resnet18_fcbn_256, num_classes=NUM_CLASSES, pretrained=True):
    "0.899319|0.928656, LB: 0.92841"
    model = MixNet(resnet18_fcbn_256, [2, 2, 1, 1], num_classes, pretrained)
    layers = [(model.tif_model, 1), (model.jpg_model.fc_bn, 0.1), (model.jpg_model.fc1, 0.1), (model.fc_final, 1)]
    return model, layers


def mix_net_v2(resnet18_fcbn_256, num_classes=NUM_CLASSES, pretrained=True):
    "no better, 0.08475"
    model = MixNet(resnet18_fcbn_256, [1, 1, 1, 1], num_classes, pretrained, tif_embedding_len=256)
    layers = [(model.tif_model, 1), (model.jpg_model.fc_bn, 0.1), (model.jpg_model.fc1, 0.1), (model.fc_final, 1)]
    return model, layers


def mix_net_v3(resnet18_fcbn_256, num_classes=NUM_CLASSES, pretrained=True):
    "loss=0.08269, F2=0.928917, LB: 0.92964"
    model = MixNet(resnet18_fcbn_256, [2, 2, 1, 1], num_classes, pretrained, tif_embedding_len=256)
    layers = [(model.tif_model, 1), (model.jpg_model, 0.05), (model.fc_final, 1)]
    return model, layers


def mix_net_v4(resnet18_fcbn_256, num_classes=NUM_CLASSES, pretrained=True):
    "no better than v3"
    model = MixNet(resnet18_fcbn_256, [2, 2, 2, 2], num_classes, pretrained, tif_embedding_len=256)
    layers = [(model.tif_model, 1), (model.jpg_model, 0.05), (model.fc_final, 1)]
    return model, layers


def mix_net_v5(resnet18_fcbn_256, num_classes=NUM_CLASSES, pretrained=True):
    "loss=0.08233, F2=0.929610 (on threshold=0.5 - 0.905239): LB: 0.93024"
    model = MixNet(resnet18_fcbn_256, [2, 2, 2, 2], num_classes, pretrained, tif_embedding_len=256, tif_pretrained=True,
                   iplane=64)

    tm = model.tif_model
    layers = [(tm.fc, 1), (tm.layer4, 0.5), (tm.layer3, 0.5), (tm.layer2, 0.1), (tm.layer1, 0.1),
              (model.jpg_model, 0.05), (model.fc_final, 1)]
    return model, layers


def mix_net_v6(resnet18_fcbn_256, num_classes=NUM_CLASSES, pretrained=True):
    "0.08205|0.930173(0.906202): LB: 0.93046"
    model = MixNet(resnet18_fcbn_256, [2, 2, 2, 2], num_classes, pretrained, tif_embedding_len=256, tif_pretrained=True,
                   iplane=64)

    tm = model.tif_model
    layers = [(tm.fc, 1), (tm.layer4, 1), (tm.layer3, 1), (tm.layer2, 0.1), (tm.layer1, 0.1),
              (model.jpg_model, 0.05), (model.fc_final, 1)]
    return model, layers


def mix_net_v7(resnet18_fcbn_256, num_classes=NUM_CLASSES, pretrained=True):
    "To run"
    model = MixNet(resnet18_fcbn_256, [2, 2, 2, 2], num_classes, pretrained, tif_embedding_len=256, tif_pretrained=True,
                   iplane=64)

    tm = model.tif_model
    layers = [(tm.fc, 1), (tm.layer4, 0, 0.5), (tm.layer3, 0, 0.5), (tm.layer2, 0, 0.5), (tm.layer1, 0, 0.5),
              (model.jpg_model, 0, 0.1), (model.fc_final, 1)]
    return model, layers


def mix_wide_v1(resnet18_fcbn_256, num_classes=NUM_CLASSES, pretrained=False):
    "loss: 0.08258, F2: 0.929308 (thr=0.5 - 0.903947), LB: 0.93002"
    model = MixWideNet(resnet18_fcbn_256, 2, [2, 2, 2, 2], num_classes, pretrained, tif_embedding_len=256)
    layers = [(model.tif_model, 1), (model.jpg_model, 0.05), (model.fc_final, 1)]
    return model, layers


def mix_wide_v2(resnet18_fcbn_256, num_classes=NUM_CLASSES, pretrained=True):
    model = MixWideNet(resnet18_fcbn_256, 5, [2, 2, 1, 1], num_classes, pretrained, tif_embedding_len=128)
    layers = [(model.tif_model, 1), (model.jpg_model, 0.05), (model.fc_final, 1)]
    return model, layers


def mix_wideselu_v1(resnet18_fcbn_256, num_classes=NUM_CLASSES, pretrained=True):
    model = MixWideSeluNet(resnet18_fcbn_256, 2, [2, 2, 2, 2], num_classes, pretrained, tif_embedding_len=256)
    layers = [(model.tif_model, 1), (model.jpg_model, 0.05), (model.fc_final, 1)]
    return model, layers


def main():
    from transform_rules import nozoom_256, mix_index_nozoom_256
    import os
    from torchvision.transforms import ToPILImage
    from __init__ import TEST_FOLDER_JPG, TEST_FOLDER_TIF
    from folder import default_loader, mix_loader ,jpg_nir_loader
    from model import resnet18_fcbn_256, resnet18_conv_only

    fname = 'test_11556'
    image_path = os.path.join(TEST_FOLDER_JPG, '%s.jpg' % fname)
    tif_path = os.path.join(TEST_FOLDER_TIF, '%s.tif' % fname) #test_11556
    source_jpg = default_loader(image_path)
    source_mix = jpg_nir_loader(tif_path)
    tr_jpg = nozoom_256()['val']
    tf_mix = mix_index_nozoom_256()['val']

    res_jpg = tr_jpg(source_jpg)
    res_mix = tf_mix(source_mix)

    #print res_jpg
    #print res_mix

    print res_jpg.size(), res_mix.size()

    #model, _ = mix_net_v5(resnet18_fcbn_256, pretrained=True)
    #model = MixNet18(resnet18_conv_only)
    model, _ = mix_nir_v1(resnet18_fcbn_256)
    #model = WideResNet(2, 1000, [3,4,6,3], num_channel=64)
    #model = WideSeluResNet(2, 17, [2,2,2,2], drop_rate=0.0)

    print model
    model = model.cuda()
    model.eval()

    x = torch.FloatTensor(torch.zeros((64, 4, 256, 256)))
    #x = res_mix.unsqueeze(0)

    input = torch.autograd.Variable(x.cuda())
    res = model(input)
    print res.size(), 'res'
    #print list(enumerate(torch.sigmoid(res).data.cpu().numpy()[0]))

if __name__ == '__main__':
    sys.exit(main())