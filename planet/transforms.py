#-*- coding: utf8 -*-
import sys, random, math
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Scale


class RandomVerticalFlip(object):
    """Randomly vertically flips the given PIL.Image with a probability of 0.5"""

    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class RandomRotate(object):

    def __call__(self, img):
        return img.rotate(90 * random.randint(0, 4))


def rotate_or_flip(img, number):
    if number <= 3:
        return img.rotate(90 * number)
    elif number == 4:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        return img.transpose(Image.FLIP_TOP_BOTTOM)


class RandomRotateOrFlip(object):

    def __call__(self, img):
        number = random.randint(0, 6)
        return rotate_or_flip(img, number)


class RandomSizedCropV2(object):
    """Random crop the given PIL.Image to a random size of (x1, x2) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, crop_size_percent=(0.08, 1.0), aspect_ratio_range=(3./4, 4./3),
                 interpolation=Image.BICUBIC):
        self.size = size
        self.crop_size_percent = crop_size_percent
        self.aspect_ratio_range = aspect_ratio_range or (1, 1)
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*self.crop_size_percent) * area
            aspect_ratio = random.uniform(*self.aspect_ratio_range)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))
                return img.resize((self.size, self.size), self.interpolation)

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(img))


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class ColorJitter(RandomOrder):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        transforms = []
        if brightness != 0:
            transforms.append(Brightness(brightness))
        if contrast != 0:
            transforms.append(Contrast(contrast))
        if saturation != 0:
            transforms.append(Saturation(saturation))

        RandomOrder.__init__(self, transforms)


class TenCrop(object):
    "Four corner patches and center crop from image and its horizontal reflection"

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size

        output = []
        for _img in (img, img.transpose(Image.FLIP_LEFT_RIGHT)):
            output.append(CenterCrop(self.size)(_img))
            output.append(_img.crop((0, 0, self.size, self.size)))
            output.append(_img.crop((w-self.size, 0, w, self.size)))
            output.append(_img.crop((0, h - self.size, self.size, h)))
            output.append(_img.crop((w-self.size, h-self.size, w, h)))

        return output


class TenCropPick(object):
    """Four corner patches and center crop from image and its horizontal reflection.
    Pick one of the crop specified in the constructor 
    """
    def __init__(self, size, index=0):
        self.size = size
        self.__index = index

    def __init_functions(self, w, h):
        funcs = []
        transp = lambda x: x.transpose(Image.FLIP_LEFT_RIGHT)
        funcs.append(lambda _img: CenterCrop(self.size)(_img))
        funcs.append(lambda _img: _img.crop((0, 0, self.size, self.size)))
        funcs.append(lambda _img: _img.crop((w - self.size, 0, w, self.size)))
        funcs.append(lambda _img: _img.crop((0, h - self.size, self.size, h)))
        funcs.append(lambda _img: _img.crop((w - self.size, h - self.size, w, h)))
        funcs.append(lambda _img: CenterCrop(self.size)(transp(_img)))
        funcs.append(lambda _img: transp(_img).crop((0, 0, self.size, self.size)))
        funcs.append(lambda _img: transp(_img).crop((w - self.size, 0, w, self.size)))
        funcs.append(lambda _img: transp(_img).crop((0, h - self.size, self.size, h)))
        funcs.append(lambda _img: transp(_img).crop((w - self.size, h - self.size, w, h)))
        return funcs

    def setter(self, value):
        if value < 0 or value > 9:
            raise ValueError, 'out of bounds [0,9]'
        self.__index = value

    index = property(fset=setter)

    def __call__(self, img):
        w, h = img.size
        func = self.__init_functions(w, h)[self.__index]

        return func(img)

    def __len__(self):
        return 10


class SpatialPick(object):
    def __init__(self, index=0):
        self.__index = index

    def setter(self, value):
        max_value = self.__len__() - 1
        if value < 0 or value > max_value:
            raise ValueError, 'out of bounds [0,%d]' % max_value
        self.__index = value

    def __len__(self):
        return 6

    index = property(fset=setter)

    def __call__(self, img):
        return rotate_or_flip(img, self.__index)


def main():
    import os
    from __init__ import TEST_FOLDER_JPG
    image_path = os.path.join(TEST_FOLDER_JPG, 'test_11553.jpg')
    from PIL import Image
    import cv2
    import PIL
    import numpy as np
    from torchvision.transforms import ToTensor, ToPILImage

    source = Image.open(image_path)
    #print source, image_path
    for _ in xrange(100):
        new_x =  RandomSizedCropV2(224)(source)
    #new_x.save('/home/tyantov/test.jpg')

#    res = TenCrop(224)(source)
#    for i, img in enumerate(res):
#        img.save('/home/tyantov/test_%d.jpg' % i)

    # x = ToTensor()(source)
    # print source.getpixel((1, 1))
    # print x[1]
    # y = x#ColorJitter()(x)
    # z = ToPILImage()(y)
    # print z
    # #z.save('/home/tyantov/test.jpg')
    # print z.getpixel((1, 1))
    # print np.mean(source), np.mean(z)
    # import copy
    # x = TenCropPick(224)
    # for i in xrange(0,10):
    #     x.index = i
    #     img = x(copy.deepcopy(source))
    #     img.save('/home/tyantov/test2_%d.jpg' % i)


if __name__ == '__main__':
    sys.exit(main())