#-*- coding: utf8 -*-
import sys, random, math
import cv2
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import torch
from torchvision import transforms


def get_normalize(norm):
    norm = float(norm)
    _to_norm = lambda x: x/norm
    return transforms.Normalize(mean=map(_to_norm, [4988, 4270, 3074, 6398]),
                                 std=map(_to_norm, [399, 408, 453, 858]))


def get_normalize_plus(norm):
    "Reduntant, b/c Normalize omits absent dimensions. Done it to sleep well."
    norm = float(norm)
    _to_norm = lambda x: x/norm
    return transforms.Normalize(mean=map(_to_norm, [4988, 4270, 3074, 6398, 0, 0]),
                                 std=map(_to_norm, [399, 408, 453, 858, 1, 1]))


def get_normalize_mix():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406, 6398, 0, 0],
                                 std=[0.229, 0.224, 0.225, 858, 1, 1])


class TifToTensor(object):
    def __init__(self, norm=65536):
        self.norm = norm

    def __call__(self, pic):
        if not isinstance(pic, np.ndarray):
            raise ValueError, 'Type must be np.ndarray'

        img = torch.from_numpy(pic.transpose((2,0,1)).astype(np.float32)) # (B,G,R,NIR): [w,h,c] -> [c,w,h]

        if self.norm != 1:
            img = img.div(self.norm) # -> [0,1]

        return img


class MixToTensor(object):

    def __call__(self, pic):
        if not isinstance(pic, np.ndarray):
            raise ValueError, 'Type must be np.ndarray'

        img = torch.from_numpy(pic.transpose((2,0,1)).astype(np.float32)) # (R,G,B,NIR,NDWI,SAVI): [w,h,c] -> [c,w,h]

        img[0] = img[0].div(255) # R
        img[1] = img[1].div(255) # G
        img[2] = img[2].div(255) # B

        return img


def random_shift_scale_rotate(img, p=0.75, shift_limit=16, scale_limit=0.1, rotate_limit=45):
    if random.random() < p:
        height,width,channel = img.shape

        angle = random.uniform(-rotate_limit,rotate_limit)  # degree
        scale = random.uniform(1-scale_limit,1+scale_limit)

        dx = round(random.uniform(-shift_limit,shift_limit))  # pixel
        dy = round(random.uniform(-shift_limit,shift_limit))

        cc = math.cos(angle/180*math.pi) * scale
        ss = math.sin(angle/180*math.pi) * scale
        rotate_matrix = np.array([ [cc,-ss], [ss,cc] ])

        box0 = np.array([[0,0], [width,0], [width,height], [0,height]])
        box1 = box0 - np.array([width/2,height/2])
        box1 = np.dot(box1,rotate_matrix.T) + np.array([width/2+dx,height/2+dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)

        mat = cv2.getPerspectiveTransform(box0,box1)
        img = cv2.warpPerspective(img, mat, (width,height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)

    return img


def rotate(img, angle):
    if angle == 0 or angle == 360:
        return np.copy(img)

    "short version of the above random_shift_scale_rotate"
    height,width,channel = img.shape

    cc = math.cos(angle/180.0*math.pi)
    ss = math.sin(angle/180.0*math.pi)
    rotate_matrix = np.array([ [cc,-ss], [ss,cc] ])

    box0 = np.array([[0,0], [width,0], [width,height], [0,height]])
    box1 = box0 - np.array([width/2,height/2])
    box1 = np.dot(box1,rotate_matrix.T) + np.array([width/2,height/2])

    mat = cv2.getPerspectiveTransform(box0.astype(np.float32),box1.astype(np.float32))
    img = cv2.warpPerspective(img, mat, (width,height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)

    return img


def random_flip(img, p=0.5):
    if random.random() < p:
        img = cv2.flip(img,random.randint(-1,1))
    return img


def random_transpose(img, p=0.5):
    if random.random() < p:
        img = img.transpose(1,0,2)
        # img = cv2.transpose(img)
    return img


def elastic_transform(image, p=0.5, alpha=200, sigma=10, alpha_affine=1, random_state=None):
    "infeasible"
    if random.random() > p:
        return image

    if random_state is None:
        random_state = np.random.RandomState(None)

    if type(random_state) == int:
        random_state = np.random.RandomState(random_state)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def rotate_or_flip(img, number):
    if number == 0:
        return img
    elif number <= 3:
        return rotate(img, 90 * number)
    elif number == 4:
        return cv2.flip(img, 0)
    elif number == 5:
        return cv2.flip(img, 1)
    else:
        return random_shift_scale_rotate(img, 1.0)
#    elif number == 6:
#        return cv2.flip(img, -1)
#    elif number == 6:
#        return img.transpose(1,0,2)


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
    pass

if __name__ == '__main__':
    sys.exit(main())