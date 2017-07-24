#-*- coding: utf8 -*-
import os, sys
from collections import defaultdict
import pandas as pd
import numpy as np
from skimage import io
import cv2
import torch
import torch.utils.data as data
from torchvision.datasets.folder import default_loader

# Indexes constants
L_SAVI, L_EVI, G, C1, C2 = 0.5, 1, 2.5, 6, 7.5
f_ndwi = lambda green, nir: (green - nir) / (green + nir)  # (GREEN - NIR) / (GREEN + NIR)
f_savi = lambda nir, red: (1 + L_SAVI) * (nir - red) / (nir + red + L_SAVI)  # (1+L)(NIR - R)/(NIR + R + L)
f_evi = lambda nir, red, blue: G * (nir - red) / (nir + C1 * red - C2 * blue + L_EVI)


def tif_loader(path):
    return io.imread(path)


def jpg_cv2_loader(path):
    return cv2.imread(path,1)


def tif_index_loader(path):
    "Load 4-band + compute ndwi, savi, evi(off)"
    img = io.imread(path).astype(np.float32)
    green = img[:, :, 1]
    nir = img[:, :, 3]
    red = img[:, :, 2]
    #blue = img[:, :, 0]

    ndwi = f_ndwi(green, nir)
    savi = f_savi(nir, red)
    #evi = f_evi(nir, red, blue)

    shp = (img.shape[0], img.shape[1], 1)
    new_img = np.append(img, ndwi.reshape(*shp), 2)
    new_img = np.append(new_img, savi.reshape(*shp), 2)
    #new_img = np.append(new_img, evi.reshape(*shp), 2)

    return new_img


def mix_loader(tif_path):
    "Loads tif and jpg. Returns RGB of jpg + NIR,NDWI,SAVI of tif as one numpy array"
    tif_img = io.imread(tif_path).astype(np.float32)
    green = tif_img[:, :, 1]
    nir = tif_img[:, :, 3]
    red = tif_img[:, :, 2]
    ndwi = f_ndwi(green, nir)
    savi = f_savi(nir, red)

    jpg_path = tif_path.replace('tif', 'jpg').replace('-v2', '')  # TODO: cleaner
    jpg_img = np.array(default_loader(jpg_path))

    shp = (jpg_img.shape[0], jpg_img.shape[1], 1)
    new_img = np.append(jpg_img, nir.reshape(*shp), 2)
    new_img = np.append(new_img, ndwi.reshape(*shp), 2)
    new_img = np.append(new_img, savi.reshape(*shp), 2)

    return new_img


def jpg_nir_loader(tif_path):
    tif_img = io.imread(tif_path).astype(np.float32)
    nir = tif_img[:, :, 3]
    jpg_path = tif_path.replace('tif', 'jpg').replace('-v2', '')  # TODO: cleaner
    jpg_img = np.array(default_loader(jpg_path))

    shp = (jpg_img.shape[0], jpg_img.shape[1], 1)
    new_img = np.append(jpg_img, nir.reshape(*shp), 2)

    return new_img


class ImageFolder(data.Dataset):
    def __init__(self, label_file, root, permitted_filenames=None, transform=None, target_transform=None,
                 loader=default_loader):
        self.root = root
        labels, classes, class_to_idx, class_freq = self._init_classes(label_file)
        self.labels = labels
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.class_freq = class_freq
        self.imgs = self.make_dataset(permitted_filenames)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.weights = self._create_weights(self.imgs, class_freq, class_to_idx)

    def _init_classes(self, label_file):
        df = pd.read_csv(label_file)
        labels = {k: v.split() for k,v in df.values}

        class_freq = defaultdict(int)
        for tags in labels.values():
            for tag in tags:
                class_freq[tag] += 1

        classes = list(class_freq.keys())
        classes.sort()
        class_to_idx = {classes[i]: i for i in xrange(len(classes))}

        return labels, classes, class_to_idx, class_freq

    def _create_weights(self, data, class_freq, class_to_idx):
        xsum = float(sum(class_freq.values()))
        id_freq = {class_to_idx[k]: (1 - v/xsum) for k,v in class_freq.iteritems()}
        freq = [v for _, v in sorted(id_freq.iteritems())]
        freq = torch.FloatTensor(freq)

        weights = []
        for _, target in data:
            target_weights = target * freq
            weight = torch.min(target_weights.masked_select(target_weights>0))
            weights.append(weight)

        return weights

    def make_dataset(self, permitted_filenames):
        images = []
        for filename in os.listdir(self.root):
            if permitted_filenames is None or filename in permitted_filenames:
                labels = set(self.labels[filename.split('.')[0]])
                target = [int(class_name in labels) for class_name in self.classes]
                images.append((filename, torch.FloatTensor(target)))

        return images

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(os.path.join(self.root, path))

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class ImageTestFolder(data.Dataset):
    def __init__(self, root, transform=None, loader=default_loader):
        imgs = os.listdir(root)
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(os.path.join(self.root, path))
        if self.transform is not None:
            img = self.transform(img)

        return img, path

    def __len__(self):
        return len(self.imgs)


def main():
    import numpy as np
    from __init__ import TRAIN_FOLDER_JPG, LABEL_FILE, TRAIN_FOLDER_TIF, TEST_FOLDER_TIF
    #itf = ImageFolder(LABEL_FILE, TRAIN_FOLDER_TIF, loader=tif_loader)
    itf = ImageTestFolder(TEST_FOLDER_TIF, loader=mix_loader)
    print len(itf)
    for i, (img, _) in enumerate(itf):
        print img.shape
        print img[-1,-1,-1]
        print img[0, 0:10, 0]
        sys.exit()

    #print itf.classes
    #print itf.class_freq



if __name__ == '__main__':
    sys.exit(main())