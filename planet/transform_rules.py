#-*- coding: utf8 -*-
import random, sys, cv2
from PIL import Image
import torchvision.transforms as transforms
from transforms import ColorJitter, TenCropPick, RandomRotateOrFlip, SpatialPick, RandomSizedCropV2
import tif_transforms as tif


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def imagenet_like():
    train_transformations = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        lambda img: img if random.random() < 0.5 else img.transpose(Image.FLIP_TOP_BOTTOM),
        transforms.ToTensor(),
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        normalize,
    ])

    val_transformations = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_transformation = transforms.Compose([
        TenCropPick(224),
        transforms.ToTensor(),
        normalize,
    ])

    return {'train': train_transformations, 'val': val_transformations, 'test': test_transformation}


def nozoom_256():
    train_transformations = transforms.Compose([
        RandomRotateOrFlip(),
        transforms.ToTensor(),
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        normalize,
    ])

    val_transformations = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    test_transformation = transforms.Compose([
        SpatialPick(),
        transforms.ToTensor(),
        normalize,
    ])

    return {'train': train_transformations, 'val': val_transformations, 'test': test_transformation}


def low_zoom_224():
    train_transformations = transforms.Compose([
        RandomSizedCropV2(224, crop_size_percent=(0.5, 1.0), aspect_ratio_range=(3. / 4, 4. / 3)),
        RandomRotateOrFlip(),
        transforms.ToTensor(),
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        normalize,
    ])

    val_transformations = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_transformation = transforms.Compose([
        TenCropPick(224),
        transforms.ToTensor(),
        normalize,
    ])

    return {'train': train_transformations, 'val': val_transformations, 'test': test_transformation}


def zoom_256():
    train_transformations = transforms.Compose([
        RandomSizedCropV2(256, crop_size_percent=(0.8, 1.0), aspect_ratio_range=(9. / 10, 10. / 9)),
        RandomRotateOrFlip(),
        transforms.ToTensor(),
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        normalize,
    ])

    val_transformations = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    test_transformation = transforms.Compose([
        SpatialPick(),
        transforms.ToTensor(),
        normalize,
    ])

    return {'train': train_transformations, 'val': val_transformations, 'test': test_transformation}


def tif_nozoom_256():
    norm = 1#41486

    train_transformations = transforms.Compose([
        tif.random_shift_scale_rotate,
        tif.random_flip,
        tif.random_transpose,
        tif.TifToTensor(norm=norm),
        tif.get_normalize(norm=norm),
    ])

    val_transformations = transforms.Compose([
        tif.TifToTensor(norm=norm),
        tif.get_normalize(norm=norm),
    ])

    test_transformation = transforms.Compose([
        tif.SpatialPick(),
        tif.TifToTensor(norm=norm),
        tif.get_normalize(norm=norm),
    ])

    return {'train': train_transformations, 'val': val_transformations, 'test': test_transformation}


def tif_index_nozoom_256():
    norm = 1

    train_transformations = transforms.Compose([
        tif.random_shift_scale_rotate,
        tif.random_flip,
        tif.random_transpose,
        tif.TifToTensor(norm=norm),
        tif.get_normalize_plus(norm=norm),
    ])

    val_transformations = transforms.Compose([
        tif.TifToTensor(norm=norm),
        tif.get_normalize_plus(norm=norm),
    ])

    test_transformation = transforms.Compose([
        tif.SpatialPick(),
        tif.TifToTensor(norm=norm),
        tif.get_normalize_plus(norm=norm),
    ])

    return {'train': train_transformations, 'val': val_transformations, 'test': test_transformation}


def mix_index_nozoom_256():

    train_transformations = transforms.Compose([
        tif.random_shift_scale_rotate,
        tif.random_flip,
        tif.random_transpose,
        tif.MixToTensor(),
        tif.get_normalize_mix(),
    ])

    val_transformations = transforms.Compose([
        tif.MixToTensor(),
        tif.get_normalize_mix(),
    ])

    test_transformation = transforms.Compose([
        tif.SpatialPick(),
        tif.MixToTensor(),
        tif.get_normalize_mix(),
    ])

    return {'train': train_transformations, 'val': val_transformations, 'test': test_transformation}


def mix_224():
    test_img_size = 224

    train_transformations = transforms.Compose([
        lambda x: cv2.resize(x, (test_img_size, test_img_size), interpolation=cv2.INTER_CUBIC),  # resize
        tif.random_shift_scale_rotate,
        tif.random_flip,
        tif.random_transpose,
        tif.MixToTensor(),
        tif.get_normalize_mix(),
    ])

    val_transformations = transforms.Compose([
        lambda x: cv2.resize(x, (test_img_size, test_img_size), interpolation=cv2.INTER_CUBIC),  # resize
        tif.MixToTensor(),
        tif.get_normalize_mix(),
    ])

    test_transformation = transforms.Compose([
        tif.SpatialPick(),
        lambda x: cv2.resize(x, (test_img_size, test_img_size), interpolation=cv2.INTER_CUBIC), #resize
        tif.MixToTensor(),
        tif.get_normalize_mix(),
    ])

    return {'train': train_transformations, 'val': val_transformations, 'test': test_transformation}


def aae_jpg_256():

    train_transformations = transforms.Compose([
        tif.random_shift_scale_rotate,
        tif.random_flip,
        tif.random_transpose,
        tif.MixToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    return {'train': train_transformations}


def split_brain_256():
    train_transformations = transforms.Compose([
        lambda x: cv2.resize(x, (128, 128), interpolation=cv2.INTER_CUBIC),  # resize
        tif.random_shift_scale_rotate,
        tif.random_flip,
        tif.random_transpose,
        tif.MixToTensor(),
        transforms.Normalize((0.0, 0.0, 0.0, 0.0, -1, -1), (1, 1, 1, 41486.0, 2.2, 2.2)), #NIR only TODO: calc on data
    ])

    return {'train': train_transformations}


def main():
    import os
    from torchvision.transforms import ToPILImage
    from __init__ import TEST_FOLDER_JPG, TEST_FOLDER_TIF
    from folder import default_loader, mix_loader
    image_path = os.path.join(TEST_FOLDER_JPG, 'test_11556.jpg')
    tif_path = os.path.join(TEST_FOLDER_TIF, 'test_11556.tif')
    source_jpg = default_loader(image_path)
    source_mix = mix_loader(tif_path)
    tr_jpg = nozoom_256()['val']
    tf_mix = mix_index_nozoom_256()['test']

    res_jpg = tr_jpg(source_jpg)
    tf_mix.transforms[0].index = 7
    res_mix = tf_mix(source_mix)


    #print res_jpg
    print res_mix
    print res_mix.size()

    #print res_jpg.size(), res_mix.size()

if __name__ == '__main__':
    sys.exit(main())