#-*- coding: utf8 -*-
import os
import logging


__all__ = ['TRAIN_FOLDER_JPG', 'TRAIN_FOLDER_TIF', 'TEST_FOLDER_JPG', 'TEST_FOLDER_TIF' ,'LABEL_FILE', 'PACKAGE_DIR',
           'MODEL_FOLDER', 'PRE_TRAINED_RESNET18', 'NUM_CLASSES', 'data_folder', 'AEE_RESULT_DIR']


PACKAGE_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), '')
RESULT_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, '../', 'results{0}'))
AEE_RESULT_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, '../', 'aae_results'))
data_folder = os.path.realpath(os.path.join(PACKAGE_DIR, '../data/'))
TRAIN_FOLDER_JPG = os.path.join(data_folder, 'train-jpg')
TRAIN_FOLDER_TIF = os.path.join(data_folder, 'train-tif-v2')
TEST_FOLDER_JPG = os.path.join(data_folder, 'test-jpg')
TEST_FOLDER_TIF = os.path.join(data_folder, 'test-tif-v2')
MODEL_FOLDER = os.path.join(PACKAGE_DIR, 'pretrained')
PRE_TRAINED_RESNET18 = os.path.join(MODEL_FOLDER, 'resnet18_planet.pth.tar')
PRE_TRAINED_TIF_RESNET18 = os.path.join(MODEL_FOLDER, 'tif_resnet18.pth.tar')
PRE_TRAINED_RESNET18_FCBN_256 = os.path.join(MODEL_FOLDER, 'resnet18_fcbn_256.pth.tar')
SPLIT_FILE = os.path.join(data_folder, 'split.txt')
HOLDOUT_FILE = os.path.join(data_folder, 'holdout.txt')
BLACKLIST_FILE = os.path.join(data_folder, 'blacklist_v2.txt')
NUM_CLASSES = 17

LABEL_FILE = os.path.join(data_folder, 'train.csv')
#LABEL_FILE = os.path.join(data_folder, 'train_v2.csv')
#LABEL_FILE = os.path.join(data_folder, 'train_v3.csv')

assert os.path.exists(data_folder)
assert os.path.exists(TRAIN_FOLDER_JPG)
assert os.path.exists(TRAIN_FOLDER_TIF)
assert os.path.exists(TEST_FOLDER_JPG)
assert os.path.exists(TEST_FOLDER_TIF)
assert os.path.exists(MODEL_FOLDER)
assert os.path.exists(PRE_TRAINED_RESNET18)

if not os.path.exists(RESULT_DIR.format('')):
    os.mkdir(RESULT_DIR.format(''))

if not os.path.exists(AEE_RESULT_DIR):
    os.mkdir(AEE_RESULT_DIR)

#logging
logger = logging.getLogger('app')
fh = logging.FileHandler(os.path.join(RESULT_DIR.format(''), 'application.log'))
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(message)s', '%Y-%m-%d %H:%M:%S')
fh.setFormatter(formatter)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

#Test
#import torch
#print torch.has_cudnn
