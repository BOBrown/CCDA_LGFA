# --------------------------------------------------------
# ALL CONFIG
# Copyright (c) 2021 EDL Lab
#
# Written by Bo Zhang
# Adapted from https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/fast_rcnn/config.py
# --------------------------------------------------------
# This CONFIG is an example of adaptation from PotsdamIRRG to Vaihingen
# Another adaptation direction, such as Vaihingen->PotsdamIRRG, should change the codes in this FILE

import os.path as osp

import numpy as np
from easydict import EasyDict

from advent.utils import project_root
from advent.utils.serialization import yaml_load

cfg = EasyDict()

# COMMON CONFIGS
# source domain
cfg.SOURCE = 'PotsdamIRRG'
#cfg.SOURCE = 'Vaihingen'

# target domain
cfg.TARGET = 'Vaihingen'
#cfg.TARGET = 'PotsdamIRRG'

# Number of workers for dataloading
cfg.NUM_WORKERS = 4
# List of training images
cfg.DATA_LIST_SOURCE = str(project_root / 'advent/dataset/PotsdamIRRG/{}.txt')
cfg.DATA_LIST_TARGET = str(project_root / 'advent/dataset/Vaihingen/{}.txt')
#------reverse-------
#cfg.DATA_LIST_SOURCE = str(project_root / 'advent/dataset/Vaihingen/{}.txt')
#cfg.DATA_LIST_TARGET = str(project_root / 'advent/dataset/PotsdamIRRG/{}.txt')

# Directories
cfg.DATA_DIRECTORY_SOURCE = str(project_root / 'data/PotsdamIRRG')
cfg.DATA_DIRECTORY_TARGET = str(project_root / 'data/Vaihingen')
#------reverse-------
#cfg.DATA_DIRECTORY_SOURCE = str(project_root / 'data/Vaihingen')
#cfg.DATA_DIRECTORY_TARGET = str(project_root / 'data/PotsdamIRRG')

# Number of object classes
cfg.NUM_CLASSES = 6 # PotsdamIRRG and Vaihingen datasets both have 6 categories
# Exp dirs
#cfg.EXP_NAME = 'Pots_to_Vaih'
#cfg.EXP_NAME = 'Vaih_to_PotsIRRG'

cfg.EXP_NAME = 'PotsIRRG_to_Vaih'

cfg.EXP_ROOT = project_root / 'experiments'
cfg.EXP_ROOT_SNAPSHOT = osp.join(cfg.EXP_ROOT, 'snapshots')
cfg.EXP_ROOT_LOGS = osp.join(cfg.EXP_ROOT, 'logs')
# CUDA
cfg.GPU_ID = 2

# TRAIN CONFIGS
cfg.TRAIN = EasyDict()
cfg.TRAIN.SET_SOURCE = 'train' # train.txt for stage-one;  train_and_pseudo_r_0.5.txt for stage-two
cfg.TRAIN.SET_TARGET = 'train'
cfg.TRAIN.BATCH_SIZE_SOURCE = 4 # Bo Zhang: NEED TO CHANGE; 4 for training; 1 for entropy calculation
cfg.TRAIN.BATCH_SIZE_TARGET = 4 # Bo Zhang: NEED TO CHANGE; 4 for training; 1 for entropy calculation
cfg.TRAIN.IGNORE_LABEL = 255
cfg.TRAIN.INPUT_SIZE_SOURCE = (512, 512)
cfg.TRAIN.INPUT_SIZE_TARGET = (512, 512)
# Class info
cfg.TRAIN.INFO_SOURCE = ''
cfg.TRAIN.INFO_TARGET = str(project_root / 'advent/dataset/Vaihingen/info.json')
# Segmentation network params
cfg.TRAIN.MODEL = 'DeepLabv2'
cfg.TRAIN.MULTI_LEVEL = False
cfg.TRAIN.RESTORE_FROM = ''

#----Vaihingen------#
#R_mean is 119.997608, G_mean is 81.249737, B_mean is 80.672294
#R_var is 54.817944, G_var is 38.977894, B_var is 37.568813

#----PotsdamRGB------#
#R_mean is 86.761490, G_mean is 92.735321, B_mean is 86.099505
#R_var is 35.850524, G_var is 35.415636, B_var is 36.807795

#------PotsdamIRRG----#
#R_mean is 97.835882, G_mean is 92.735321, B_mean is 86.099505
#R_var is 36.242923, G_var is 35.415636, B_var is 36.807795


cfg.TRAIN.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
cfg.TRAIN.IMG_MEAN_V = np.array((80.672294, 81.249737, 119.997608), dtype=np.float32)
cfg.TRAIN.IMG_MEAN_PR = np.array((86.099505, 92.735321, 86.761490), dtype=np.float32)
cfg.TRAIN.IMG_MEAN_PIR = np.array((86.099505, 92.735321, 97.835882), dtype=np.float32)
cfg.TRAIN.LEARNING_RATE = 2.5e-4 # Bo Zhang: NEED TO CHANGE; 2.5e-4 for deeplab;
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WEIGHT_DECAY = 0.0005
cfg.TRAIN.POWER = 0.9  # learning rate decay
cfg.TRAIN.LAMBDA_SEG_MAIN = 1.0
cfg.TRAIN.LAMBDA_SEG_AUX = 0.1
cfg.TRAIN.LAMBDA_SEG_LOW = 0.1
# Domain adaptation
cfg.TRAIN.DA_METHOD = 'AdvEnt'
# Adversarial training params
cfg.TRAIN.GANLOSS = 'LS' # Option: BCE and LS
cfg.TRAIN.LEARNING_RATE_D = 1e-4
cfg.TRAIN.LAMBDA_ADV_MAIN = 0.001
cfg.TRAIN.LAMBDA_ADV_AUX = 0.0002
cfg.TRAIN.LAMBDA_ADV_LOW = 0.0002
# MinEnt params
cfg.TRAIN.LAMBDA_ENT_MAIN = 0.001  # 0.001
cfg.TRAIN.LAMBDA_ENT_AUX = 0.0002  # 0.0002
# Other params
cfg.TRAIN.MAX_ITERS = 6000
cfg.TRAIN.EARLY_STOP = 3000
cfg.TRAIN.SAVE_PRED_EVERY = 100
cfg.TRAIN.SNAPSHOT_DIR = ''
cfg.TRAIN.RANDOM_SEED = 1234
cfg.TRAIN.TENSORBOARD_LOGDIR = ''
cfg.TRAIN.TENSORBOARD_VIZRATE = 50

# TEST CONFIGS
cfg.TEST = EasyDict()
cfg.TEST.MODE = 'best'  # {'single', 'best'}
# model
cfg.TEST.MODEL = ('DeepLabv2',)  # Bo Zhang:NEED TO CHANGE DeepLabv2
cfg.TEST.MODEL_WEIGHT = (1.0,)
cfg.TEST.MULTI_LEVEL = (False,)
cfg.TEST.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
cfg.TEST.IMG_MEAN_V = np.array((80.672294, 81.249737, 119.997608), dtype=np.float32)
cfg.TEST.IMG_MEAN_PR = np.array((86.099505, 92.735321, 86.761490), dtype=np.float32)
cfg.TEST.IMG_MEAN_PIR = np.array((86.099505, 92.735321, 97.835882), dtype=np.float32)
cfg.TEST.RESTORE_FROM = ('',)
cfg.TEST.SNAPSHOT_DIR = ('',)  # used in 'best' mode
cfg.TEST.SNAPSHOT_STEP = 100  # used in 'best' mode
cfg.TEST.SNAPSHOT_MAXITER = 6000  #60000  # used in 'best' mode
# Test sets
cfg.TEST.SET_TARGET = 'test' #test Vaihingen_validation Vaihingen_test.txt
cfg.TEST.BATCH_SIZE_TARGET = 1
cfg.TEST.INPUT_SIZE_TARGET = (512, 512)
cfg.TEST.OUTPUT_SIZE_TARGET = (512, 512)
cfg.TEST.INFO_TARGET = str(project_root / 'advent/dataset/Vaihingen/info.json')
cfg.TEST.WAIT_MODEL = True


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not EasyDict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        # if not b.has_key(k):
        if k not in b:
            raise KeyError(f'{k} is not a valid config key')

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(f'Type mismatch ({type(b[k])} vs. {type(v)}) '
                                 f'for config key: {k}')

        # recursively merge dicts
        if type(v) is EasyDict:
            try:
                _merge_a_into_b(a[k], b[k])
            except Exception:
                print(f'Error under config key: {k}')
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options.
    """
    yaml_cfg = EasyDict(yaml_load(filename))
    _merge_a_into_b(yaml_cfg, cfg)
