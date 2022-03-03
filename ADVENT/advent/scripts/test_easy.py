# --------------------------------------------------------
# Network Inference
# Copyright (c) 2021 EDL Lab
#
# Written by Bo Zhang
# --------------------------------------------------------
import sys
import pdb

import argparse
import os
import os.path as osp
import pprint
import warnings

from torch.utils import data
from advent.dataset.Potsdam import Potsdam
from advent.dataset.Vaihingen import Vaihingen
from advent.domain_adaptation.config import cfg, cfg_from_file

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for evaluation")
    parser.add_argument('--cfg', type=str, default=None,
                        help='optional config file', )
    parser.add_argument('--status', type=int, default=0,
                        help='0 denotes the Feature-Level(FL) adaptation, while 1 denotes the Entropy-Level(EL) adaptation', )
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    return parser.parse_args()


def main(config_file, exp_suffix):
    # LOAD ARGS
    assert config_file is not None, 'Missing cfg file'
    cfg_from_file(config_file)

    # For feature-level adaptation
    if args.status == 0:
        from advent.model.deeplabv3 import get_deeplab_v3_FL_Adapt as get_deeplab_v3
        from advent.domain_adaptation.eval_UDA_single_FL import evaluate_domain_adaptation

    # For entropy-level adaptation
    if args.status == 1:
        from advent.model.deeplabv3 import get_deeplab_v3_EL_Adapt as get_deeplab_v3
        from advent.domain_adaptation.eval_UDA_single_EL import evaluate_domain_adaptation

    # auto-generate exp name if not specified
    if cfg.EXP_NAME == '':
        cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}'
    if exp_suffix:
        cfg.EXP_NAME += f'_{exp_suffix}'
    # auto-generate snapshot path if not specified
    if cfg.TEST.SNAPSHOT_DIR[0] == '':
        cfg.TEST.SNAPSHOT_DIR[0] = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)
        os.makedirs(cfg.TEST.SNAPSHOT_DIR[0], exist_ok=True)

    print('Using config:')
    pprint.pprint(cfg)
    # load models
    models = []
    n_models = len(cfg.TEST.MODEL)
    if cfg.TEST.MODE == 'best':
        assert n_models == 1, 'Not yet supported'
    for i in range(n_models):
        if cfg.TEST.MODEL[i] == 'DeepLabv3':
            model = get_deeplab_v3(num_classes=cfg.NUM_CLASSES,
                                   multi_level=cfg.TEST.MULTI_LEVEL[i])
        elif cfg.TEST.MODEL[i] == 'UNET':
            model = UNet(n_channels=3, n_classes=cfg.NUM_CLASSES, bilinear=True)
        else:
            raise NotImplementedError(f"Not yet supported {cfg.TEST.MODEL[i]}")
        models.append(model)

    if os.environ.get('ADVENT_DRY_RUN', '0') == '1':
        return

    # dataloaders
    #pdb.set_trace()
    test_dataset = Vaihingen(root=cfg.DATA_DIRECTORY_TARGET,
                                     list_path='/root/code/CCDA_LGFA/ADVENT/advent/dataset/Vaihingen/{}.txt',
                                     set=cfg.TEST.SET_TARGET,
                                     info_path=cfg.TEST.INFO_TARGET,
                                     crop_size=cfg.TEST.INPUT_SIZE_TARGET,
                                     mean=cfg.TEST.IMG_MEAN,
                                     labels_size=cfg.TEST.OUTPUT_SIZE_TARGET)
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=cfg.TEST.BATCH_SIZE_TARGET,
                                  num_workers=cfg.NUM_WORKERS,
                                  shuffle=False,
                                  pin_memory=True)
    #-----reverse-----
    # test_dataset = Potsdam(root=cfg.DATA_DIRECTORY_TARGET,
    #                          list_path='/root/code/CCDA_LGFA/ADVENT/advent/dataset/PotsdamIRRG/{}.txt',
    #                          set=cfg.TEST.SET_TARGET,
    #                          info_path=cfg.TEST.INFO_TARGET,
    #                          crop_size=cfg.TEST.INPUT_SIZE_TARGET,
    #                          mean=cfg.TEST.IMG_MEAN)
    # test_loader = data.DataLoader(test_dataset,
    #                               batch_size=cfg.TEST.BATCH_SIZE_TARGET,
    #                               num_workers=cfg.NUM_WORKERS,
    #                               shuffle=False,
    #                               pin_memory=True)

    # eval
    evaluate_domain_adaptation(models, test_loader, cfg)


if __name__ == '__main__':
    args = get_arguments()
    print('Called with args:')
    print(args)
    main(args.cfg, args.exp_suffix)
