# --------------------------------------------------------
# Adversarial training network
# Copyright (c) 2021 EDL Lab
#
# Written by Bo Zhang
# --------------------------------------------------------
import os
import sys
from pathlib import Path

import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from torchvision.utils import make_grid
from tqdm import tqdm

from advent.model.discriminator import get_fc_discriminator
from advent.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from advent.utils.func import loss_calc, bce_loss, ls_loss
from advent.utils.loss import entropy_loss
from advent.utils.func import prob_2_entropy
from advent.utils.viz_segmask import colorize_mask


def train_advent(model, trainloader, targetloader, cfg):
    ''' UDA training with advent
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # DISCRIMINATOR NETWORK
    # feature-level
    d_aux = get_fc_discriminator(num_classes=num_classes)
    d_aux.train()
    d_aux.to(device)

    # seg maps, i.e. output, level
    d_main = get_fc_discriminator(num_classes=num_classes)
    d_main.train()
    d_main.to(device)

    # seg maps, i.e. class-wise

    d_cls_main_0 = get_fc_discriminator(num_classes=1)
    d_cls_main_0.train()
    d_cls_main_0.to(device)

    d_cls_main_1 = get_fc_discriminator(num_classes=1)
    d_cls_main_1.train()
    d_cls_main_1.to(device)

    d_cls_main_2 = get_fc_discriminator(num_classes=1)
    d_cls_main_2.train()
    d_cls_main_2.to(device)

    d_cls_main_3 = get_fc_discriminator(num_classes=1)
    d_cls_main_3.train()
    d_cls_main_3.to(device)

    d_cls_main_4 = get_fc_discriminator(num_classes=1)
    d_cls_main_4.train()
    d_cls_main_4.to(device)

    d_cls_main_5 = get_fc_discriminator(num_classes=1)
    d_cls_main_5.train()
    d_cls_main_5.to(device)

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # discriminators' optimizers
    optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                 betas=(0.9, 0.99))
    optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))

    optimizer_d_cls = optim.Adam([
                                {'params': d_cls_main_0.parameters()},
                                {'params': d_cls_main_1.parameters()},
                                {'params': d_cls_main_2.parameters()},
                                {'params': d_cls_main_3.parameters()},
                                {'params': d_cls_main_4.parameters()},
                                {'params': d_cls_main_5.parameters()}], lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1
    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):

        # reset optimizers
        optimizer.zero_grad()
        optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()
        optimizer_d_cls.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_cls, i_iter, cfg)

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        for param in d_aux.parameters():
            param.requires_grad = False
        for param in d_main.parameters():
            param.requires_grad = False

        for param in d_cls_main_0.parameters():
            param.requires_grad = False
        for param in d_cls_main_1.parameters():
            param.requires_grad = False
        for param in d_cls_main_2.parameters():
            param.requires_grad = False
        for param in d_cls_main_3.parameters():
            param.requires_grad = False
        for param in d_cls_main_4.parameters():
            param.requires_grad = False
        for param in d_cls_main_5.parameters():
            param.requires_grad = False
        # Holobo: Changed
        # for param in model.parameters():
        #     param.requires_grad = False
        #
        # for name, param in model.named_parameters():
        #     if "layer6" in name:
        #         param.requires_grad = False

        # train on source
        _, batch = trainloader_iter.__next__()
        images_source, labels, _, _ = batch
        pred_src_aux, pred_src_main = model(images_source.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux)
            loss_seg_src_aux = loss_calc(pred_src_aux, labels, device)
        else:
            loss_seg_src_aux = 0
        pred_src_main = interp(pred_src_main)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        loss.backward()

        # adversarial training ot fool the discriminator
        _, batch = targetloader_iter.__next__()
        images, _, _, _ = batch
        pred_trg_aux, pred_trg_main = model(images.cuda(device))

        # Non Class-wise
        # if cfg.TRAIN.MULTI_LEVEL:
        #     pred_trg_aux = interp_target(pred_trg_aux)
        #     d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
        #     if cfg.TRAIN.GANLOSS == 'BCE':
        #         loss_adv_trg_aux = bce_loss(d_out_aux, source_label)
        #     elif cfg.TRAIN.GANLOSS == 'LS':
        #         loss_adv_trg_aux = ls_loss(d_out_aux, source_label)
        # else:
        #     loss_adv_trg_aux = 0
        # pred_trg_main = interp_target(pred_trg_main)
        # d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
        # if cfg.TRAIN.GANLOSS == 'BCE':
        #     loss_adv_trg_main = bce_loss(d_out_main, source_label)
        # elif cfg.TRAIN.GANLOSS == 'LS':
        #     loss_adv_trg_main = ls_loss(d_out_main, source_label)
        # loss_adv = (cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main
        #         + cfg.TRAIN.LAMBDA_ADV_AUX * loss_adv_trg_aux)


        # Class-wise Entropy-based Adversarial
        sig_trg_main = F.sigmoid(pred_trg_main)
        enl_trg_main = prob_2_entropy(sig_trg_main)
        enl_trg_main_0, enl_trg_main_1, enl_trg_main_2, enl_trg_main_3, enl_trg_main_4, enl_trg_main_5 = enl_trg_main.split(1, 1)

        #---CLS 0---
        d_out_cls_0 = d_cls_main_0(enl_trg_main_0)
        if cfg.TRAIN.GANLOSS == 'BCE':
            loss_cls_0_adv_trg_main = bce_loss(d_out_cls_0, source_label)
        elif cfg.TRAIN.GANLOSS == 'LS':
            loss_cls_0_adv_trg_main = ls_loss(d_out_cls_0, source_label)

        # ---CLS 1---
        d_out_cls_1 = d_cls_main_1(enl_trg_main_1)
        if cfg.TRAIN.GANLOSS == 'BCE':
            loss_cls_1_adv_trg_main = bce_loss(d_out_cls_1, source_label)
        elif cfg.TRAIN.GANLOSS == 'LS':
            loss_cls_1_adv_trg_main = ls_loss(d_out_cls_1, source_label)

        # ---CLS 2---
        d_out_cls_2 = d_cls_main_2(enl_trg_main_2)
        if cfg.TRAIN.GANLOSS == 'BCE':
            loss_cls_2_adv_trg_main = bce_loss(d_out_cls_2, source_label)
        elif cfg.TRAIN.GANLOSS == 'LS':
            loss_cls_2_adv_trg_main = ls_loss(d_out_cls_2, source_label)

        # ---CLS 2---
        d_out_cls_3 = d_cls_main_3(enl_trg_main_3)
        if cfg.TRAIN.GANLOSS == 'BCE':
            loss_cls_3_adv_trg_main = bce_loss(d_out_cls_3, source_label)
        elif cfg.TRAIN.GANLOSS == 'LS':
            loss_cls_3_adv_trg_main = ls_loss(d_out_cls_3, source_label)

        # ---CLS 4---
        d_out_cls_4 = d_cls_main_4(enl_trg_main_4)
        if cfg.TRAIN.GANLOSS == 'BCE':
            loss_cls_4_adv_trg_main = bce_loss(d_out_cls_4, source_label)
        elif cfg.TRAIN.GANLOSS == 'LS':
            loss_cls_4_adv_trg_main = ls_loss(d_out_cls_4, source_label)

        # ---CLS 5---
        d_out_cls_5 = d_cls_main_5(enl_trg_main_5)
        if cfg.TRAIN.GANLOSS == 'BCE':
            loss_cls_5_adv_trg_main = bce_loss(d_out_cls_5, source_label)
        elif cfg.TRAIN.GANLOSS == 'LS':
            loss_cls_5_adv_trg_main = ls_loss(d_out_cls_5, source_label)

        loss_cls_wise = cfg.TRAIN.LAMBDA_ADV_AUX * (loss_cls_0_adv_trg_main + loss_cls_1_adv_trg_main + loss_cls_2_adv_trg_main + loss_cls_3_adv_trg_main + loss_cls_4_adv_trg_main + loss_cls_5_adv_trg_main)

        loss_adv_all = loss_cls_wise
        loss_adv_all.backward()

        # Train discriminator networks
        # enable training mode on discriminator networks
        for param in d_aux.parameters():
            param.requires_grad = True
        for param in d_main.parameters():
            param.requires_grad = True

        for param in d_cls_main_0.parameters():
            param.requires_grad = True
        for param in d_cls_main_1.parameters():
            param.requires_grad = True
        for param in d_cls_main_2.parameters():
            param.requires_grad = True
        for param in d_cls_main_3.parameters():
            param.requires_grad = True
        for param in d_cls_main_4.parameters():
            param.requires_grad = True
        for param in d_cls_main_5.parameters():
            param.requires_grad = True
        # train with source
        # if cfg.TRAIN.MULTI_LEVEL:
        #     pred_src_aux = pred_src_aux.detach()
        #     d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_src_aux)))
        #     if cfg.TRAIN.GANLOSS == 'BCE':
        #         loss_d_aux = bce_loss(d_out_aux, source_label)
        #     elif cfg.TRAIN.GANLOSS == 'LS':
        #         loss_d_aux = ls_loss(d_out_aux, source_label)
        #     loss_d_aux = loss_d_aux / 2
        #     loss_d_aux.backward()
        # pred_src_main = pred_src_main.detach()
        # d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main)))
        # if cfg.TRAIN.GANLOSS == 'BCE':
        #     loss_d_main = bce_loss(d_out_main, source_label)
        # elif cfg.TRAIN.GANLOSS == 'LS':
        #     loss_d_main = ls_loss(d_out_main, source_label)
        # loss_d_main = loss_d_main / 2
        # loss_d_main.backward()
        #
        # # train with target
        # if cfg.TRAIN.MULTI_LEVEL:
        #     pred_trg_aux = pred_trg_aux.detach()
        #     d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
        #     if cfg.TRAIN.GANLOSS == 'BCE':
        #         loss_d_aux = bce_loss(d_out_aux, target_label)
        #     elif cfg.TRAIN.GANLOSS == 'LS':
        #         loss_d_aux = ls_loss(d_out_aux, target_label)
        #     loss_d_aux = loss_d_aux / 2
        #     loss_d_aux.backward()
        # else:
        #     loss_d_aux = 0
        # pred_trg_main = pred_trg_main.detach()
        # d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
        # if cfg.TRAIN.GANLOSS == 'BCE':
        #     loss_d_main = bce_loss(d_out_main, target_label)
        # elif cfg.TRAIN.GANLOSS == 'LS':
        #     loss_d_main = ls_loss(d_out_main, target_label)
        # loss_d_main = loss_d_main / 2
        # loss_d_main.backward()


        # Class-wise Entropy-based Adversarial
        sig_src_main = F.sigmoid(pred_src_main)
        enl_src_main = prob_2_entropy(sig_src_main)
        enl_src_main_0, enl_src_main_1, enl_src_main_2, enl_src_main_3, enl_src_main_4, enl_src_main_5 = enl_src_main.split(1, 1)
        enl_src_main_0, enl_src_main_1, enl_src_main_2, enl_src_main_3, enl_src_main_4, enl_src_main_5 = enl_src_main_0.detach(), enl_src_main_1.detach(), enl_src_main_2.detach(), enl_src_main_3.detach(), enl_src_main_4.detach(), enl_src_main_5.detach()

        enl_trg_main_0, enl_trg_main_1, enl_trg_main_2, enl_trg_main_3, enl_trg_main_4, enl_trg_main_5 = enl_trg_main_0.detach(), enl_trg_main_1.detach(), enl_trg_main_2.detach(), enl_trg_main_3.detach(), enl_trg_main_4.detach(), enl_trg_main_5.detach()


        # ---CLS 0---
        d_out_main_0 = d_cls_main_0(enl_src_main_0)
        if cfg.TRAIN.GANLOSS == 'BCE':
            loss_d_main_cls_0 = bce_loss(d_out_main_0, source_label)
        elif cfg.TRAIN.GANLOSS == 'LS':
            loss_d_main_cls_0 = ls_loss(d_out_main_0, source_label)
        loss_d_main_cls_0 = loss_d_main_cls_0 / 2
        loss_d_main_cls_0.backward()

        # train with target
        d_out_main_0 = d_cls_main_0(enl_trg_main_0)
        if cfg.TRAIN.GANLOSS == 'BCE':
            loss_d_main_cls_0 = bce_loss(d_out_main_0, target_label)
        elif cfg.TRAIN.GANLOSS == 'LS':
            loss_d_main_cls_0 = ls_loss(d_out_main_0, target_label)
        loss_d_main_cls_0 = loss_d_main_cls_0 / 2
        loss_d_main_cls_0.backward()

        # ---CLS 1---
        d_out_main_1 = d_cls_main_1(enl_src_main_1)
        if cfg.TRAIN.GANLOSS == 'BCE':
            loss_d_main_cls_1 = bce_loss(d_out_main_1, source_label)
        elif cfg.TRAIN.GANLOSS == 'LS':
            loss_d_main_cls_1 = ls_loss(d_out_main_1, source_label)
        loss_d_main_cls_1 = loss_d_main_cls_1 / 2
        loss_d_main_cls_1.backward()

        # train with target
        d_out_main_1 = d_cls_main_1(enl_trg_main_1)
        if cfg.TRAIN.GANLOSS == 'BCE':
            loss_d_main_cls_1 = bce_loss(d_out_main_1, target_label)
        elif cfg.TRAIN.GANLOSS == 'LS':
            loss_d_main_cls_1 = ls_loss(d_out_main_1, target_label)
        loss_d_main_cls_1 = loss_d_main_cls_1 / 2
        loss_d_main_cls_1.backward()

        # ---CLS 2---
        d_out_main_2 = d_cls_main_2(enl_src_main_2)
        if cfg.TRAIN.GANLOSS == 'BCE':
            loss_d_main_cls_2 = bce_loss(d_out_main_2, source_label)
        elif cfg.TRAIN.GANLOSS == 'LS':
            loss_d_main_cls_2 = ls_loss(d_out_main_2, source_label)
        loss_d_main_cls_2 = loss_d_main_cls_2 / 2
        loss_d_main_cls_2.backward()

        # train with target
        d_out_main_2 = d_cls_main_2(enl_trg_main_2)
        if cfg.TRAIN.GANLOSS == 'BCE':
            loss_d_main_cls_2 = bce_loss(d_out_main_2, target_label)
        elif cfg.TRAIN.GANLOSS == 'LS':
            loss_d_main_cls_2 = ls_loss(d_out_main_2, target_label)
        loss_d_main_cls_2 = loss_d_main_cls_2 / 2
        loss_d_main_cls_2.backward()

        # ---CLS 3---
        d_out_main_3 = d_cls_main_3(enl_src_main_3)
        if cfg.TRAIN.GANLOSS == 'BCE':
            loss_d_main_cls_3 = bce_loss(d_out_main_3, source_label)
        elif cfg.TRAIN.GANLOSS == 'LS':
            loss_d_main_cls_3 = ls_loss(d_out_main_3, source_label)
        loss_d_main_cls_3 = loss_d_main_cls_3 / 2
        loss_d_main_cls_3.backward()

        # train with target
        d_out_main_3 = d_cls_main_3(enl_trg_main_3)
        if cfg.TRAIN.GANLOSS == 'BCE':
            loss_d_main_cls_3 = bce_loss(d_out_main_3, target_label)
        elif cfg.TRAIN.GANLOSS == 'LS':
            loss_d_main_cls_3 = ls_loss(d_out_main_3, target_label)
        loss_d_main_cls_3 = loss_d_main_cls_3 / 2
        loss_d_main_cls_3.backward()

        # ---CLS 4---
        d_out_main_4 = d_cls_main_4(enl_src_main_4)
        if cfg.TRAIN.GANLOSS == 'BCE':
            loss_d_main_cls_4 = bce_loss(d_out_main_4, source_label)
        elif cfg.TRAIN.GANLOSS == 'LS':
            loss_d_main_cls_4 = ls_loss(d_out_main_4, source_label)
        loss_d_main_cls_4 = loss_d_main_cls_4 / 2
        loss_d_main_cls_4.backward()

        # train with target
        d_out_main_4 = d_cls_main_4(enl_trg_main_4)
        if cfg.TRAIN.GANLOSS == 'BCE':
            loss_d_main_cls_4 = bce_loss(d_out_main_4, target_label)
        elif cfg.TRAIN.GANLOSS == 'LS':
            loss_d_main_cls_4 = ls_loss(d_out_main_4, target_label)
        loss_d_main_cls_4 = loss_d_main_cls_4 / 2
        loss_d_main_cls_4.backward()

        # ---CLS 5---
        d_out_main_5 = d_cls_main_5(enl_src_main_5)
        if cfg.TRAIN.GANLOSS == 'BCE':
            loss_d_main_cls_5 = bce_loss(d_out_main_5, source_label)
        elif cfg.TRAIN.GANLOSS == 'LS':
            loss_d_main_cls_5 = ls_loss(d_out_main_5, source_label)
        loss_d_main_cls_5 = loss_d_main_cls_5 / 2
        loss_d_main_cls_5.backward()

        # train with target
        d_out_main_5 = d_cls_main_5(enl_trg_main_5)
        if cfg.TRAIN.GANLOSS == 'BCE':
            loss_d_main_cls_5 = bce_loss(d_out_main_5, target_label)
        elif cfg.TRAIN.GANLOSS == 'LS':
            loss_d_main_cls_5 = ls_loss(d_out_main_5, target_label)
        loss_d_main_cls_5 = loss_d_main_cls_5 / 2
        loss_d_main_cls_5.backward()


        optimizer.step()
        if cfg.TRAIN.MULTI_LEVEL:
            optimizer_d_aux.step()
        #optimizer_d_main.step()
        optimizer_d_cls.step()

        current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_seg_src_main': loss_seg_src_main,
                          #'loss_adv_trg_aux': loss_adv_trg_aux,
                          #'loss_adv_trg_main': loss_adv_trg_main,
                          #'loss_d_aux': loss_d_aux,
                          #'loss_d_main': loss_d_main,
                          'loss_cls_0_adv_trg_main': loss_cls_0_adv_trg_main,
                          'loss_cls_1_adv_trg_main': loss_cls_1_adv_trg_main,
                          'loss_cls_2_adv_trg_main': loss_cls_2_adv_trg_main,
                          'loss_cls_3_adv_trg_main': loss_cls_3_adv_trg_main,
                          'loss_cls_4_adv_trg_main': loss_cls_4_adv_trg_main,
                          'loss_cls_5_adv_trg_main': loss_cls_5_adv_trg_main,
                          'loss_d_main_cls_0': loss_d_main_cls_0,
                          'loss_d_main_cls_1': loss_d_main_cls_1,
                          'loss_d_main_cls_2': loss_d_main_cls_2,
                          'loss_d_main_cls_3': loss_d_main_cls_3,
                          'loss_d_main_cls_4': loss_d_main_cls_4,
                          'loss_d_main_cls_5': loss_d_main_cls_5}
        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            torch.save(d_aux.state_dict(), snapshot_dir / f'model_{i_iter}_D_aux.pth')
            torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D_main.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images, i_iter, pred_trg_main, num_classes, 'T')
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S')


def draw_in_tensorboard(writer, images, i_iter, pred_main, num_classes, type_):
    grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image(f'Image - {type_}', grid_image, i_iter)

    grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(np.asarray(
        np.argmax(F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0),
                  axis=2), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,
                           normalize=False, range=(0, 255))
    writer.add_image(f'Prediction - {type_}', grid_image, i_iter)

    output_sm = F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0)
    output_ent = np.sum(-np.multiply(output_sm, np.log2(output_sm)), axis=2,
                        keepdims=False)
    grid_image = make_grid(torch.from_numpy(output_ent), 3, normalize=True,
                           range=(0, np.log2(num_classes)))
    writer.add_image(f'Entropy - {type_}', grid_image, i_iter)


def train_minent(model, trainloader, targetloader, cfg):
    ''' UDA training with minEnt
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP)):

        # reset optimizers
        optimizer.zero_grad()

        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)

        # UDA Training
        # train on source
        _, batch = trainloader_iter.__next__()
        images_source, labels, _, _ = batch
        pred_src_aux, pred_src_main = model(images_source.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux)
            loss_seg_src_aux = loss_calc(pred_src_aux, labels, device)
        else:
            loss_seg_src_aux = 0
        pred_src_main = interp(pred_src_main)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        loss.backward()

        # adversarial training with minent
        _, batch = targetloader_iter.__next__()
        images, _, _, _ = batch
        pred_trg_aux, pred_trg_main = model(images.cuda(device))
        pred_trg_aux = interp_target(pred_trg_aux)
        pred_trg_main = interp_target(pred_trg_main)
        pred_prob_trg_aux = F.softmax(pred_trg_aux)
        pred_prob_trg_main = F.softmax(pred_trg_main)

        loss_target_entp_aux = entropy_loss(pred_prob_trg_aux)
        loss_target_entp_main = entropy_loss(pred_prob_trg_main)
        loss = (cfg.TRAIN.LAMBDA_ENT_AUX * loss_target_entp_aux
                + cfg.TRAIN.LAMBDA_ENT_MAIN * loss_target_entp_main)
        loss.backward()
        optimizer.step()

        current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_seg_src_main': loss_seg_src_main,
                          'loss_ent_aux': loss_target_entp_aux,
                          'loss_ent_main': loss_target_entp_main}

        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(),
                       osp.join(cfg.TRAIN.SNAPSHOT_DIR, f'model_{i_iter}.pth'))
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images, i_iter, pred_trg_main, num_classes, 'T')
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S')


def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iter} {full_string}')


def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)


def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()


def train_domain_adaptation(model, trainloader, targetloader, cfg):
    if cfg.TRAIN.DA_METHOD == 'MinEnt':
        train_minent(model, trainloader, targetloader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'AdvEnt':
        train_advent(model, trainloader, targetloader, cfg)
    else:
        raise NotImplementedError(f"Not yet supported DA method {cfg.TRAIN.DA_METHOD}")
