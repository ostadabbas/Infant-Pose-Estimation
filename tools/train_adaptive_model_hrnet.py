# ------------------------------------------------------------------------------
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xiaofei Huang (xhuang@ece.neu.edu) and Nihang Fu (nihang@ece.neu.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import train_adaptive
from core.function import validate_adaptive
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary
from utils.sampler import BalancedBatchSampler

import numpy as np

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args
    

def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model_p, model_d = eval('models.'+cfg.MODEL.NAME+'.get_adaptive_pose_net')(
        cfg, is_train=True
    )

    if cfg.TRAIN.CHECKPOINT:
        logger.info('=> loading model from {}'.format(cfg.TRAIN.CHECKPOINT))
        model_p.load_state_dict(torch.load(cfg.TRAIN.CHECKPOINT))


    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'pre_train_global_steps': 0,
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand((1,
                             3,
                             cfg.MODEL.IMAGE_SIZE[1],
                             cfg.MODEL.IMAGE_SIZE[0]))
    writer_dict['writer'].add_graph(model_p, (dump_input, ), verbose=False)
    
    logger.info(get_model_summary(model_p, dump_input))
    
    model_p = torch.nn.DataParallel(model_p, device_ids=cfg.GPUS).cuda()
    model_d = torch.nn.DataParallel(model_d, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer for pose_net
    criterion_p = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()
    
    optimizer_p = get_optimizer(cfg, model_p)


    # define loss function (criterion) and optimizer for domain
    criterion_d = torch.nn.BCEWithLogitsLoss().cuda()
    optimizer_d = get_optimizer(cfg, model_d)


    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_pre_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg,
        cfg.DATASET.ROOT,
        cfg.DATASET.TRAIN_PRE_SET,
        True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg,
        cfg.DATASET.ROOT,
        cfg.DATASET.TRAIN_SET,
        True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg,
        cfg.DATASET.ROOT,
        cfg.DATASET.TEST_SET,
        False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_pre_loader = torch.utils.data.DataLoader(
        train_pre_dataset,
        batch_size=cfg.TRAIN.PRE_BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    
    syn_labels = train_dataset._load_syrip_syn_annotations()
    train_loader = torch.utils.data.DataLoader(   
        train_dataset,
        sampler=BalancedBatchSampler(train_dataset, syn_labels),
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    
    
    best_perf = 0.0
    best_model = False
    last_epoch = -1
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model_p.load_state_dict(checkpoint['state_dict'])

        optimizer_p.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))
    
    # freeze some layers
    idx = 0 
    for param in model_p.parameters():
        
        if idx <= 108:  #fix 108 for stage 2 + bottleneck  or fix 483 for stage 3 + stage 2+ bottleneck
           param.requires_grad = False
            #print(param.data.shape)
        idx = idx + 1
                
    
    lr_scheduler_p = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_p, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR, last_epoch=last_epoch
    )

    lr_scheduler_d = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_d, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR
    )
    
    epoch_D = cfg.TRAIN.PRE_EPOCH
    losses_D_list = []
    acces_D_list = []
    acc_num_total = 0
    num = 0
    losses_d = AverageMeter()
    
    # Pretrained Stage
    print('Pretrained Stage:')
    print('Start to train Domain Classifier-------')    
    for epoch_d in range(epoch_D):  # epoch
        model_d.train()
        model_p.train()
    
        for i, (input, target, target_weight, meta) in enumerate(train_pre_loader):  # iteration     
            # compute output for pose_net
            feature_outputs, outputs = model_p(input)
            #print(feature_outputs.size())
            # compute for domain classifier
            domain_logits = model_d(feature_outputs.detach())
            domain_label = (meta['synthetic'].unsqueeze(-1)*1.0).cuda(non_blocking=True)
            # print(domain_label)
          
            loss_d = criterion_d(domain_logits, domain_label)
            loss_d.backward(retain_graph=True)
            optimizer_d.step()
            
            # compute accuracy of classifier
            acc_num = 0
            for j in range(len(domain_label)):
                if (domain_logits[j] > 0 and domain_label[j] == 1.0) or (domain_logits[j] < 0 and domain_label[j] == 0.0):
                    acc_num += 1
                    acc_num_total += 1
                num += 1
            acc_d = acc_num * 1.0 / input.size(0)
            acces_D_list.append(acc_d)
                
            optimizer_d.zero_grad()
            losses_d.update(loss_d.item(), input.size(0))

            if i % cfg.PRINT_FREQ == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Accuracy_d: {3} ({4})\t' \
                      'Loss_d: {loss_d.val:.5f} ({loss_d.avg:.5f})'.format(
                          epoch_d, i, len(train_pre_loader), acc_d, acc_num_total * 1.0 / num, loss_d = losses_d)
                logger.info(msg)
                
                writer = writer_dict['writer']
                pre_global_steps = writer_dict['pre_train_global_steps']
                writer.add_scalar('pre_train_loss_D', losses_d.val, pre_global_steps)
                writer.add_scalar('pre_train_acc_D', acc_d, pre_global_steps)
                writer_dict['pre_train_global_steps'] = pre_global_steps + 1
                              
            losses_D_list.append(losses_d.val)          
    
        
    print('Training Stage (Step I and II):')
    losses_P_list = []
    acces_P_list = []
    losses_p = AverageMeter() 
    acces_p = AverageMeter()
    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        lr_scheduler_p.step()

        # train for one epoch
        losses_P_list,losses_D_list, acces_P_list, acces_D_list = train_adaptive(cfg, train_loader, model_p, model_d, criterion_p, criterion_d, optimizer_p, optimizer_d, epoch, final_output_dir, tb_log_dir, writer_dict, losses_P_list, losses_D_list, acces_P_list, acces_D_list, acc_num_total, num, losses_p, acces_p, losses_d)
 
        # evaluate on validation set
        perf_indicator = validate_adaptive(cfg, valid_loader, valid_dataset, model_p,
                                  criterion_p, final_output_dir, tb_log_dir,
                                  writer_dict)

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME, 
            'state_dict': model_p.state_dict(),
            'best_state_dict': model_p.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer_p.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model_p.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()

    np.save('./losses_D.npy', np.array(losses_D_list))  # Adversarial-D
    np.save('./losses_P.npy', np.array(losses_P_list))   # P
    np.save('./acces_P.npy', np.array(acces_P_list))   # P   
    np.save('./acces_D.npy', np.array(acces_D_list))   # D
    
   
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

if __name__ == '__main__':
    main()
