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
from tensorboardX import SummaryWriter

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.config import get_model_name
from core.loss import JointsMSELoss
from core.function import train
from core.function import train_adaptive
from core.function import validate_adaptive
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger

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

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)

    parser.add_argument('--checkpoint',
                        help='checkpoint file',
                        type=str)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.checkpoint:
        config.TRAIN.CHECKPOINT = args.checkpoint


def main():
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model_p, model_d = eval('models.adaptive_pose_resnet.get_adaptive_PoseResNet')(
        config, is_train=True
    )

    if config.TRAIN.CHECKPOINT:
        logger.info('=> loading model from {}'.format(config.TRAIN.CHECKPOINT))
        model_p.load_state_dict(torch.load(config.TRAIN.CHECKPOINT))
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth.tar')
        logger.info('=> loading model from {}'.format(model_state_file))
        model_p.load_state_dict(torch.load(model_state_file))

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models/adaptive_pose_resnet.py'),
        final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand((config.TRAIN.BATCH_SIZE,
                             3,
                             config.MODEL.IMAGE_SIZE[1],
                             config.MODEL.IMAGE_SIZE[0]))
    writer_dict['writer'].add_graph(model_p, (dump_input, ), verbose=False)

    gpus = [int(i) for i in config.GPUS.split(',')]
    model_p = torch.nn.DataParallel(model_p, device_ids=gpus).cuda()
    model_d = torch.nn.DataParallel(model_d, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer for pose_restnet
    criterion_p = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT
    ).cuda()
    
    
    # freeze some layers
    idx = 0 
    for param in model_p.parameters():
        if idx <= 71:  #fix 140 for res 1,2,3,4 or fix 71 for res 1,2,3 or fix 33 for res 1,2
           param.requires_grad = False
            #print(param.data.shape)
        idx = idx + 1
    
    optimizer_p = get_optimizer(config, model_p)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_p, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
    )

    # define loss function (criterion) and optimizer for domain
    criterion_d = torch.nn.BCEWithLogitsLoss().cuda()
    optimizer_d = get_optimizer(config, model_d)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_d, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
    )


    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_pre_dataset = eval('dataset.'+config.DATASET.DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TRAIN_PRE_SET,
        True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_dataset = eval('dataset.'+config.DATASET.DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TRAIN_SET,
        True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    valid_dataset = eval('dataset.'+config.DATASET.DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TEST_SET,
        False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_pre_loader = torch.utils.data.DataLoader(
        train_pre_dataset,
        batch_size=config.TRAIN.BATCH_SIZE1,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    
    
    train_loader = torch.utils.data.DataLoader(   
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    
    
    best_perf = 0.0
    best_model = False
    epoch_D = 10
    losses_D_list = []
    acces_D_list = []
    acc_num_total = 0
    num = 0
    losses_d = AverageMeter()
    
    # Pretrained Stage
    print('Pretrained Stage:')
    print('Start to train Domain Classifier-------')    
    for epoch_d in range(epoch_D):  # epoch
        for i, (input, target, target_weight, meta) in enumerate(train_pre_loader):  # iteration     
            # compute output for pose_resnet
            feature_output, kpt_output = model_p(input)
            
            # compute for domain classifier
            domain_logits = model_d(feature_output.detach())
            domain_label = (meta['synthetic'].unsqueeze(-1)*1.0).cuda()
          
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

            if i % config.PRINT_FREQ == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Accuracy_d: {3} ({4})\t'\
                      'Loss_d: {loss_d.val:.5f} ({loss_d.avg:.5f})'.format(
                          epoch_d, i, len(train_pre_loader), acc_d, acc_num_total * 1.0 / num, loss_d = losses_d)
                logger.info(msg)
            losses_D_list.append(losses_d.val)
    
        
    print('Stage I and II:')
    losses_P_list = []
    acces_P_list = []
    losses_p = AverageMeter() 
    acces_p = AverageMeter()
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        lr_scheduler.step()

        # train for one epoch
        losses_P_list,losses_D_list, acces_P_list, acces_D_list = train_adaptive(config, train_loader, model_p, model_d, criterion_p, criterion_d, optimizer_p, optimizer_d, epoch, final_output_dir, tb_log_dir, writer_dict, losses_P_list, losses_D_list, acces_P_list, acces_D_list, acc_num_total, num, losses_p, acces_p, losses_d)
 

        # evaluate on validation set
        perf_indicator = validate_adaptive(config, valid_loader, valid_dataset, model_p,
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
            'model': get_model_name(config), 
            'state_dict': model_p.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer_p.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model_p.state_dict(), final_model_state_file)
    writer_dict['writer'].close()

    np.save('./losses_D.npy', np.array(losses_D_list))  # Adverial-D
    np.save('./losses_P.npy', np.array(losses_P_list))   # P
    np.save('./acces_P.npy', np.array(acces_P_list))   # D    
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
