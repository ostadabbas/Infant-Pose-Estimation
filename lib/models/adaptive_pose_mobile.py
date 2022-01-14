# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Xiaofei Huang (xhuang@ece.neu.edu) and Nihang Fu (nihang@ece.neu.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import math
from .DUC import DUC

import torch
import torch.nn as nn
import torch.nn.functional as F
import dsntnn

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class DomainClassifier(nn.Module):

    def __init__(self, feature_output):
        super(DomainClassifier, self).__init__()
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=1, stride=1),     # no padding with bias
            nn.BatchNorm2d(512, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, stride=1),  
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 16, kernel_size=1, stride=1),
            nn.BatchNorm2d(16, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.layer = nn.Sequential(
            nn.Linear(16*3*3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )


    def forward(self, feature_output):
        x = self.conv_layer(feature_output)
        #print(x.size())
        out = self.layer(x.view(-1, 16 * 3 * 3))
        return out


class PoseMobileNet(nn.Module):
    def __init__(self, cfg, n_class=1000, input_size=224, width_mult=1.):
        super(PoseMobileNet, self).__init__()
        self.feature_output = None
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        self.conv_compress = nn.Conv2d(1280, 256, 1, 1, 0, bias=False)
        self.duc1 = DUC(256, 512, upscale_factor=2)
        self.duc2 = DUC(128, 256, upscale_factor=2)
        self.duc3 = DUC(64, 128, upscale_factor=2)

        # building classifier
        #self.classifier = nn.Sequential(
        #    nn.Dropout(0.2),
        #    nn.Linear(self.last_channel, n_class),
        #)
        
        # Use a 1x1 conv to get one unnormalized heatmap per location
        self.hm_conv = nn.Conv2d(32, 17, kernel_size=1, bias=False)


    def forward(self, x):
        x = self.features(x)
        self.feature_output = x
        #x = x.mean(3).mean(2)
        x = self.conv_compress(x)
        x = self.duc1(x)
        x = self.duc2(x)
        x = self.duc3(x)
        #x = self.classifier(x)
        #print(feature.size())
        x = self.hm_conv(x)
        x = dsntnn.flat_softmax(x)
        #print(x.size())
        return self.feature_output, x
        
    def get_feature_output(self):
        return self.feature_output


    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.weight.size(1)
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():                
                need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
            '''
            checkpoint = torch.load(pretrained)
            
            state_dict_old = checkpoint
            state_dict = OrderedDict()
            # delete 'module.' because it is saved from DataParallel module
            for key in state_dict_old.keys():
                if key.startswith('module.'):
                    # state_dict[key[7:]] = state_dict[key]
                    # state_dict.pop(key)
                    state_dict[key] = state_dict_old[key]
                else:
                    state_dict['module.'+key] = state_dict_old[key]
            self.load_state_dict(state_dict, strict=False)
            '''
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


                
                
def get_adaptive_pose_net(cfg, is_train, **kwargs):
    model_p = PoseMobileNet(cfg, **kwargs)
    model_d = DomainClassifier(model_p.get_feature_output())

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        print('ssssssssssssssssssss')
        print(cfg.MODEL.PRETRAINED)
        model_p.init_weights(cfg.MODEL.PRETRAINED)
    
    #for param in model_p.parameters():
     #   param.requires_grad = True
        
    return model_p, model_d

