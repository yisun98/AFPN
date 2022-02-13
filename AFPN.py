# -*- coding:utf-8  -*-
import os
import itertools
from collections import OrderedDict
import torch.nn as nn
import torch
from .base_model import BaseModel
from . import networks

import numpy as np
import cv2
import kornia

import torch

import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.common import *
from torchvision.transforms import *
import torch.nn.functional as F

# -*- coding:utf-8  -*-
import os
import itertools
from collections import OrderedDict
import torch.nn as nn
import torch
from .base_model import BaseModel
from . import networks

import numpy as np
import cv2
import kornia

import torch

# 末尾没用relu的
class Residual_Block(nn.Module):
    def __init__(self, channels):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.PReLU(init=0.5)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        fea = self.relu(self.conv1(x))
        fea = self.conv2(fea)
        result = fea + x
        return result


class SSPN_model(nn.Module):
    def __init__(self, args):
        super(SSPN_model, self).__init__()
        self.args = args
        self.bicubic = networks.bicubic()
        self.block = 2 # SRPPNN 16


        self.conv_mul_pre_p1x = nn.Conv2d(in_channels=args.mul_channel, out_channels=32, kernel_size=3, stride=1,
                                         padding=1)

        self.res_mul_p1x_layer = self.make_layer(Residual_Block, self.block, 32)

        self.conv_mul_post_p1x = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.mul_p1x = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)


        self.conv_mul_pre_p2x = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,
                                          padding=1)

        self.res_mul_p2x_layer = self.make_layer(Residual_Block, self.block, 32)

        self.conv_mul_post_p2x = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)


        self.mul_p2x = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)


        self.conv_mul_pre_p4x = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,
                                          padding=1)

        self.res_mul_p4x_layer = self.make_layer(Residual_Block, self.block, 32)

        self.conv_mul_post_p4x = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.ms_ps_up_4x = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32 * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
        )

        self.mul_p4x = nn.Conv2d(in_channels=32, out_channels=args.mul_channel, kernel_size=3, stride=1, padding=1)


        self.conv_mul_pan_p1x = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,
                                          padding=1)

        self.res_mul_pan_p1x_layer = self.make_layer(Residual_Block, self.block, 32)


        self.conv_mul_pan_p2x = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,
                                          padding=1)

        self.res_mul_pan_p2x_layer = self.make_layer(Residual_Block, self.block, 32)

        self.conv_mul_pan_p4x = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,
                                          padding=1)

        self.res_mul_pan_p4x_layer = self.make_layer(Residual_Block, self.block, 32)


        self.conv_pan_pre_p1x = nn.Conv2d(in_channels=args.pan_channel, out_channels=32, kernel_size=3, stride=1,
                                          padding=1)

        self.res_pan_p1x_layer = self.make_layer(Residual_Block, self.block, 32)

        self.conv_pan_post_p1x = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.pan_ps_down_to_1x_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
        )
        self.pan_ps_down_to_1x_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
        )

        self.conv_pan_pre_p2x = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,
                                          padding=1)

        self.res_pan_p2x_layer = self.make_layer(Residual_Block, self.block, 32)

        self.conv_pan_post_p2x = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.pan_ps_down_to_2x = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
        )



        self.conv_pan_pre_p4x = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,
                                          padding=1)

        self.res_pan_p4x_layer = self.make_layer(Residual_Block, self.block, 32)

        self.conv_pan_post_p4x = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)


        self.pan_modulate1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),


            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),

            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        )
        self.pan_modulate2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),


            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),

            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        )

        self.pan_modulate3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),


            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),

            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        )


    def make_layer(self, block, num_of_layer, channels):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channels))
        return nn.Sequential(*layers)

    def forward(self, x, y):

        inputs_mul_up_p1 = self.bicubic(x, scale=2)
        inputs_mul_up_p2 = self.bicubic(x, scale=4)


        inputs_pan = y
        inputs_pan_blur = kornia.filters.GaussianBlur2d((11, 11), (1, 1))(y)
        inputs_pan_hp = inputs_pan - inputs_pan_blur
        y = inputs_pan_hp




        pre_inputs_mul_p1_feature = self.conv_mul_pre_p1x(x) # 4->32
        x = pre_inputs_mul_p1_feature
        x = self.res_mul_p1x_layer(x)
        post_inputs_mul_p1_feature = self.conv_mul_post_p1x(x) # 32->32
        inputs_mul_p1_feature = pre_inputs_mul_p1_feature + post_inputs_mul_p1_feature


        pre_inputs_pan_p1_feature = self.conv_pan_pre_p1x(y)
        y = pre_inputs_pan_p1_feature
        y = self.res_pan_p1x_layer(y)
        post_inputs_pan_p1_feature = self.conv_pan_post_p1x(y)  # 32->32
        inputs_pan_p1_feature = pre_inputs_pan_p1_feature + post_inputs_pan_p1_feature

        inject_pan_p1_feature_1 = self.pan_ps_down_to_1x_1(inputs_pan_p1_feature)
        inject_pan_p1_feature_2 = self.pan_ps_down_to_1x_1(inject_pan_p1_feature_1)

        g1 = self.pan_modulate1(torch.cat([inputs_mul_p1_feature, inject_pan_p1_feature_2], 1))
        inject_p1 = inputs_mul_p1_feature +  g1 * inject_pan_p1_feature_2

        inject_p1 = self.res_mul_pan_p1x_layer(inject_p1)

        net_up1 = self.bicubic(inject_p1, scale=2)

        net_mp1 = self.mul_p1x(net_up1)

        pre_inputs_mul_p2_feature = self.conv_mul_pre_p2x(net_mp1)  # 4->32
        x = pre_inputs_mul_p2_feature
        x = self.res_mul_p2x_layer(x)
        post_inputs_mul_p2_feature = self.conv_mul_post_p2x(x)  # 32->32
        inputs_mul_p2_feature = pre_inputs_mul_p2_feature + post_inputs_mul_p2_feature

        pre_inputs_pan_p2_feature = self.conv_pan_pre_p2x(inputs_pan_p1_feature)  # 1->32
        y = pre_inputs_pan_p2_feature
        y = self.res_pan_p2x_layer(y)
        post_inputs_pan_p2_feature = self.conv_pan_post_p2x(y)  # 32->32
        inputs_pan_p2_feature = pre_inputs_pan_p2_feature + post_inputs_pan_p2_feature

        inject_pan_p2_feature = self.pan_ps_down_to_2x(inputs_pan_p2_feature) + inject_pan_p1_feature_1

        g2 = self.pan_modulate2(torch.cat([inputs_mul_p2_feature, inject_pan_p2_feature], 1))
        inject_p2 = inputs_mul_p2_feature + g2 * inject_pan_p2_feature
        inject_p2 = self.res_mul_pan_p2x_layer(inject_p2)

        net_up2 = self.bicubic(inject_p2, scale=2)

        net_mp2 = self.mul_p2x(net_up2)


        pre_inputs_mul_p4_feature = self.conv_mul_pre_p4x(net_mp2)  # 4->32
        x = pre_inputs_mul_p4_feature
        x = self.res_mul_p4x_layer(x)
        post_inputs_mul_p4_feature = self.conv_mul_post_p4x(x)  # 32->32
        inputs_mul_p4_feature = pre_inputs_mul_p4_feature + post_inputs_mul_p4_feature

        pre_inputs_pan_p4_feature = self.conv_pan_pre_p4x(inputs_pan_p2_feature)  # 1->32
        y = pre_inputs_pan_p4_feature
        y = self.res_pan_p4x_layer(y)
        post_inputs_pan_p4_feature = self.conv_pan_post_p4x(y)  # 32->32
        inputs_pan_p4_feature = pre_inputs_pan_p4_feature + post_inputs_pan_p4_feature

        inject_pan_p4_feature = inputs_pan_p4_feature + inputs_pan_p2_feature

        g3 = self.pan_modulate3(torch.cat([inputs_mul_p4_feature, inject_pan_p4_feature], 1))

        inject_p4 = inputs_mul_p4_feature + g3 * inject_pan_p4_feature

        inject_p4 = self.res_mul_pan_p4x_layer(inject_p4)

        net_mp4 = self.mul_p4x(inject_p4) + inputs_mul_up_p2

        return net_mp4





