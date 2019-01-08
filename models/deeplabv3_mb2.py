"""

model combined mobilenetv2 with deeplab3
"""
import torch
import torch.nn as nn

from .mobilenet_v2 import get_inverted_residual_blocks, InvertedResidual
from .asppplus import ASPPPlus

class DeepLabV3MobileNetV2(nn.Module):

    def __init__(self, num_classes):
        super(DeepLabV3MobileNetV2, self).__init__()
        
        # same config params in MobileNetV2
        # each layer channel
        self.c = [32, 16, 24, 32, 64, 96, 160]
        # each layer expansion times
        self.t = [1, 1, 6, 6, 6, 6, 6] 
        # each layer expansion stride
        self.s = [2, 1, 2, 2, 2, 1, 1]
        # each layer repeat time
        self.n = [1, 1, 2, 3, 4, 3, 3]
        self.down_sample_rate = 32
        self.output_stride = 16
        self.multi_grid = (1, 2, 4)
        self.aspp = (6, 12, 18)


        # all blocks goes here
        self.blocks = []
        # build MobileNetV2 backbone first
        self.blocks.append(
            nn.Sequential(
                nn.Conv2d(3, self.c[0], 3, stride=self.s[0], padding=1, bias=False),
                nn.BatchNorm2d(self.c[0]),
                nn.ReLU6()           
            )
        )
        
        
        for i in range(6):
            self.blocks.extend(
                get_inverted_residual_blocks(self.c[i], self.c[i+1], t=self.t[i+1], s=self.s[i+1], n=self.n[i+1])
            )
        
        # dilated conv
        rate = self.down_sample_rate // self.output_stride
        self.blocks.append(InvertedResidual(self.c[6], self.c[6], expand_ratio=self.t[6], stride=1, dilation=rate))
        for i in range(3):
            self.blocks.append(InvertedResidual(self.c[6], self.c[6], expand_ratio=self.t[6], stride=1, dilation=rate*self.multi_grid[i]))
        # append ASPP layer
        self.blocks.append(ASPPPlus(self.c[-1], self.aspp))

        # last conv layer
        self.blocks.append(nn.Conv2d(256, num_classes, 1))
        self.blocks.append(nn.Upsample(scale_factor=self.output_stride, mode='bilinear', align_corners=False))
        self.model = nn.Sequential(*self.blocks)
    
    def forward(self, x):
        return self.model(x)