from abc import ABC

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import random
from blocks import *


class NeuralNet(nn.Module, ABC):
    def __init__(self, list_blocks, initial_image_size,
                 total_classes, number_input_channels, dropout):
        super().__init__()
        self.__list_blocks = list_blocks
        self.__initial_image_size = initial_image_size
        self.__total_classes = total_classes
        self.__nb_input_channels = number_input_channels
        self.dropout = dropout
        self.features = None
        self.classifier = None
        self.last_channel = None
        self.expand_coeff = 1.5
        self.squeeze_coeff = 0.75

        # check that the ids are correct
        for i in range(len(list_blocks)):
            assert list_blocks[i] in range(-1, 10), 'Block id must be between -1 and 9'

        self.build_network()

    @property
    def list_blocks(self):
        return self.__list_blocks

    @list_blocks.setter
    def list_blocks(self, lb):
        self.__list_blocks = lb

    @property
    def list_connexions(self):
        return self.__list_connexions

    @property
    def nb_classes(self):
        return self.__total_classes

    @property
    def initial_image_size(self):
        return self.__initial_image_size

    def build_network(self):

        # clean list: remove all -1 occurences
        self.list_blocks = [y for y in self.list_blocks if y != -1]

        n = len(self.list_blocks)
        print('Number of blocks:', n)
        input_channel = 64  # 148

        self.features = [nn.Conv2d(3, input_channel, kernel_size=3, stride=2),
                         nn.ReLU6(inplace=True),
                         nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)]

        for i in range(n):
            block = None
            size_output = None
            block_type = self.list_blocks[i]

            if block_type == 0:
                # VGG block that reduces the nb of channels
                coeff = self.squeeze_coeff
                block = VGGConv(input_channel, coeff)
                size_output = int(coeff * coeff * input_channel)

            if block_type == 1:
                # VGG block that increases the nb of channels
                coeff = self.expand_coeff
                block = VGGConv(input_channel, coeff)
                size_output = int(coeff * coeff * input_channel)

            if block_type == 2:
                # Buld a Dense block
                # 32, (6, 12, 32, 32), 64
                growth_rate = 32
                bn_size = 4
                drop_rate = self.dropout

                dense_block = DenseBlock(num_layers=6,
                                         num_input_features=input_channel,
                                         bn_size=bn_size,
                                         growth_rate=growth_rate,
                                         drop_rate=drop_rate,
                                         memory_efficient=True)
                size_output = input_channel + 6 * growth_rate

                trans = Transition(num_input_features=size_output,
                                   num_output_features=size_output // 2)
                block = nn.Sequential(dense_block, trans)

                size_output = size_output // 2

            if block_type == 3:
                # Inverted residual
                # 10 24 5 2 3
                # t, c, n, s, k
                # width multi = 1.75
                size_output = int(24 * self.expand_coeff)
                block = InvertedResidual(input_channel, size_output, 2, 10, 3)

            if block_type == 4:
                #  148 33 1182 1
                # Fire is a bottleneck
                block = nn.Sequential(Fire(input_channel, 33, 591, 591),
                                      nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
                size_output = 1182

            if block_type == 5:
                # Inception block
                block = Inception(input_channel, 54, 86, 138, 21, 32, 32)
                # output = ch1x1 + ch3x3 + ch5x5 + pool_proj
                size_output = 256

            if block_type == 6:
                # Squeeze-and-excite
                squeeze_ratio = 0.2
                se_channel = int(input_channel * squeeze_ratio)
                block = SE(input_channel, se_channel)
                size_output = input_channel

            if block_type == 7:
                # Fire - SE - BasicConv
                block = nn.Sequential(Fire(input_channel, 33, 591, 591),
                                      nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                                      SE(1182, 236),
                                      BasicConv(1182, 512))
                size_output = 512

            if block_type == 8:
                # From EfficientNet - B0
                # output was 178
                block = EfficientBlock(input_channel,
                                       512,
                                       3,
                                       2,
                                       6,
                                       se_ratio=0.25,
                                       drop_rate=self.dropout)
                size_output = 512

            if block_type == 9:
                block = SharpSepConv(input_channel, input_channel * 2, 3, 2)
                size_output = input_channel * 2
            self.features.append(block)
            input_channel = size_output

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        final_conv = nn.Conv2d(input_channel, self.nb_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            final_conv,
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = self.classifier(x)
        return torch.flatten(x, 1)


if __name__ == '__main__':
    list_blocks = [0, 2, 3]
    list_connexions = []
    initial_image_size = 32
    total_classes = 10
    number_input_channels = 3

    net = NeuralNet(list_blocks, initial_image_size,
                    total_classes, number_input_channels)
    print(net)

