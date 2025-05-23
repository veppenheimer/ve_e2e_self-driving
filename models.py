#!/usr/bin/env python
# -*- encoding: utf-8 -*-


# 导入PyTorch库
import torch.nn as nn
import torch.nn.functional as F


class AutoDriveNet(nn.Module):
    '''
    端到端自动驾驶模型
    '''

    def __init__(self):
        """
        初始化
        """
        super(AutoDriveNet, self).__init__()
        self.conv_layers = nn.Sequential(nn.Conv2d(3, 24, 5, stride=2),
                                         nn.ELU(),
                                         nn.Conv2d(24, 36, 5, stride=2),
                                         nn.ELU(),
                                         nn.Conv2d(36, 48, 5, stride=2),
                                         nn.ELU(), nn.Conv2d(48, 64, 3),
                                         nn.ELU(), nn.Conv2d(64, 64, 3),
                                         nn.Dropout(0.5))
        self.linear_layers = nn.Sequential(
            #nn.Linear(in_features=64 * 2 * 33, out_features=100),
            nn.Linear(in_features=64 * 8 * 13, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1))

    def forward(self, input):
        '''
        前向推理
        '''
        input = input.view(input.size(0), 3, 120, 160)
        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output