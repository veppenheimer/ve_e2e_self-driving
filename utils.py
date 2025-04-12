#!/usr/bin/env python
# -*- encoding: utf-8 -*-

class AverageMeter(object):
    '''
    平均器类,用于计算平均值、总和
    '''
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
        self.avg = self.sum / self.count