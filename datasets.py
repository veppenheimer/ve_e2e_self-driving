#!/usr/bin/env python
# -*- encoding: utf-8 -*-


# 导入系统库
import os
import numpy as np
import cv2
from PIL import Image  # 添加这一行

# 导入PyTorch库
import torch
from torch.utils.data import Dataset
 
 
class AutoDriveDataset(Dataset):
    """
    数据集加载器
    """
 
    def __init__(self, data_folder, mode, transform=None):
        """
        :参数 data_folder: # 数据文件所在文件夹根路径(train.txt和val.txt所在文件夹路径)
        :参数 mode: 'train' 或者 'val'
        :参数 normalize_type: 图像归一化处理方式
        """
 
        self.data_folder = data_folder
        self.mode = mode.lower()
        self.transform = transform
 
        assert self.mode in {'train', 'val'}
 
        # 读取图像列表路径
        if self.mode == 'train':
            file_path=os.path.join(data_folder, 'train.txt')            
        else:
            file_path=os.path.join(data_folder, 'val.txt')
        
        self.file_list=list()      
        with open(file_path, 'r') as f:
            files = f.readlines()
            for file in files:
                if file.strip() is None:
                    continue
                self.file_list.append([file.split(' ')[0],float(file.split(' ')[1])])
                
 
    def __getitem__(self, i):
        """
        :参数 i: 图像检索号
        :返回: 返回第i个图像和标签
        """
        # 读取图像
        img = cv2.imread(self.file_list[i][0])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)
        # 读取标签
        label = self.file_list[i][1]
        label = torch.from_numpy(np.array([label])).float()
        return img, label
 
    def __len__(self):
        """
        为了使用PyTorch的DataLoader,必须提供该方法.
        :返回: 加载的图像总数
        """
        return len(self.file_list)