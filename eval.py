#!/usr/bin/env python
# -*- encoding: utf-8 -*-


# 导入系统库
import time

# 导入PyTorch库
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

# 导入自定义库
from datasets import AutoDriveDataset
from models import AutoDriveNet
from utils import *


def main():
    # 测试集目录
    data_folder = "./data/simulate"
    
    # 定义运行的GPU数量
    ngpu = 1
    
    #cudnn.benchmark = True
    
    # 定义设备运行环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    # 加载预训练模型
    checkpoint = torch.load("checkpoint.pth")
    model = AutoDriveNet()
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])
 
    # 多GPU封装
    if torch.cuda.is_available() and ngpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(ngpu)))
   
    # 定制化的dataloader
    transformations = transforms.Compose([
        transforms.ToTensor(),  # 通道置前并且将0-255RGB值映射至0-1
        # transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],  # 归一化至[-1,1] mean std 来自imagenet 计算
        #     std=[0.229, 0.224, 0.225])
    ])
    val_dataset = AutoDriveDataset(data_folder,
                                     mode='val',
                                     transform=transformations
                                     )
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1,
                                            pin_memory=True)
    
    # 定义评估指标
    criterion = nn.MSELoss().to(device)

    # 记录误差值
    MSEs = AverageMeter()

    # 记录测试时间
    model.eval()
    start = time.time()

    with torch.no_grad():
        # 逐批样本进行推理计算
        for i, (imgs, labels) in enumerate(val_loader):
            
            # 数据移至默认设备进行推理
            imgs = imgs.to(device)
            labels = labels.to(device)   

            # 前向传播
            pre_labels = model(imgs)

            # 计算误差
            loss = criterion(pre_labels, labels)     
            MSEs.update(loss.item(), imgs.size(0))
            
    # 输出平均均方误差
    print('MSE  {mses.avg: .3f}'.format(mses=MSEs))
    print('平均单张样本用时  {:.3f} 秒'.format((time.time()-start)/len(val_dataset)))

 
if __name__ == '__main__':
    '''
    程序入口
    '''
    main()
    