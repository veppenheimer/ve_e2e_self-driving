#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# 导入torch库
import torch.backends.cudnn as cudnn
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# 导入自定义库
from models import AutoDriveNet
from datasets import AutoDriveDataset
from utils import *


def main():
    """
    训练.
    """
    # 数据集路径
    data_folder = 'data/simulate/data'

    # 学习参数
    checkpoint = None  # 预训练模型路径，如果不存在则为None
    batch_size = 32  # 批大小
    start_epoch = 1  # 轮数起始位置
    epochs = 100                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             # 迭代轮数
    lr = 1e-4  # 学习率

    # 设备参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("使用 GPU:", torch.cuda.get_device_name(0))
    else:
        print("当前使用 CPU")
    ngpu = 1  # 用来运行的gpu数量
    cudnn.benchmark = True  # 对卷积进行加速
    writer = SummaryWriter()  # 实时监控     使用命令 tensorboard --logdir runs  进行查看

    # 初始化模型
    model = AutoDriveNet()


    # 初始化优化器
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad,
                                               model.parameters()),
                                 lr=lr)

    # 迁移至默认设备进行训练
    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    # 加载预训练模型
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # 单机多卡训练
    if torch.cuda.is_available() and ngpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(ngpu)))

    # 定制化的dataloader
    transformations = transforms.Compose([
        transforms.Resize((120, 160)),
        transforms.ToTensor(),  # 通道置前并且将0-255RGB值映射至0-1
        # transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],  # 归一化至[-1,1] mean std 来自imagenet 计算
        #     std=[0.229, 0.224, 0.225])
    ])

    train_dataset = AutoDriveDataset(data_folder,
                                     mode='train',
                                     transform=transformations)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)

    # 开始逐轮训练
    for epoch in range(start_epoch, epochs + 1):

        model.train()  # 训练模式：允许使用批样本归一化
        loss_epoch = AverageMeter()  # 统计损失函数
        n_iter = len(train_loader)

        # 按批处理
        for i, (imgs, labels) in enumerate(train_loader):
            print(f"Batch {i}: imgs.shape = {imgs.shape}")
            # 数据移至默认设备进行训练
            imgs = imgs.to(device)
            labels = labels.to(device)

            # 前向传播
            pre_labels = model(imgs)

            # 计算损失
            loss = criterion(pre_labels, labels)

            # 后向传播
            optimizer.zero_grad()
            loss.backward()

            # 更新模型
            optimizer.step()

            # 记录损失值
            loss_epoch.update(loss.item(), imgs.size(0))

            # 打印结果
            print("第 " + str(i) + " 个batch训练结束")

        # 手动释放内存
        del imgs, labels, pre_labels

        # 监控损失值变化
        writer.add_scalar('MSE_Loss', loss_epoch.avg, epoch)
        print('epoch:' + str(epoch) + '  MSE_Loss:' + str(loss_epoch.avg))

        # 如果 model 使用了 DataParallel，则 model.module 是实际模型，否则直接用 model
        model_state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

        torch.save({
            'model': model_state_dict,
            'optimizer': optimizer.state_dict(),
        }, 've2.pth')

    # 训练结束关闭监控
    writer.close()


if __name__ == '__main__':
    '''
    程序入口
    '''
    main()