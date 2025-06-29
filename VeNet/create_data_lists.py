#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# 导入系统库
import os
import random


def creat_data_list(dataset_path, file_list, mode='train'):
    '''
    创建txt文件列表
    '''
    with open(os.path.join(dataset_path, (mode + '.txt')), 'w') as f:
        for (imgpath, angle) in file_list:
            f.write(imgpath + ' ' + str(angle) + '\n')
    print(mode + '.txt 已生成')


def getFileList(dir, Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir: 文件夹根目录
    输入 ext: 扩展名
    返回: 文件路径列表
    """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist


def main():
    '''
    主函数
    '''
    # 设置参数
    org_img_folder = './data/1_1'  # 数据集根目录
    train_ratio = 0.8  # 训练集占比

    # 检索jpg文件
    jpglist = getFileList(org_img_folder, [], 'jpg')
    print('本次执行检索到 ' + str(len(jpglist)) + ' 个jpg文件\n')

    file_list = list()
    # 解析转向值
    for jpgpath in jpglist:
        print(jpgpath)
        curDataDir = os.path.dirname(jpgpath)
        basename = os.path.basename(jpgpath)
        angle = (basename[:-4]).split('_')[-1]
        imgPath = os.path.join(curDataDir, basename).replace("\\", "/")
        file_list.append((imgPath, angle))

    # 切分数据
    random.seed(256)
    random.shuffle(file_list)
    train_num = int(len(file_list) * train_ratio)
    train_list = file_list[0:train_num]
    val_list = file_list[train_num:]

    # 创建列表文件
    creat_data_list(org_img_folder, train_list, mode='train')
    creat_data_list(org_img_folder, val_list, mode='val')


if __name__ == "__main__":
    '''
    程序入口
    '''
    main()