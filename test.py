"""
读取模型及预测
"""

import torch
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PIL import Image
from torchvision.transforms import functional as F
from train import TrashNet
from matplotlib import pyplot as plt


def reg(path):
    # raw = Image.open('ga2/test/trash/trash37.jpg').convert('RGB')
    # 读取图片
    raw = Image.open(path).convert('RGB')
    # 图片变换
    new = F.resize(raw, size=[32, 32])
    # 转变为张量
    new = F.to_tensor(new)
    # 均值与方差
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    new = F.normalize(new, norm_mean, norm_std)
    new = new.expand(1, 3, 32, 32)
    # 加载模型
    model = TrashNet(classes=3)
    model.load_state_dict(torch.load('trash200.pkl'))
    # 识别
    out = model(new)
    _, pred = torch.max(out, dim=1)
    rs = pred.data.item()
    msg = ''
    if rs == 0:
        msg = '纸垃圾'
    if rs == 1:
        msg = '塑料垃圾'
    if rs == 2:
        msg = '生活垃圾'

    # print(msg)
    plt.imshow(raw)
    plt.show()
    return msg


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    path_img, _ = QFileDialog.getOpenFileName(None, '打开图片文件', './',
                                              'Image Files(*.png *.jpg *.bmp *.jpeg);;ALL(*.*)')

    print(reg(path_img))
