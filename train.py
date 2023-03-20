"""
模型训练及保存
"""

import os
import random
import shutil
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

data_path = 'ga'
train_test_path = 'ga2'


# 文件分割
def file_split():
    for root, child, files in os.walk(data_path):
        for ch in child:
            childPath = os.path.join(root, ch)
            # 遍历目录下所有文件
            fname = os.listdir(childPath)
            # 打乱文件次序,提升数据随机性
            random.shuffle(fname)
            # 文件数量
            cfiles = len(fname)
            # 分配训练集，测试集比例
            train = cfiles * 0.9
            # test = cfiles * 0.1
            for index, f in enumerate(fname):
                img_name = os.path.join(childPath, f)
                # 标签
                label = childPath.split('\\')[-1]
                if index < train:
                    dataPath = train_test_path + '\\train\\' + label
                else:
                    dataPath = train_test_path + '\\test\\' + label

                # 创建文件夹
                os.makedirs(dataPath, exist_ok=True)
                dataPath = dataPath + '\\' + f
                shutil.copy(img_name, dataPath)


# 图片标注
class DataUtils(Dataset):
    # 图片标注
    def __init__(self, path='', transform=None):
        # 储存图片名称和所属种类
        self.img_info = []
        for root, child, files in os.walk(path):
            for file in files:
                # 所有图片的路径
                fname = os.path.join(root, file)
                # 图片的标签
                label = fname.split('\\')[2]
                label2 = 0
                if label == 'paper':
                    label2 = 0
                if label == 'plastic':
                    label2 = 1
                if label == 'trash':
                    label2 = 2
                # 对图片进行标注
                self.img_info.append([fname, label2])
        # print(self.img_info)
        self.transform = transform

    # 输出变换后的图片
    def __getitem__(self, item):
        # 取出每一张图片
        fname, label = self.img_info[item]
        img = Image.open(fname)
        # 图片转换
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    # 图片数量
    def __len__(self):
        return len(self.img_info)


# 卷积神经网络模型
class TrashNet(torch.nn.Module):
    def __init__(self, classes):
        super(TrashNet, self).__init__()
        # 卷积层：输入数据有3个特征,输出数据有6个特征
        # 卷积核宽高为5
        self.conv1 = nn.Conv2d(3, 6, (5, 5))

        # 第二层卷积
        self.conv2 = nn.Conv2d(6, 16, (5, 5))

        # 第三层,全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)

        # 第四层,全连接层
        self.fc2 = nn.Linear(120, 84)

        # 第五层,输出层
        self.fc3 = nn.Linear(84, classes)

    def forward(self, x):
        # 对第一层卷积计算的结果使用relu激活函数
        out = torch.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        # 输出图片的类别
        out = self.fc3(out)
        return out


# 建模
def fit(times):
    # 图片大小
    Image_size = 32
    # 方差与均值
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    # 图片变换(旋转，伸缩)器
    transformer = transforms.Compose([
        transforms.Resize((Image_size, Image_size)),
        transforms.RandomGrayscale(p=0.9),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    train = DataUtils(path="ga2\\train", transform=transformer)
    # 读取数据
    trainloader = DataLoader(
        dataset=train,
        batch_size=20,
        shuffle=True,
        drop_last=True
    )

    # 建模
    model = TrashNet(classes=3)
    # 损失函数
    lossfn = nn.CrossEntropyLoss()
    # lr学习率，越小越精确，但是对性能影响较高
    opt = SGD(model.parameters(), lr=0.1)

    for e in range(times):
        for batch in trainloader:
            img, label = batch
            out = model(img)
            lossval = lossfn(out, label)
            opt.zero_grad()
            lossval.backward()
            opt.step()

    # 保存模型
    # torch.save(model.state_dict(), f'trash{times}.pkl')


if __name__ == '__main__':
    # 划分训练集
    if Path(train_test_path).exists():
        a = input("测试训练集已存在，是否重新划分？（y/n）")
        if a == 'y':
            print("开始重新划分")
            shutil.rmtree(Path(train_test_path))
            file_split()
            print("划分完毕")
    else:
        file_split()
        print("划分完毕")

    # 模型创建训练及保存
    fit(200)  # 训练200次
