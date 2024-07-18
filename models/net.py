import torch
from torch import nn


# 定义神经网络模型（Modified LeNet）
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, 5, 1, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)

        # 池化层
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)  # 增加一个池化层

        # 全连接层
        self.fc1 = nn.Linear(3 * 3 * 128, 1024)  # 调整输入维度以匹配新的卷积层输出
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool2(x)
        x = nn.functional.relu(self.conv3(x))
        x = self.pool3(x)

        # 展平特征图
        x = x.view(-1, 3 * 3 * 128)

        # 全连接层
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, p=0.5, training=self.training)  # 添加dropout以防止过拟合
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        # 返回对数概率
        return nn.functional.log_softmax(x, dim=1)