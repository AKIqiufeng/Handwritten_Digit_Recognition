# 导入所需的库
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torch.optim as optim
from models.net import *
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("\033[1;37;42m CUDA ON \033[0m")
else:
    print("\033[31m CUDA OFF\033[0m")
# 设置随机种子，以确保结果的可重复性
torch.manual_seed(0)


class RandomTranslatePixel(object):
    def __init__(self):
        pass

    def __call__(self, image):
        # 将PIL图像转换为numpy数组
        image_np = np.array(image)
        # 随机生成平移量，-1, 0, 1 表示向左、不动、向右平移一个像素
        dx, dy = np.random.choice([-1, 0, 1], size=2)

        # 使用roll函数进行平移（注意：这里只处理了简单情况，对于边缘像素可能不是最佳处理）
        shifted_image_np = np.roll(image_np, (dx, dy), axis=(0, 1))

        # 将numpy数组转换回PIL图像
        shifted_image = Image.fromarray(shifted_image_np)

        return shifted_image

    # 定义数据预处理，包括自定义的随机平移


# 定义数据预处理
transform = transforms.Compose([
    # transforms.ToPILImage(),  # 如果输入已经是Tensor，则需要先转换为PIL Image
    RandomTranslatePixel(),  # 应用自定义的随机平移
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化图像
])

# 加载MNIST数据集
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)  # 训练数据集
test_dataset = datasets.MNIST('data', train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]))  # 测试数据集

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)  # 训练数据加载器
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)  # 测试数据加载器

# 创建神经网络模型实例
model = Net().to(device)

# 定义优化器和损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss().to(device)

# 定义储存路径，添加tensorboard
save_dir = './LeNet'
writer = SummaryWriter("{}/logs".format(save_dir))

# 创建储存文件夹
if not os.path.exists('./LeNet/epoch'):
    os.makedirs('./LeNet/epoch')


# 训练模型
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_step = len(train_loader) * (epoch - 1) + batch_idx + 1
        writer.add_scalar("train_loss", loss.item(), train_step)

        if batch_idx % 100 == 0:
            print('训练 Epoch: {} [{:05d}/{} ({:2.0f}%)]\t损失: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# 测试模型
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\n测试集: 总损失: {:.4f}, 准确率: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    writer.add_scalar("test_loss", test_loss, epoch)
    writer.add_scalar("test_accuracy", correct / len(test_loader.dataset), epoch)

    torch.save(model.state_dict(), "{}/epoch/module_{}.pth".format(save_dir, epoch))
    print("saved epoch {}\n".format(epoch))


# 进行模型训练和测试
for epoch in range(1, 31):
    print(f"\033[34m-------epoch  {epoch}-------\033[0m")
    train(epoch)
    test()
writer.close()

torch.save(model.state_dict(), f'{save_dir}/mnist_model.pth')
print(f"\033[36mSave Model\033[0m")
