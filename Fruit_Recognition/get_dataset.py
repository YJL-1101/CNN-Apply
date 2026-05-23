import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import ImageFolder
# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.67597389,0.57529323,0.51096168), (0.29731594,0.35303732,0.38785615))
])

# 加载数据集
train_data = ImageFolder(root='./data/train',transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# 获得一个Batch的数据
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
batch_x = b_x.permute(0, 2, 3, 1).numpy()  # 调整为 (B,H,W,C) 供 imshow 使用
batch_x = batch_x * 0.5 + 0.5  # 反归一化: [-1,1] -> [0,1]
batch_y = b_y.numpy()  # 将张量转换成Numpy数组
class_label = train_data.classes  # 训练集的标签
# print(class_label)
print("The size of batch in train data:", batch_x.shape)  # 每个mini-batch的维度是64*227*227*3

# 可视化一个Batch的图像
plt.figure(figsize=(12, 5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4, 16, ii + 1)
    plt.imshow(batch_x[ii, :, :, :])
    plt.title(class_label[batch_y[ii]], size=10)
    plt.axis("off")
    plt.subplots_adjust(wspace=0.05)
plt.show()