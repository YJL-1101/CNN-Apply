import copy
import time

from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt

from get_dataset import train_loader
from model import GoogLeNet,Inception
import torch
from torch import nn
import pandas as pd
#数据（训练集，测试集）处理
def data_process():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.486, 0.453, 0.415), (0.263, 0.256, 0.259))
    ])
    data = ImageFolder(root='./data/train',transform=transform)
    #分成训练集和验证集
    train_data,val_data = Data.random_split(data,[round(0.8*len(data)),round(0.2*len(data))])

    #加载
    train_data_loader = Data.DataLoader(dataset=train_data,
                                   batch_size=128,
                                   shuffle=True,
                                   num_workers=0)


    val_data_loader = Data.DataLoader(dataset=val_data,
                                   batch_size=128,
                                   shuffle=True,
                                   num_workers=0)

    return train_data_loader,val_data_loader

def train_model_process(model,train_data_loader,val_data_loader,num_epochs):
    #选择训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #定义优化器（梯度下降的方法）
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    #定义损失函数
    criterion =  nn.CrossEntropyLoss()
    #模型放入训练设备中
    model = model.to(device)
    #复制当前模型参数
    best_model_wts = copy.deepcopy(model.state_dict())


    #初始化参数
    #准确率
    best_acc = 0.0
    #记录训练损失
    train_loss_all = []
    #记录验证损失
    val_loss_all = []
    #记录训练准确率
    train_acc_all = []
    #记录验证准确率
    val_acc_all = []

    #记录当前时间,用于计算训练的时间
    since = time.time()

    #训练
    for epoch in range(num_epochs):
        print("epoch {}/{}".format(epoch+1,num_epochs))
        print('-'*10)

        #初始化参数
        train_loss = 0.0
        val_loss = 0.0
        train_acc = 0.0
        val_acc =  0.0
        #训练集样本数量
        train_num = 0
        #验证集样本数量
        val_num = 0

        #对每一个小批次进行训练，b_x数据特征，b_y标签
        for step,(b_x,b_y) in enumerate(train_data_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            #将模型设置为训练模式
            model.train()

            #前向传播
            output = model(b_x)
            #找出预测类别结果（找最大值）
            pre_lab = torch.argmax(output,dim=1)
            #损失
            loss = criterion(output,b_y)
            #梯度置为0
            optimizer.zero_grad()
            #误差反向传播
            loss.backward()
            #更新网络参数
            optimizer.step()
            #累加损失（该批次的累加loss）
            train_loss += loss.item()*b_x.size(0)
            #统计预测正确的个数
            train_acc +=torch.sum(pre_lab == b_y)
            #统计当前训练样本数量
            train_num+=b_x.size(0)
        #验证过程（训练过程去除反向传播）
        for step, (b_x, b_y) in enumerate(val_data_loader):

            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()
            output = model(b_x)

            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)

            # 累加损失
            val_loss += loss.item() * b_x.size(0)
            # 统计预测正确的个数
            val_acc += torch.sum(pre_lab == b_y)
            # 统计当前验证样本数量
            val_num += b_x.size(0)

        #该轮次（epoch）的平均loss值
        train_loss_all.append(train_loss / train_num)
        val_loss_all.append(val_loss / val_num)
        # 该轮次（epoch）的平均acc值
        #.double().item()可以将tensor转为python数字
        train_acc_all.append(train_acc.double().cpu().item() / train_num)
        val_acc_all.append(val_acc.double().cpu().item() / val_num)

        print("{} train loss:{:.4f} train acc: {:.4f}".format(epoch+1, train_loss_all[-1], train_acc_all[-1]))
        print("{} val loss:{:.4f} val acc: {:.4f}".format(epoch+1, val_loss_all[-1], val_acc_all[-1]))


        if val_acc_all[-1] > best_acc:
            #保存最高准确度
            best_acc = val_acc_all[-1]
            #保存最高权重参数
            #model.state_dict()返回模型中的所有可学习参数
            #copy.deepcopy()真正复制一份独立参数
            best_model_wts = copy.deepcopy(model.state_dict())

        #训练耗费时间
        time_use = time.time()-since
        print("训练和验证耗费时间{:.0f}min{:.0f}s".format(time_use//60,time_use%60))

    torch.save(best_model_wts,"./best_model.pth")

    #把训练过程中每一轮 epoch 的损失和准确率整理成一个表格
    train_process = pd.DataFrame(data={
        "epoch": range(num_epochs),
        "train_loss_all": train_loss_all,
        "val_loss_all": val_loss_all,
        "train_acc_all": train_acc_all,
        "val_acc_all": val_acc_all,
    })
    return  train_process


#把训练过程中的 loss 和 accuracy 画成两张图
def matplot_acc_loss(train_process,save_path="./train_acc_loss.png"):
    # 显示每一次迭代后的训练集和验证集的损失函数和准确率
    plt.figure(figsize=(12, 4))

    # 第一张子图：loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")

    # 第二张子图：accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()

    # 保存图片
    plt.savefig(save_path)
    print(f"训练曲线已保存到 {save_path}")

    plt.show()

if __name__ == "__main__":
    #模型实例化
    GoogLeNet = GoogLeNet(Inception)
    #加载数据
    train_data_loader,val_data_loader = data_process()
    #训练模型
    train_process = train_model_process(GoogLeNet,train_data_loader,val_data_loader,20)
    #画图
    matplot_acc_loss(train_process,save_path="./GoogLeNet_train_acc_loss.png")