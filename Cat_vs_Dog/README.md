# Cat vs Dog — GoogLeNet 图像二分类应用

使用 PyTorch 实现的 GoogLeNet（Inception v1）卷积神经网络，在 Kaggle Cat vs Dog 数据集上进行猫狗图像二分类任务。

## 项目简介

本项目使用 GoogLeNet 经典卷积神经网络架构，对猫和狗的图像进行二分类识别。与经典 GoogLeNet 将最后一层输出改为 2 类（Cat / Dog），并基于数据集自行计算了适用于该数据集的均值和标准差进行归一化处理。数据集来自 Kaggle 上的 [Cat vs Dog Dataset](https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset)。

### GoogLeNet 核心：Inception 模块

Inception 模块包含四条并行的路径，最后在通道维度拼接：

| 路径  | 结构                | 作用             |
| --- | ----------------- | -------------- |
| 路径1 | 1×1 卷积            | 跨通道信息交互与降维     |
| 路径2 | 1×1 卷积 → 3×3 卷积   | 先降维再进行中等尺度特征提取 |
| 路径3 | 1×1 卷积 → 5×5 卷积   | 先降维再进行大尺度特征提取  |
| 路径4 | 3×3 最大池化 → 1×1 卷积 | 池化后降维，保留空间信息   |

### 网络结构

| 层       | 类型                      | 输入通道 | 输出通道 / 配置                    |
| ------- | ----------------------- | ---- | ---------------------------- |
| Block 1 | Conv 7×7 (s=2) + ReLU   | 3    | 64                           |
| <br />  | MaxPool 3×3 (s=2)       | —    | (224→112)                    |
| Block 2 | Conv 1×1 + ReLU         | 64   | 64                           |
| <br />  | Conv 3×3 + ReLU         | 64   | 192                          |
| <br />  | MaxPool 3×3 (s=2)       | —    | (112→56)                     |
| Block 3 | Inception (3a)          | 192  | 64 + 128 + 32 + 32 = 256     |
| <br />  | Inception (3b)          | 256  | 128 + 192 + 96 + 64 = 480    |
| <br />  | MaxPool 3×3 (s=2)       | —    | (56→28)                      |
| Block 4 | Inception (4a)          | 480  | 192 + 208 + 48 + 64 = 512    |
| <br />  | Inception (4b)          | 512  | 160 + 224 + 64 + 64 = 512    |
| <br />  | Inception (4c)          | 512  | 128 + 256 + 64 + 64 = 512    |
| <br />  | Inception (4d)          | 512  | 112 + 288 + 64 + 64 = 528    |
| <br />  | Inception (4e)          | 528  | 256 + 320 + 128 + 128 = 832  |
| <br />  | MaxPool 3×3 (s=2)       | —    | (28→14)                      |
| Block 5 | Inception (5a)          | 832  | 256 + 320 + 128 + 128 = 832  |
| <br />  | Inception (5b)          | 832  | 384 + 384 + 128 + 128 = 1024 |
| 输出层     | AdaptiveAvgPool2d (1×1) | —    | 1024                         |
| <br />  | Flatten                 | —    | 1024                         |
| <br />  | Dropout (0.4)           | —    | —                            |
| <br />  | Linear                  | 1024 | **2**                        |

> 与原版 GoogLeNet 的区别：最后一层全连接输出为 2（Cat / Dog），而非原版的 1000 类。

权重初始化使用 Kaiming 正态分布，激活函数使用 ReLU。输入图像尺寸为 224×224 彩色图。

## 项目结构

```
CNN-Apply/
└── Cat_vs_Dog/
    ├── data_partitioning.py         # 数据集划分脚本（从 PetImages 划分训练/测试集）
    ├── mean_std.py                  # 计算数据集均值和标准差
    ├── model.py                     # GoogLeNet网络结构定义（含 Inception 模块）
    ├── model_train.py               # 训练脚本（含训练/验证曲线绘制）
    ├── model_test.py                # 测试脚本（批量测试 & 单张预测）
    ├── model_inference.py           # 单张图片推理脚本
    ├── get_dataset.py               # 数据集加载与可视化
    ├── best_model.pth               # 训练保存的最优模型权重
    ├── GoogLeNet_train_acc_loss.png # 训练过程的 Loss 和 Accuracy 曲线
    ├── cat.png                      # 推理测试用猫图片
    ├── dog.png                      # 推理测试用狗图片
    └── data/                        # 数据集目录
        ├── train/
        │   ├── Cat/
        │   └── Dog/
        └── test/
            ├── Cat/
            └── Dog/
```

## 环境依赖

- torch==1.10.1
- torchsummary==1.5.1
- numpy==1.23.2
- pandas==1.3.4
- matplotlib==3.5.0

安装依赖：

```bash
pip install -r ../requirements.txt
```

## 使用方法

### 1. 准备数据集

从 Kaggle 下载 [Cat vs Dog Dataset](https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset)，将解压后的 `PetImages` 文件夹放在 `Cat_vs_Dog/` 目录下。

### 2. 划分数据集

将原始数据按 9:1 的比例划分为训练集和测试集：

```bash
python data_partitioning.py
```

### 3. 计算数据集均值与标准差（可选）

计算数据集的均值和标准差，用于归一化预处理：

```bash
python mean_std.py
```

> 本项目已预先计算好：mean=(0.486, 0.453, 0.415)，std=(0.263, 0.256, 0.259)

### 4. 训练模型

```bash
python model_train.py
```

训练过程会自动：

- 加载 `./data/train/` 下的训练数据
- 按 8:2 划分训练集和验证集
- 训练 20 个 epoch，使用 Adam 优化器（lr=0.001），Batch Size=128
- 保存最优模型权重到 `best_model.pth`
- 绘制训练/验证的 Loss 和 Accuracy 曲线，保存为 `GoogLeNet_train_acc_loss.png`

### 5. 测试模型

```bash
python model_test.py
```

支持两种测试模式：

- `test_model_process`：批量测试，输出整体准确率
- `test_model_process_v2`：逐张预测，打印每张图片的预测结果与真实标签

### 6. 单张图片推理

```bash
python model_inference.py
```

使用训练好的模型对单张图片（默认为 `dog.png`）进行推理，输出预测类别。可以替换 `image_process('./dog.png')` 中的路径来推理其他图片。

### 7. 可视化数据集

```bash
python get_dataset.py
```

展示训练集的一个 batch 图像，查看数据加载效果。

## 数据集

[Kaggle Cat vs Dog Dataset](https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset) — 包含猫和狗两类动物的彩色图像：

| 标签 | 类别  |
| -- | --- |
| 0  | Cat |
| 1  | Dog |

数据集通过 `data_partitioning.py` 按 9:1 划分训练集与测试集，训练时再按 8:2 划分训练集与验证集。所有图像统一 Resize 至 224×224 以适配 GoogLeNet 输入层。

## 参考

- Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., & Rabinovich, A. (2015). Going Deeper with Convolutions.
- [Kaggle Cat vs Dog Dataset](https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset)
- [Classic-CNN-PyTorch](https://github.com/YJL-1101/Classic-CNN-PyTorch) — 经典 CNN 模型实现参考

