# Fruit Recognition — GoogLeNet 11 类水果分类

使用 PyTorch 实现的 GoogLeNet（Inception v1）卷积神经网络，在 Fruits-360 子集上进行 11 种水果的图像分类任务。

## 项目简介

本项目从 [Fruits-360](https://github.com/fruits-360) 数据集中精选 11 种常见水果（Apple、Banana、Cherry、Grape、Lemon、Orange、Peach、Pear、Raspberry、Strawberry、Watermelon），将每个类别下的所有品种合并后均匀采样，构建了一个**类别间完全均衡、品种内充分多样**的子数据集。模型采用 GoogLeNet 经典架构，输出层改为 11 类，并基于数据集自行计算了均值和标准差进行归一化处理。

### 数据集处理流程

原始 Fruits-360 共 260 个品种目录、182,945 张图片，且相同水果的不同品种被存储为独立类别。本项目通过以下步骤构建均衡子集：

1. **正则分组**：将 `Apple 10`、`Apple 11` 等品种目录按基础名合并为 11 个大类
2. **品种均分采样**：每类的图片总数取所有类别中最小的值（Watermelon = 475 张），各品种均匀分配配额
3. **打乱与划分**：全局随机 shuffle 后按 8:2 划分为训练集和测试集
4. **归一化参数计算**：遍历全部 5,225 张图片，计算 RGB 三通道的均值与标准差

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
| <br />  | Linear                  | 1024 | **11**                       |

> 与原版 GoogLeNet 的区别：最后一层全连接输出为 11（11 种水果），而非原版的 1000 类。

权重初始化使用 Kaiming 正态分布，激活函数使用 ReLU。输入图像尺寸为 224×224 彩色图。

## 项目结构

```
CNN-Apply/
└── Fruit_Recognition/
    ├── split_to_data.py             # 数据集构建脚本（从 fruit11 划分 Train/Test）
    ├── mean_std.py                  # 计算数据集均值和标准差
    ├── model.py                     # GoogLeNet 网络结构定义（含 Inception 模块）
    ├── model_train.py               # 训练脚本（含训练/验证曲线绘制）
    ├── model_test.py                # 测试脚本（批量测试 & 逐张预测）
    ├── model_inference.py           # 单张图片推理脚本
    ├── get_dataset.py               # 数据集加载与可视化
    ├── best_model.pth               # 训练保存的最优模型权重
    ├── GoogLeNet_train_acc_loss.png # 训练过程的 Loss 和 Accuracy 曲线
    ├── apple.png                    # 推理测试用苹果图片
    ├── img.png                      # 推理测试用水果图片 1
    ├── img_1.png                    # 推理测试用水果图片 2
    ├── fruit11/                     # 均衡后的数据集（平铺结构，每类一个文件夹）
    │   ├── Apple/
    │   ├── Banana/
    │   └── ...                      # 每类 475 张
    └── data/                        # 划分后的数据集目录
        ├── train/
        │   ├── Apple/               # 380 张
        │   ├── Banana/              # 380 张
        │   └── ...                  # 共 11 类
        └── test/
            ├── Apple/               # 95 张
            ├── Banana/              # 95 张
            └── ...                  # 共 11 类
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

从 [Fruits-360 GitHub](https://github.com/fruits-360/fruits-360-100x100) 下载 **fruits-360-100x100** 数据集，将解压后的 `Training` 和 `Test` 文件夹放在 `Fruit_Recognition/fruits-360-100x100-main/` 目录下。

### 2. 构建均衡子集

从原始 Training 数据中提取 11 类、均衡采样，生成 `fruit11/`：

```bash
python data_preprocess.py
```

输出 `fruit11/` 目录，每类一个文件夹，各 475 张图片。

### 3. 划分 Train/Test

将 `fruit11/` 中每类的图片全局打乱，按 8:2 划分训练/测试集：

```bash
python split_to_data.py
```

输出 `data/` 目录，包含 `train/` 和 `test/` 子目录。

### 4. 计算数据集均值与标准差

计算数据集的均值和标准差，用于归一化预处理：

```bash
python mean_std.py
```

> 本项目已预先计算好：mean=(0.6760, 0.5753, 0.5110)，std=(0.2973, 0.3530, 0.3879)

### 5. 训练模型

```bash
python model_train.py
```

训练过程会自动：

- 加载 `./data/train/` 下的训练数据
- 按 8:2 划分训练集和验证集
- 训练 20 个 epoch，使用 Adam 优化器（lr=0.001），Batch Size=128
- 保存最优模型权重到 `best_model.pth`
- 绘制训练/验证的 Loss 和 Accuracy 曲线，保存为 `GoogLeNet_train_acc_loss.png`

### 6. 测试模型

```bash
python model_test.py
```

支持两种测试模式：

- `test_model_process`：批量测试，输出整体准确率
- `test_model_process_v2`：逐张预测，打印每张图片的预测结果与真实标签

### 7. 单张图片推理

```bash
python model_inference.py
```

使用训练好的模型对单张图片（默认为 `img_1.png`）进行推理，输出预测的水果类别。可以替换 `image_process('./img_1.png')` 中的路径来推理其他图片。

### 8. 可视化数据集

```bash
python get_dataset.py
```

展示训练集的一个 batch 图像，查看数据加载效果。

## 数据集

处理后的数据集共包含 **11 种水果**，均来自 [Fruits-360](https://github.com/fruits-360)（CC BY-SA 4.0），原作者 Mihai Oltean：

| 类别         | 标签 | 训练集 | 测试集 | 品种数 |
| ---------- | -- | --- | --- | --- |
| Apple      | 0  | 380 | 95  | 17  |
| Banana     | 1  | 380 | 95  | 5   |
| Cherry     | 2  | 380 | 95  | 5   |
| Grape      | 3  | 380 | 95  | 4   |
| Lemon      | 4  | 380 | 95  | 2   |
| Orange     | 5  | 380 | 95  | 4   |
| Peach      | 6  | 380 | 95  | 2   |
| Pear       | 7  | 380 | 95  | 10  |
| Raspberry  | 8  | 380 | 95  | 6   |
| Strawberry | 9  | 380 | 95  | 4   |
| Watermelon | 10 | 380 | 95  | 1   |

> 训练集总计 4,180 张，测试集总计 1,045 张，所有图像统一 Resize 至 224×224 以适配 GoogLeNet 输入层。每类覆盖了多个品种变体（如 Apple 涵盖 17 个不同品种），保证了数据的品种多样性，同时类别间数量完全均衡。

## 参考

- Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., & Rabinovich, A. (2015). Going Deeper with Convolutions.
- [Fruits-360 Dataset](https://github.com/fruits-360) — CC BY-SA 4.0, Mihai Oltean
- [Classic-CNN-PyTorch](https://github.com/YJL-1101/Classic-CNN-PyTorch) — 经典 CNN 模型实现参考
- [fuits-360-subset-11](https://github.com/YJL-1101/fruits-360-subset-11) — 处理后的数据集，包含 11 种水果


