import os
from shutil import copy
import random


def mkfile(path):
    """如果文件夹不存在，则创建文件夹"""
    if not os.path.exists(path):
        os.makedirs(path)



# 1. 基本路径设置


# 原始数据集路径
src_path = 'PetImages'

# 划分后的数据集保存路径
dst_path = 'data'

# 获取所有类别名称，例如 ['cat', 'dog']
class_names = [cla for cla in os.listdir(src_path)
               if os.path.isdir(os.path.join(src_path, cla))]

# 设置随机种子，保证每次运行划分结果一致
random.seed(0)

# 测试集比例：训练集 : 测试集 = 9 : 1
split_rate = 0.1



# 2. 创建 train 和 test 文件夹


# 创建训练集和测试集主目录
train_path = os.path.join(dst_path, 'train')
test_path = os.path.join(dst_path, 'test')

mkfile(train_path)
mkfile(test_path)

# 在 train 和 test 下创建每个类别的子文件夹
for cla in class_names:
    mkfile(os.path.join(train_path, cla))
    mkfile(os.path.join(test_path, cla))



# 3. 划分数据集


for cla in class_names:
    # 当前类别的原始图片路径
    cla_path = os.path.join(src_path, cla)

    # 获取当前类别下所有图片文件名
    images = os.listdir(cla_path)
    num = len(images)

    # 随机抽取一部分图片作为测试集
    test_images = random.sample(images, k=int(num * split_rate))

    for index, image in enumerate(images):
        # 原始图片完整路径
        image_path = os.path.join(cla_path, image)

        # 如果当前图片被抽中，则复制到测试集
        if image in test_images:
            new_path = os.path.join(test_path, cla)
        # 否则复制到训练集
        else:
            new_path = os.path.join(train_path, cla)

        copy(image_path, new_path)

        # 打印处理进度
        print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")

    print()

print("processing done!")