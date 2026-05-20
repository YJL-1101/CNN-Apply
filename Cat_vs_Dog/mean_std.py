from PIL import Image
import os
import numpy as np

# 数据集文件夹路径
folder_path = 'PetImages'

# 初始化变量
total_pixels = 0
sum_pixel_values = np.zeros(3)

# 支持的图片格式
img_exts = ('.jpg', '.jpeg', '.png', '.bmp')

# 第一次遍历：计算RGB三个通道的像素值总和
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.lower().endswith(img_exts):
            image_path = os.path.join(root, filename)

            # 统一转换为RGB三通道
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image) / 255.0

            # 统计像素数量，不包含通道数
            total_pixels += image_array.shape[0] * image_array.shape[1]

            # 累加RGB三个通道的像素值
            sum_pixel_values += np.sum(image_array, axis=(0, 1))

# 计算均值
mean = sum_pixel_values / total_pixels

# 第二次遍历：计算方差
sum_squared_diff = np.zeros(3)

for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.lower().endswith(img_exts):
            image_path = os.path.join(root, filename)

            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image) / 255.0

            diff = (image_array - mean) ** 2
            sum_squared_diff += np.sum(diff, axis=(0, 1))

# 计算方差
variance = sum_squared_diff / total_pixels

# 计算标准差
std = np.sqrt(variance)

print("Mean:", mean)
print("Variance:", variance)
print("Std:", std)