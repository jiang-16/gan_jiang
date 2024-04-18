import os
import numpy as np
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# 文件夹路径
folder1 = "D:\\参考文献\\Joint-Network paper\\Test-resnet\\P-change\\used"
folder2 = "D:\\参考文献\\Joint-Network paper\\input\\used\\used"


# 获取文件夹中的文件列表
files1 = sorted(os.listdir(folder1))
files2 = sorted(os.listdir(folder2))

# 初始化存储 PSNR 和 SSIM 值的列表
psnr_values = []
ssim_values = []

# 遍历每个文件并计算 PSNR 和 SSIM
for file1, file2 in zip(files1, files2):
    # 读取灰度图像
    image1 = imread(os.path.join(folder1, file1), as_gray=True)
    image2 = imread(os.path.join(folder2, file2), as_gray=True)

    # 将灰度图像转换为 0-255 范围内的整数
    #image1 = np.uint8(image1*255)
    #image2 = np.uint8(image2 * 255)

    # 阈值处理：灰度值大于100的设置为255，小于等于100的设置为0
    image1 = np.where(image1 < 100,0,image1)
    image2 = np.where(image2 < 100,0,image2)

    # 计算峰值信噪比
    psnr = peak_signal_noise_ratio(image1, image2)

    # 计算结构相似性指数
    ssim = structural_similarity(image1, image2)

    # 将 PSNR 和 SSIM 值添加到列表中
    psnr_values.append(psnr)
    ssim_values.append(ssim)

# 打印 PSNR 和 SSIM 值列表
print("PSNR values:", psnr_values)
print("SSIM values:", ssim_values)