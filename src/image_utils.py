import cv2
import numpy as np
import random
from matplotlib import pyplot as plt

def add_gaussian_noise(image_array, mean=0.0, var=40):
    std = var ** 0.5  # **幂运算符
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped


def flip_image(image_array):
    return cv2.flip(image_array, 1)  # 图像翻转。第二个参数决定以什么形式翻转：1为水平翻转 0为垂直翻转 -1为水平垂直翻转


def color2gray(image_array):   # RGB图像转化成三通道灰度图
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    gray_img_3d = image_array.copy()
    gray_img_3d[:, :, 0] = gray
    gray_img_3d[:, :, 1] = gray
    gray_img_3d[:, :, 2] = gray
    return gray_img_3d


def rotate(img, limit_up=15, limit_down=-15):

    rows, cols = img.shape[:2]
    center_coordinate = (int(cols / 2), int(rows / 2))
    angle = random.uniform(limit_down, limit_up)
    # angle=limit_up

    M = cv2.getRotationMatrix2D(center_coordinate, angle, 1)

    # 仿射变换
    out_size = (cols, rows)
    rotate_img = cv2.warpAffine(img, M, out_size, borderMode=cv2.BORDER_REPLICATE)

    return rotate_img


def shift(img, distance_down=5, distance_up=5):

    rows, cols = img.shape[:2]
    y_shift = random.uniform(distance_down, distance_up)
    x_shift = random.uniform(distance_down, distance_up)

    # 生成平移矩阵
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    # 平移
    img_shift = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

    return img_shift


def crop(img, crop_x=40, crop_y=40):

    rows, cols = img.shape[:2]
    # 偏移像素点
    x_offset = random.randint(0, cols - crop_x)
    y_offset = random.randint(0, rows - crop_y)

    # 读取部分图像
    img_part = img[y_offset:(y_offset+crop_y), x_offset:(x_offset+crop_x)]


    return img_part


def lighting_adjust(img, k_down=0.5, k_up=1.3, b_down=2, b_up=2):

    # 对比度调整系数
    slope = random.uniform(k_down, k_up)
    # 亮度调整系数
    bias = random.uniform(b_down, b_up)

    # 图像亮度和对比度调整
    img = img * slope + bias
    # 灰度值截断，防止超出255
    img = np.clip(img, 0, 255)

    return img.astype(np.uint8)


