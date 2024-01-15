#!/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2024/1/14 12:35
# @Author : Lucas
# @File : cannyDection.py
import cv2
import numpy as np


def canny_dection(image):
    # 高斯滤波
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # 计算梯度
    gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度幅值和方向
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_direction = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)

    # 非极大值抑制
    edges = cv2.Canny(blurred, 30, 100)
    # 获取图像的高度和宽度
    height, width = edges.shape

    # 创建一个三通道的空白图像（黑色背景）
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    color_image = np.clip(color_image, 0, 255)

    # 将灰度图像的值复制到每个通道
    color_image[:, :, 0] = edges
    color_image[:, :, 1] = edges
    color_image[:, :, 2] = edges
    return color_image