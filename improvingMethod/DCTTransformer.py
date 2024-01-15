#!/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2024/1/14 15:58
# @Author : Lucas
# @File : DCTTransformer.py
import numpy as np
from scipy.fft import idct
from skimage import color


def dctTransformer(image):
    # 读取图像
    # image = color.rgb2gray(image)  # 转为灰度图

    # 应用DCT变换
    from scipy.fft import dct
    dct_coeffs = dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')

    # 选择保留的DCT系数
    keep_percentage = 0.7  # 保留右上角70%的系数
    coeffs_shape = dct_coeffs.shape
    mask = np.zeros_like(dct_coeffs)
    mask[:int(keep_percentage * coeffs_shape[0]), :int(keep_percentage * coeffs_shape[1])] = 1

    # 乘以掩码以保留高频信息
    dct_coeffs *= mask

    # 应用逆DCT变换
    restored_image = idct(idct(dct_coeffs, axis=0, norm='ortho'), axis=1, norm='ortho')
    # restored_image = np.interp(restored_image, (restored_image.min(), restored_image.max()), (0, 255))
    restored_image = np.clip(restored_image, 0, 255).astype(np.uint8)
    return restored_image