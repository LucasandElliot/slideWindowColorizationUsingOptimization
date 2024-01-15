#!/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2024/1/14 11:28
# @Author : Lucas
# @File : waveletTransform.py
import cv2
import matplotlib.pyplot as plt
import pywt
import numpy as np
def waveletTransform(image):
    # 分离通道
    channel_blue, channel_green, channel_red = cv2.split(image)
    # 进行2级离散小波变换
    coeffs_blue = pywt.dwt2(channel_blue, 'haar')
    coeffs_green = pywt.dwt2(channel_green, 'haar')
    coeffs_red = pywt.dwt2(channel_red, 'haar')
    # 提取高频信息
    cA_blue, (cH_blue, cV_blue, cD_blue) = coeffs_blue
    cA_green, (cH_green, cV_green, cD_green) = coeffs_green
    cA_red, (cH_red, cV_red, cD_red) = coeffs_red

    high_freq_blue = np.sqrt(cH_blue ** 2 + cV_blue ** 2 + cD_blue ** 2)
    high_freq_green = np.sqrt(cH_green ** 2 + cV_green ** 2 + cD_green ** 2)
    high_freq_red = np.sqrt(cH_red ** 2 + cV_red ** 2 + cD_red ** 2)
    # 合并高频信息
    high_freq_image = cv2.merge([high_freq_blue, high_freq_green, high_freq_red])
    height, width, _ = image.shape
    high_freq_image = cv2.resize(high_freq_image, (height, width))
    high_freq_image = high_freq_image.astype(np.int32)

    # 逆小波变换并合并通道
    reconstructed_blue = pywt.idwt2((np.zeros_like(cA_blue), (cH_blue, cV_blue, cD_blue)), 'haar')
    reconstructed_green = pywt.idwt2((np.zeros_like(cA_green),  (cH_green, cV_green, cD_green)), 'haar')
    reconstructed_red = pywt.idwt2((np.zeros_like(cA_red), (cH_red, cV_red, cD_red)), 'haar')

    # 合并通道
    reconstructed_image = cv2.merge([reconstructed_blue, reconstructed_green, reconstructed_red])
    return high_freq_image

if __name__ == '__main__':
    image = cv2.imread('../data/results/result.png')
    image = cv2.resize(image, (256, 256))
    image = image[:, :, ::-1]
    wave_image = waveletTransform(image)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image)
    ax[1].imshow(wave_image, cmap='gray')
    plt.show()
