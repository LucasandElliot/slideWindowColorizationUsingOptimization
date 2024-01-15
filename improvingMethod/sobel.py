#!/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2024/1/15 13:50
# @Author : Lucas
# @File : sobel.py
import cv2
import numpy as np


def soble(image):
    # 使用Sobel算子进行边缘检测
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobelx = np.uint8(np.absolute(sobelx))
    sobely = np.uint8(np.absolute(sobely))
    sobelCom = cv2.bitwise_or(sobelx, sobely)
    # 浮点型转成uint8型
    sobelCom = cv2.convertScaleAbs(sobelCom)
    return sobelCom