#!/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2024/1/15 17:09
# @Author : Lucas
# @File : improve_edges.py
import cv2
import numpy as np
from matplotlib import pyplot as plt
def improve_edges(image):

    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    sobelx = np.uint8(np.absolute(sobelx))
    sobely = np.uint8(np.absolute(sobely))
    sobelCom = cv2.bitwise_or(sobelx, sobely)

    C = image + laplacian
    E = cv2.blur(sobelCom, (5, 5))
    F = C * E

    mina = np.min(F)
    maxa = np.max(F)

    F = np.uint8(255 * (F - mina) / (maxa - mina))
    G = image + F
    H = cv2.pow(G / 255.0, 0.5)
    return G
