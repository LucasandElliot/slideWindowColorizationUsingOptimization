#!/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2024/1/15 17:07
# @Author : Lucas
# @File : laplacian.py
import cv2


def laplacian(image):
    edges = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
    edges = cv2.convertScaleAbs(edges)
    return edges