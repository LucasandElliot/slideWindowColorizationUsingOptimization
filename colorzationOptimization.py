#!/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2023/12/8 14:32
# @Author : Lucas
# @File : colorzationOptimization.py
# 窗口半径为1，效果最好
import time

import numpy as np
import cv2 as cv
import colorsys
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
photo_name_list = ['waterfall', 'cats', 'gili', 'hair', 'monaco', 'example', 'example2']
# photo_name_list = ['example2']
solve_name_list = ['spsolve', 'gmres', 'lsmr', 'lgrmes']
spend_time_list = []
for photo_name in photo_name_list:
    photo_file_suffix = 'bmp'
    src = cv.imread('./data/{}.{}'.format(photo_name, photo_file_suffix))  # cv读取的是BGR格式
    solve_name = 'spsolve'
    src = src[:, :, ::-1]  # 第一通道和第三通道互换，实现BGR到RGB转换
    _src = src.astype(float) / 255
    marked = cv.imread('./data/{}_marked.{}'.format(photo_name, photo_file_suffix))
    model_name = "colorization_using_optimization_{}_{}结果.jpg".format(photo_name, solve_name)
    marked = marked[:, :, ::-1]
    _marked = marked.astype(float) / 255
    Y, _, _ = colorsys.rgb_to_yiq(_src[:, :, 0], _src[:, :, 1], _src[:, :, 2])  # Y通道是原灰度图的
    _, U, V = colorsys.rgb_to_yiq(_marked[:, :, 0], _marked[:, :, 1], _marked[:, :, 2])  # 待求的U和V是marked图像的
    yuv = colorsys.rgb_to_yiq(_marked[:, :, 0], _marked[:, :, 1], _marked[:, :, 2])
    yuv = np.stack(yuv, axis=2)
    y = yuv[:, :, 0]

    rows = _src.shape[0]
    cols = _src.shape[1]
    size = rows * cols
    # 统计marked图像中标记过颜色的像素位置
    # hhash_copy = isColored = abs(_src - _marked).sum(2) > 0.01 # 灰度图的U和V为0，但是有颜色的话就会大于0
    hhash = (abs(U) + abs(V)) > 1e-4
    # hhash = np.logical_and(hhash, hhash_copy)
    W = sparse.lil_matrix((size, size))


    def find_neighbors(center, pic):
        neighbors = []
        # 1. 求出该像素的邻居遍历范围，同时要考虑像素在边界
        r_min = max(0, center[0] - 1)
        r_max = min(pic.shape[0], center[0] + 2)
        c_min = max(0, center[1] - 1)
        c_max = min(pic.shape[1], center[1] + 2)
        # 遍历所有的邻居像素
        for r in range(r_min, r_max):
            for c in range(c_min, c_max):
                # 自己本身忽略
                if r == center[0] and c == center[1]:
                    continue
                # 2. 存放邻居像素的xy位置，以及邻居像素的强度，用于后面计算权重的
                neighbors.append([r, c, Y[r, c]])
        return neighbors


    def affinity_a(neighbors, center):
        # 创建一个新的数组，存放权重，同时保留邻居像素的信息，因此可以直接copy数组neighbors
        nbs = np.array(neighbors)
        # 1. 获取邻居像素的强度和中间像素的强度
        sY = nbs[:, 2]  # 邻居像素的强度
        cY = center[2]  # 中间像素的强度
        # 2. 强度差值
        diff = sY - cY
        # 3. 计算均方差
        sig = np.var(np.append(sY, cY))
        if sig < 1e-6:
            sig = 1e-6
        # 4. 根据公式求权重
        wrs = np.exp(- np.power(diff, 2) / (sig * 2.0))
        # 5. 加权求和，记得wrs是负数
        wrs = - wrs / np.sum(wrs)
        nbs[:, 2] = wrs  # 记录权重
        return nbs


    def to_seq(r, c, rows):
        return int(c * rows + r)


    # 遍历所有像素
    # 遍历所有像素
    start_time = time.time()
    for c in range(cols):
        for r in range(rows):
            # 1. 将该像素的位置和其强度存在center里面，并计算索引
            center = [r, c, Y[r, c]]
            c_idx = to_seq(r, c, rows)
            # 2. 如果该像素没有上过色
            if not hhash[r, c]:
                # 2.1 找到该像素的邻居像素
                neighbors = find_neighbors(center, yuv)
                # 2.2 计算权重，weight[0]、weight[1]表示邻居的xy位置，weight[2]表示权重
                weights = affinity_a(neighbors, center)
                # 2.3 放入对应行，因为像素是按顺序遍历的，所以weightData存放的也是按顺序的
                for e in weights:
                    # 2.3.1 计算center像素和邻居像素的索引
                    n_idx = to_seq(e[0], e[1], rows)
                    # 2.3.2 放入矩阵
                    W[c_idx, n_idx] = e[2]
            # 3. 如果该像素上过色，则直接放入自己本身的信息，权重为1
            W[c_idx, c_idx] = 1.0
    matA = W.tocsr()

    b_u = np.zeros(size)
    b_v = np.zeros(size)
    idx_colored = np.nonzero(hhash.reshape(size, order='F'))
    u = yuv[:, :, 1].reshape(size, order='F')
    b_u[idx_colored] = u[idx_colored]
    v = yuv[:, :, 2].reshape(size, order='F')
    b_v[idx_colored] = v[idx_colored]

    ansU = sparse.linalg.spsolve(matA, b_u)
    ansV = sparse.linalg.spsolve(matA, b_v)
    spend_time_list.append(time.time() - start_time)


    def yuv_to_rgb(cY, cU, cV):
        ansRGB = [colorsys.yiq_to_rgb(cY[i], cU[i], cV[i]) for i in range(len(ansY))]
        ansRGB = np.array(ansRGB)
        ans = np.zeros(yuv.shape)
        ans[:, :, 0] = ansRGB[:, 0].reshape(rows, cols, order='F')
        ans[:, :, 1] = ansRGB[:, 1].reshape(rows, cols, order='F')
        ans[:, :, 2] = ansRGB[:, 2].reshape(rows, cols, order='F')
        return ans


    ansY = Y.reshape(size, order='F')
    ans1 = yuv_to_rgb(ansY, ansU, ansV)

    plt.imshow(ans1)
    plt.title("Colorized_without_sidewindow")
    plt.show()
    plt.imsave("./exp/{}".format(model_name), ans1)
data = pd.DataFrame(spend_time_list, columns=['花费时间'])
data.to_excel('./exp/{}.xlsx'.format(model_name), index=False)