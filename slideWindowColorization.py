#!/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2024/1/15 20:21
# @Author : Lucas
# @File : slideWindowColorization.py

import argparse
import colorsys
import os
import scipy.sparse as sparse
import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.linalg import gmres

from improvingMethod import laplacian
from sideWindow.slideWindowAlgorithm import find_best_window

parser = argparse.ArgumentParser(description='运用侧窗彩色化图像处理')
# type是要传入的参数的数据类型  help是该参数的提示信息
parser.add_argument('--padding', type=int, help='padding，为侧窗的窗口代销', default=2)
parser.add_argument('--gary_photo_file', type=str, help='文件名这里放置图片名字', default='example.png')
parser.add_argument('--marked_photo_file', type=str, help='文件名这里放置图片名字', default='example_marked.png')
parser.add_argument('--gray_data_dir', type=str, help='这里是存放灰色图像的文件夹', default='./data/original')
parser.add_argument('--marked_data_dir', type=str, help='这里是存放灰色图像的文件夹', default='./data/marked')
parser.add_argument('--is_file', dest='is_file', action='store_true', help='这里选择是否文件或者是文件夹', default=False)
parser.add_argument('--exp_dir', type=str, help='这里是存放灰色图像的文件夹', default='./exp')
parser.add_argument('--is_reshape', dest='is_reshape', action='store_true', help='这里是放置图像处理的形状', default=False)
parser.add_argument('--is_store', dest='is_store', action='store_true', help='是否选择保存图片', default=False)
args = parser.parse_args()

padding = 2  # 窗口半径、图片填充大小
# src放置的是灰度图,marked是放置已经标记成果的图片
gray_data_dir = args.gray_data_dir
marked_data_dir = args.marked_data_dir
gray_photo_name_list = [os.path.join(gray_data_dir, i) for i in os.listdir(gray_data_dir)]
gray_photo_name_list.sort()
marked_photo_name_list = [os.path.join(marked_data_dir, i) for i in os.listdir(marked_data_dir)]
marked_photo_name_list.sort()
is_reshape = args.is_reshape
is_store = args.is_store
is_file = args.is_file

exp_dir = args.exp_dir
shape = (256, 256)
if is_file is True:
    gray_photo_name_list = [args.gary_photo_file]
    marked_photo_name_list = [args.marked_photo_file]
for index in range(len(gray_photo_name_list)):
    # cv读取的是BGR格式
    src = cv.imread(gray_photo_name_list[index])
    marked = cv.imread(marked_photo_name_list[index])

    if is_reshape is True:
        src = cv2.resize(src, shape)
        marked = cv2.resize(marked, shape)

    # 使用Laplacian算子进行边缘检测
    laplacian_image = laplacian.laplacian(src)
    # laplacian = np.clip(laplacian, 0, 255)

    # 调整权重
    alpha = 1 / 100
    src = - alpha * abs(laplacian_image) + src

    # 第一通道和第三通道互换，实现BGR到RGB转换
    src = src[:, :, ::-1]
    _src = src.astype(float) / 255



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
    # Lab空间优化
    # 转换到 CIELAB 颜色空间
    lab_image = cv2.cvtColor(marked, cv2.COLOR_RGB2Lab)
    # 提取其他通道
    hhash_copy = abs(lab_image[:, :, 1]) + abs(lab_image[:, :, 2]) > 1
    print("在Lab空间的出现差异个数为" + str(len([i for i in hhash_copy.flatten() if i == True])))
    # 统计marked图像中标记过颜色的像素位置
    hhash_copy = abs(_src - _marked).sum(2) > 1e-2
    print("在RGB归一化空间的出现差异个数为" + str(len([i for i in hhash_copy.flatten() if i == True])))
    # 灰度图的U和V为0，但是有颜色的话就会大于0
    hhash_copy = (abs(U) + abs(V)) > 1e-4
    print("在YUV空间上出现差异个数为" + str(len([i for i in hhash_copy.flatten() if i == True])))
    # YUV
    hhash = (abs(U) + abs(V)) > 1e-4
    # RGB归一化
    # hhash = abs(_src - _marked).sum(2) > 1e-2
    # LAB空间
    # hhash = abs(lab_image[:, :, 1]) + abs(lab_image[:, :, 2]) > 1

    print("在最后计算空间上出现差异个数为" + str(len([i for i in hhash.flatten() if i == True])))
    W = sparse.lil_matrix((size, size))
    print("-"*40)

    YY = np.zeros((Y.shape[0] + 2 * padding, Y.shape[1] + 2 * padding))
    for i in range(YY.shape[0]):
        for j in range(YY.shape[1]):
            YY[i, j] = -10  # 填充后可避免越界
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            YY[i + padding, j + padding] = Y[i, j]

    def find_neighbors(center):  # centre [r,c,y]
        neighbors = []
        # 选出最优窗口
        r_min, r_max, c_min, c_max = find_best_window(YY, Y,  center, padding)
        # 遍历所有的邻居像素
        for r in range(r_min, r_max):
            for c in range(c_min, c_max):
                # 自己本身忽略
                if r == center[0] and c == center[1]:
                    continue
                # 存放邻居像素的xy位置，以及邻居像素的强度，用于后面计算权重的
                neighbors.append([r, c, Y[r, c]])
        return neighbors


    def affinity_a(neighbors, center):  # (r,c,y)
        # 创建一个新的数组，存放权重，同时保留邻居像素的信息，因此可以直接copy数组neighbors
        nbs = np.array(neighbors)
        # 1. 获取邻居像素的强度和中间像素的强度
        sY = nbs[:, 2]  # 邻居像素的强度
        cY = center[2]  # 中间像素的强度
        # 2. 强度差值
        diff = sY - cY
        # 3. 计算均方差
        sig = np.var(np.append(sY, cY))  # 维度相同才可以追加
        if sig < 1e-6:
            sig = 1e-6
        # 4. 根据公式求权重
        wrs = np.exp(- np.power(diff, 2) / (sig * 2))
        # 5. 加权求和，记得wrs是负数
        wrs = - wrs / np.sum(wrs)
        nbs[:, 2] = wrs  # 记录权重
        return nbs


    # 返回值为邻居的权重：[(r,c,w),(r,c,w)]

    # 创建索引
    def to_seq(r, c, cols):
        return r * cols + c
    for r in range(rows):
        for c in range(cols):
            # 1. 将该像素的位置和其强度存在center里面，并计算索引
            center = [r, c, Y[r, c]]  # yuv[(r, c)][0]
            c_idx = to_seq(r, c, cols)
            # 2. 如果该像素没有上过色
            if not hhash[r, c]:
                # 2.1 找到该像素的邻居像素
                neighbors = find_neighbors(center)
                # 2.2 计算权重，weight[0]、weight[1]表示邻居的xy位置，weight[2]表示权重
                weights = affinity_a(neighbors, center)
                # 2.3 放入对应行，因为像素是按顺序遍历的，所以weightData存放的也是按顺序的
                for e in weights:
                    # 2.3.1 计算center像素和邻居像素的索引
                    n_idx = to_seq(e[0], e[1], cols)
                    # 多加入了转换，这里的话迭代的时间较长，所以需要注意
                    n_idx = int(n_idx)
                    c_idx = int(c_idx)
                    # 2.3.2 放入矩阵
                    W[c_idx, n_idx] = e[2]
            # 3. 如果该像素上过色，则直接放入自己本身的信息，权重为1
            W[c_idx, c_idx] = 1.0
    matA = W.tocsr()

    b_u = np.zeros(size)
    b_v = np.zeros(size)
    idx_colored = np.nonzero(hhash.reshape(size))
    u = yuv[:, :, 1].reshape(size)
    b_u[idx_colored] = u[idx_colored]
    v = yuv[:, :, 2].reshape(size)
    b_v[idx_colored] = v[idx_colored]

    # ansU = spsolve(matA, b_u).reshape(marked.shape[0], marked.shape[1])
    # ansV = spsolve(matA, b_v).reshape(marked.shape[0], marked.shape[1])
    ansU, _ = gmres(matA, b_u, maxiter=10000)
    ansU = ansU.reshape(marked.shape[0], marked.shape[1])
    ansV, _ = gmres(matA, b_v, maxiter=10000)
    ansV = ansV.reshape(marked.shape[0], marked.shape[1])

    # YUV转换成rgb格式
    r = Y + 0.9468822170900693 * ansU + 0.6235565819861433 * ansV
    r = np.clip(r, 0.0, 1.0)
    g = Y - 0.27478764629897834 * ansU - 0.6356910791873801 * ansV
    g = np.clip(g, 0.0, 1.0)
    b = Y - 1.1085450346420322 * ansU + 1.7090069284064666 * ansV
    b = np.clip(b, 0.0, 1.0)
    Ans = np.stack((r, g, b), axis=2)

    plt.imshow(Ans)
    plt.title("Colorized_with_sidewindow")
    if is_store is True:
        store_filename = os.path.join(exp_dir, gray_photo_name_list[index].replace('/', '|').replace('\\', '|').split('|')[-1])
        plt.imsave(store_filename, Ans)
    plt.show()
