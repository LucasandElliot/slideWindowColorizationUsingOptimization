import time

import cv2
import numpy as np
import cv2 as cv
import colorsys
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sparse
from sympy import *
from scipy.sparse.linalg import gmres


def Sobel_filter(img, type='gray'):
    # Step3  Sobel计算水平导数,如果最后一个参数为1,1即为竖直方向，如果为1,0为水平方向
    gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return gradient_x, gradient_y


padding = 2  # 窗口半径、图片填充大小
# src放置的是灰度图,marked是放置已经标记成果的图片
photo_name_list = ['waterfall', 'cats', 'gili', 'hair', 'monaco', 'example', 'example2']
photo_name_list = ['cats_u']
solve_name_list = ['spsolve', 'gmres', 'lsmr', 'lgrmes']
spend_time_list = []
for photo_name in photo_name_list:
    photo_file_suffix = 'bmp'
    src = cv.imread('./data/{}.{}'.format(photo_name, photo_file_suffix))# cv读取的是BGR格式
    solve_name = 'spsolve'
    # model_name = "Laplacian算法_{}_{}结果.jpg".format(photo_name, solve_name)
    # model_name = "sobel算子算法_{}_{}结果.jpg".format(photo_name, solve_name)
    # model_name = "边缘检测算法_{}_{}结果.jpg".format(photo_name, solve_name)
    model_name = "侧窗_{}_{}结果.jpg".format(photo_name, solve_name)
    # 加入soble算子增强边缘
    # tem = Sobel_filter(img=src)
    # src = abs(np.array(tem[0])) / np.max(abs(tem[0])) + src.astype(np.float64)
    # src = abs(np.array(tem[1])) / np.max(abs(tem[1])) + src.astype(np.float64)

    # # 加入边缘检测增强边缘
    # edges = abs(cv2.Canny(src, 150, 200))
    # edges_d = np.repeat(edges[:, :, np.newaxis], 3, axis=2) / np.max(edges)
    # src = edges_d + src.astype(np.float64)

    # 加入高通滤波+直方图均衡化

    # 应用Laplacian算子
    # laplacian = cv2.Laplacian(src, cv2.CV_64F)
    # 将结果取绝对值，并转换为8位图像
    # laplacian_abs = np.uint8(np.abs(laplacian)) / np.max(abs(laplacian))
    # src = src.astype(np.float64) + laplacian_abs

    # src_gary = cv2.cvtColor(src, cv.COLOR_BGR2GRAY)
    # cv2.imwrite('./data/{}.bmp'.format(photo_name.split("_")[0]), src_gary)
    src = src[:, :, ::-1]  # 第一通道和第三通道互换，实现BGR到RGB转换
    _src = src.astype(float) / 255
    marked = cv.imread('./data/{}_marked.{}'.format(photo_name, photo_file_suffix))
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
    tem = abs(_src - _marked).sum(2)
    hhash = abs(_src - _marked).sum(2) > 0.60 # 灰度图的U和V为0，但是有颜色的话就会大于0

    # hhash_copy = abs(_src - _marked).sum(2) > 1e-1
    # hhash = (abs(U) + abs(V)) > 1e-4
    # hhash = np.logical_and(hhash, hhash_copy)
    print(len([i for i in hhash.flatten() if i == true])) # 在cats_u上采样中rgb通道计算为339964，但是yuv通道为91391, 但是在cats中14698能够求解
    W = sparse.lil_matrix((size, size))

    YY = np.zeros((Y.shape[0] + 2 * padding, Y.shape[1] + 2 * padding))
    for i in range(YY.shape[0]):
        for j in range(YY.shape[1]):
            YY[i, j] = -10  # 填充后可避免越界
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            YY[i + padding, j + padding] = Y[i, j]


    def best(center):
        r = center[0] + 1 * padding
        c = center[1] + 1 * padding
        y = center[2]
        r_min = r - padding
        r_max = r + padding + 1
        c_min = c - padding
        c_max = c + padding + 1

        LL = YY[r_min:r_max, c_min:c + 1].mean()
        RR = YY[r_min:r_max, c:c_max].mean()
        UP = YY[r_min:r + 1, c_min:c_max].mean()
        DOWN = YY[r:r_max, c_min:c_max].mean()

        # NW = YY[r_min:r + 1, c_min:c + 1].mean()
        # NE = YY[r_min:r + 1, c:c_max].mean()
        # SW = YY[r:r_max, c_min:c + 1].mean()
        # SE = YY[r:r_max, c:c_max].mean()

        if (center[0] < padding or center[0] > Y.shape[0] - padding - 1) and (
                center[1] < padding or center[1] > Y.shape[1] - padding - 1):
            NW = YY[r_min:r + 1, c_min:c + 1].mean()
            NE = YY[r_min:r + 1, c:c_max].mean()
            SW = YY[r:r_max, c_min:c + 1].mean()
            SE = YY[r:r_max, c:c_max].mean()
        else:
            SE = 100.0
            SW = 100.0
            NE = 100.0
            NW = 100.0
            pass
        res = abs(np.array([NE, NW, UP, LL, RR, SW, SE, DOWN]) - y)
        rr_min = r_min
        rr_max = r_max
        cc_min = c_min
        cc_max = c_max
        MIN = res.argmin()
        for i in range(8):
            if i == MIN:
                if i == 0:
                    rr_min = r_min
                    rr_max = r + 1
                    cc_min = c
                    cc_max = c_max
                    break
                elif i == 1:
                    rr_min = r_min
                    rr_max = r + 1
                    cc_min = c_min
                    cc_max = c + 1
                    break
                elif i == 2:
                    rr_min = r_min
                    rr_max = r + 1
                    cc_min = c_min
                    cc_max = c_max
                    break
                elif i == 3:
                    rr_min = r_min
                    rr_max = r_max
                    cc_min = c_min
                    cc_max = c + 1
                    break
                elif i == 4:
                    rr_min = r_min
                    rr_max = r_max
                    cc_min = c
                    cc_max = c_max
                    break
                elif i == 5:
                    rr_min = r
                    rr_max = r_max
                    cc_min = c_min
                    cc_max = c + 1
                    break
                elif i == 6:
                    #
                    rr_min = r
                    rr_max = r_max
                    cc_min = c
                    cc_max = c_max
                    break
                else:
                    rr_min = r
                    rr_max = r_max
                    cc_min = c_min
                    cc_max = c_max
                    break
            else:
                continue
        rr_min -= 1 * padding
        rr_max -= 1 * padding
        cc_min -= 1 * padding
        cc_max -= 1 * padding
        return (rr_min, rr_max, cc_min, cc_max)


    def find_neighbors(center):  # centre [r,c,y]
        neighbors = []
        # 选出最优窗口
        r_min, r_max, c_min, c_max = best(center)
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


    # 遍历所有像素
    start_time = time.time()
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
    ansU, _ = gmres(matA, b_u, maxiter=1000)
    ansU = ansU.reshape(marked.shape[0], marked.shape[1])
    ansV, _ = gmres(matA, b_v, maxiter=1000)
    ansV = ansV.reshape(marked.shape[0], marked.shape[1])
    # ansU, istop, itn, normr, normar, norma, conda, normx = lsmr(matA, b_u)
    # ansU = ansU.reshape(marked.shape[0], marked.shape[1])
    # ansV,  istop, itn, normr, normar, norma, conda, normx = lsmr(matA, b_v)
    # ansV = ansV.reshape(marked.shape[0], marked.shape[1])
    # ansU, _ = lgmres(matA, b_u)
    # ansU = ansU.reshape(marked.shape[0], marked.shape[1])
    # ansV, _ = lgmres(matA, b_v)
    # ansV = ansV.reshape(marked.shape[0], marked.shape[1])

    # YUV转换成rgb格式
    r = Y + 0.9468822170900693 * ansU + 0.6235565819861433 * ansV
    r = np.clip(r, 0.0, 1.0)
    g = Y - 0.27478764629897834 * ansU - 0.6356910791873801 * ansV
    g = np.clip(g, 0.0, 1.0)
    b = Y - 1.1085450346420322 * ansU + 1.7090069284064666 * ansV
    b = np.clip(b, 0.0, 1.0)
    Ans = np.stack((r, g, b), axis=2)
    spend_time_list.append(time.time() - start_time)

    plt.imshow(Ans)
    plt.title("Colorized_with_sidewindow")
    plt.show()
    plt.imsave("./exp/{}".format(model_name), Ans)
data = pd.DataFrame(spend_time_list, columns=['花费时间'])
data.to_excel('./exp/{}.xlsx'.format(model_name), index=False)