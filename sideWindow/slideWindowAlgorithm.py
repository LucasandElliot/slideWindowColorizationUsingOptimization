#!/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2024/1/14 10:50
# @Author : Lucas
# @File : slideWindowAlgorithm.py
# padding = 2  # 窗口半径、图片填充大小
import numpy as np


def find_best_window(YY, Y, center, padding=2):
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