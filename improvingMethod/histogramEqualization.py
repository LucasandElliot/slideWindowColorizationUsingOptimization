import os.path

from matplotlib import pyplot as plt

'''
python代码：jpeg图像转jpg图像
https://blog.csdn.net/qq_42250789/article/details/108983375
python+opencv直方图均衡化
https://blog.csdn.net/missyougoon/article/details/81632166
'''
import cv2
import numpy as np
from PIL import Image

def histogram_equalization(image):
    # 将图像转换为YUV色彩空间
    yuv_img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # 分离YUV图像的通道
    y, u, v = cv2.split(yuv_img)
    # 对亮度通道进行直方图均衡化
    equ_y = cv2.equalizeHist(y)
    # 合并均衡化后的亮度通道和原始色度通道
    equ_yuv_img = cv2.merge((equ_y, u, v))
    # 将均衡化后的YUV图像转换回BGR色彩空间
    equ_color_img = cv2.cvtColor(equ_yuv_img, cv2.COLOR_YUV2BGR)
    return  equ_color_img


if __name__ == '__main__':
    image = cv2.imread('../data/results/result.png')
    image = cv2.resize(image, (256, 256))
    histogram_image = histogram_equalization(image)
    fig, ax = plt.subplots(1, 3)
    image = image[:, :, ::-1]
    histogram_image = histogram_image[:, :, ::-1]
    alpha = 1 / 65536
    combine_image = image + alpha * histogram_image
    combine_image = combine_image.astype(np.int32)
    combine_image = combine_image[:, :, ::-1]

    ax[0].imshow(image)
    ax[1].imshow(histogram_image)
    ax[2].imshow(combine_image)
    plt.show()



