#!/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2024/1/15 20:21
# @Author : Lucas
# @File : slideWindowColorization.py

import argparse
parser = argparse.ArgumentParser(description='运用侧窗彩色化图像处理')
# type是要传入的参数的数据类型  help是该参数的提示信息
parser.add_argument('--padding', type=int, nargs='+', help='padding，为侧窗的窗口代销', default=2)
parser.add_argument('--photo_name_list', type=str, nargs='+', help='文件名这里放置图片名字', default='example')
parser.add_argument('--gray_data_dir', type=str, nargs='+', help='这里是存放灰色图像的文件夹', default='./data/original')
parser.add_argument('--marked_data_dir', type=str, nargs='+', help='这里是存放灰色图像的文件夹', default='./data/marked')
parser.add_argument('--exp_dir', type=str, nargs='+', help='这里是存放灰色图像的文件夹', default='./exp')
parser.add_argument('--reshape', type=bool, nargs='+', help='这里是放置图像处理的形状', default=False)
args = parser.parse_args()

padding = 2  # 窗口半径、图片填充大小
# src放置的是灰度图,marked是放置已经标记成果的图片
photo_name_list = ['example{}.png'.format(i) for i in range(1, 9)]
gray_data_dir = args.gray_data_dir
marked_data_dir = args.marked_data_dir
exp_dir = args.exp_dir