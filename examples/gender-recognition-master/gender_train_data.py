import os
import random

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report


def get_img_list(dirname, flag=0):  # 读取文件列表
    rootdir = os.path.abspath('./data/' + dirname + '/')
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    files = []
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            files.append(path)
    return files


def read_img(list, flag=0):  # 读取图片为数据
    for i in range(len(list) - 1):
        if os.path.isfile(list[i]):
            images.append(cv2.imread(list[i]).flatten())
            labels.append(flag)


images = []
labels = []

# 读取图像数据并做标记
read_img(get_img_list('male'), [0, 1])
read_img(get_img_list('female'), [1, 0])

# 转换数组
images = np.array(images)
labels = np.array(labels)

# np.random.permutation(index)对指定范围的序列进行随机排序
per = np.random.permutation(labels.shape[0])
# 重新排序
all_images = images[per, :]
all_labels = labels[per, :]

# 拆分测试集与训练集，2:8
images_total = all_images.shape[0]
# 切片索引需整数
train_num = int(images_total * 0.8)
test_num = images_total - train_num

images_train = all_images[0:train_num, :].astype("float32") / 255
labels_train = all_labels[0:train_num, :].astype("float32")
images_test = all_images[train_num:images_total, :].astype("float32") / 255
labels_test = all_labels[train_num:images_total, :].astype("float32")