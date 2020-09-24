import os
import random

import cv2
import numpy as np
import tensorflow as tf


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

np.set_printoptions(suppress=True)  # 设置小数不以科学计数法输出

import matplotlib.pyplot as plt

# 绘制一张图像
input_images = images_train[1]
input_labels = labels_train[1]
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(np.reshape(input_images, (112, 92, 3)))
plt.show()

# 训练参数
train_epochs = 200
drop_out = 0.7
learning_rate = 0.0000006


def weight_init(shape):
    # 截断的产生正态分布的随机数，即随机数与均值的差值若大于两倍的标准差，则重新生成。
    weight = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(weight)


def bias_init(shape):
    bias = tf.random_normal(shape, dtype=tf.float32)
    return tf.Variable(bias)


"""
输入占位
    input_images: [None,112*92*3]
    input_labels: [None,2]
"""
images_input = tf.placeholder(tf.float32, [None, images_train.shape[1]],
                              name="images_input")
labels_input = tf.placeholder(tf.float32, [None, labels_train.shape[1]],
                              name="labels_input")

"""
input/out: [None,nH,nW,channels]
padding:
    VALID:不填充数据
    SAME:填充数据
"""


def conv2d(images, filter_input, lname):
    return tf.nn.conv2d(images, filter_input, strides=[1, 1, 1, 1],
                        padding="SAME", name=lname)


def maxpooling2d(images, lname):
    return tf.nn.max_pool(images, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding="SAME",name=lname)


x_input = tf.reshape(images_input, [-1, 112, 92, 3])

# 前向传播

# 第一层卷积  16个[3,3,3]卷积核
w1 = weight_init([3, 3, 3, 16])
b1 = bias_init([16])
conv_1 = conv2d(x_input, w1, "conv_1") + b1
relu_1 = tf.nn.relu(conv_1, name="relu_1")
max_pool_1 = maxpooling2d(relu_1, "max_pool_1")

# 第二层卷积 32个[3,3,3]卷积核
w2 = weight_init([3, 3, 16, 32])
b2 = bias_init([32])
conv_2 = conv2d(max_pool_1, w2, "conv_2") + b2
relu_2 = tf.nn.relu(conv_2, name="relu_2")
max_pool_2 = maxpooling2d(relu_2, "max_pool_2")

# 第三层卷积 64个[3,3,3]卷积核
w3 = weight_init([3, 3, 32, 64])
b3 = bias_init([64])
conv_3 = conv2d(max_pool_2, w3, "conv_3") + b3
relu_3 = tf.nn.relu(conv_3, name="relu_3")
max_pool_3 = maxpooling2d(relu_3, "max_pool_3")

print(max_pool_3)
# 展开

flatten = tf.reshape(max_pool_3, [-1, 14 * 12 * 64])
# flatten = tf.reshape(max_pool_3, [-1, 5 * 4 * 64])
# 第一层全连接
w4 = weight_init([14 * 12 * 64, 512])
b4 = bias_init([512])
f1 = tf.matmul(flatten, w4) + b4
relu_f1 = tf.nn.relu(f1)
dropout_f1 = tf.nn.dropout(relu_f1, drop_out)

# 第二层全连接
# w5 = weight_init([512, 256])
# b5 = bias_init([256])
# f2 = tf.matmul(dropout_f1, w5) + b5
# relu_f2 = tf.nn.relu(f2)
# dropout_f2 = tf.nn.dropout(relu_f2, 0.7)

# 第三层
w6 = weight_init([512, 128])
b6 = bias_init([128])
f3 = tf.matmul(dropout_f1, w6) + b6
relu_f3 = tf.nn.relu(f3)
dropout_f3 = tf.nn.dropout(relu_f3, 0.75)

# 第四层全连接，softmax输出
w7 = weight_init([128, 2])
b7 = bias_init([2])
f4 = tf.matmul(dropout_f3, w7) + b7
softmax_f = tf.nn.softmax(f4, name="softmax_f")
# softmax_f = tf.sigmoid(f4,name="sigmoid_f")

# 反向传播
# 交叉熵损失函数
cross_entry = tf.reduce_mean(
    tf.reduce_sum(-input_labels * tf.log(softmax_f)))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entry)

# 计算准确率
arg1 = tf.argmax(input_labels, -1)
arg2 = tf.argmax(softmax_f, -1)
cos = tf.equal(arg1, arg2)
acc = tf.reduce_mean(tf.cast(cos, dtype=tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
Cost = []
Accuracy = []
for i in range(train_epochs):
    # batch_size = random.randint(200, 300)
    # batch_start = random.randint(0, len(images_train) - 300)
    # train_input = images_train[batch_start:(batch_start + batch_size)]
    # train_labels = labels_train[batch_start:(batch_start + batch_size)]
    result, acc1, cross_entry_r, cos1, f_softmax1, relu_1_r = sess.run(
        [optimizer, acc, cross_entry, cos, softmax_f, relu_1],
        feed_dict={images_input: images_train, labels_input: labels_train})
    print(f"第{i}轮，\tloss={cross_entry_r},\tacc={acc1}")
    Cost.append(cross_entry_r)
    Accuracy.append(acc1)

# 代价函数曲线
fig1, ax1 = plt.subplots(figsize=(10, 7))
plt.plot(Cost)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Cost')
plt.title('Cross Loss')
plt.grid()
plt.show()

# 准确率曲线
fig7, ax7 = plt.subplots(figsize=(10, 7))
plt.plot(Accuracy)
ax7.set_xlabel('Epochs')
ax7.set_ylabel('Accuracy Rate')
plt.title('Train Accuracy Rate')
plt.grid()
plt.show()

# # 测试
# arg2_r = sess.run(arg2, feed_dict={images_input: images_test, labels_input: labels_test})
# arg1_r = sess.run(arg1, feed_dict={images_input: images_test, labels_input: labels_test})
# print(arg2_r, arg1_r)
# print(classification_report(arg1_r, arg2_r))

# 保存模型
saver = tf.compat.v1.train.Saver()
saver.save(sess, './model/my-gender-v1.0')
