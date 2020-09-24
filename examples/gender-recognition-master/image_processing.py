"""
name: 一个处理图片的小程序
version: v0.1
author: intel1999CN
"""

import tensorflow as tf
import numpy as np
import cv2
import os
import pickle

# 屏蔽警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# 获取文件路径
def get_image_path_list(root_directory, folder_name):
    print("正在获取文件路径...\t", end="")
    folder_path = root_directory + "/" + folder_name + "/"
    file_name_list = os.listdir(folder_path)  # 获取目录下所有的文件名
    file_path_list = []  # 初始化文件路径列表
    for i in range(len(file_name_list)):
        file_path_list.append(folder_path + file_name_list[i])  # 合成目录路径与文件名

    print("done.")
    return file_path_list


# 图片读取
def images_read(file_path_list):
    print("正在读取图片数据...\t", end="")
    # 初始化数据列表
    images_data_list = []

    # 进行图片数据读取
    for i in range(len(file_path_list)):
        current_image = cv2.imread(file_path_list[i])  # 获取脚标为i的图片数据
        # current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)  # 纠正颜色通道
        images_data_list.append(current_image)  # 加入列表

    print("done.")
    return np.array(images_data_list)


# 图片缩放
def images_compress(images_data_list, width=100, height=100):
    print("正在进行图片缩放...\t", end="")
    # 初始化缩放后的图片数据列表
    compress_data_list = []

    # 进行逐一缩放
    for i in range(len(images_data_list)):
        before_image = images_data_list[i]  # 获取脚标为i的未处理的图片数据
        after_compress_image = cv2.resize(before_image, (width, height), interpolation=cv2.INTER_CUBIC).flatten()  # 进行缩放
        compress_data_list.append(after_compress_image)

    print("done.")
    return np.array(compress_data_list)


def data_enhancement(compress_data_list):
    print("正在应用数据增强...\t", end="")
    # 初始化数据增强后的数据列表
    after_enhancement_list = []

    with tf.compat.v1.Session() as sess:
        # 初始化计算流
        sess.run(tf.compat.v1.global_variables_initializer())
        # 进行随机数据增强
        for i in range(len(compress_data_list)):
            original_picture = compress_data_list[i]
            after_enhancement_list.append(original_picture)
            new_image_data = [
                sess.run(tf.image.random_flip_left_right(original_picture)),  # 随机水平翻转
                sess.run(tf.image.random_flip_up_down(original_picture)),  # 随机垂直翻转
                sess.run(tf.image.random_contrast(original_picture, lower=0.2, upper=1.8)),  # 随机对比度
                sess.run(tf.image.random_hue(original_picture, max_delta=0.3)),  # 随机对比度
                sess.run(tf.image.random_saturation(original_picture, lower=0.2, upper=1.8))  # 随机对比度
            ]

            after_enhancement_list = list(np.r_[after_enhancement_list, new_image_data])

    print("done.")
    return np.array(after_enhancement_list)


def generate_dataset(images_data_list, label_num, file_name):
    print("正在生成本地文件...")
    # 进行数据标签化处理
    data_dictionary_list = []

    if not os.path.exists("data"):
        os.makedirs("data")

    file_path = f"data/ImageData{file_name}.pkl"
    with open(file_path, "wb") as file:
        for i in range(len(images_data_list)):
            data_dictionary = {
                "ImageData": images_data_list[i],
                "label": label_num
            }
            data_dictionary_list.append(data_dictionary)

        pickle.dump(data_dictionary_list, file)  # 写入文件
        print(f"生成完毕, 文件路径为: {file_path}")


# 一个简单的问答系统
def get_is_next(message):
    while True:
        temp_str = str(input(message))

        if temp_str == "y":
            return True
        elif temp_str == "n":
            return False
        else:
            print("错误: 输入的并非y或n, 请重新输入")


# 输入整形数字
def input_int(output_message):
    while True:
        # 输入信息
        temp = str(input(output_message))

        if temp.isdigit():  # 如果全是数字则输出
            return int(temp)
        else:
            print("错误：输入的并非整形数字")


# 输入序号
def input_order_num(output_message, min_order_num, max_order_num):
    while True:
        temp = input_int(output_message)

        if min_order_num <= temp <= max_order_num:  # 判断输入的数字是否在序号范围内
            return temp
        else:
            print("错误：输入的序号不存在")


def get_folder_name(folder_name_list):
    folder_name_dictionary = {}
    for i in range(len(folder_name_list)):
        folder_name_dictionary[i + 1] = folder_name_list[i]

    return folder_name_dictionary


def main():
    while True:
        root_directory = str(input("请输入数据集根目录:"))
        if os.path.exists(root_directory):
            while True:
                folder_name_list = os.listdir(root_directory)
                folder_name_dictionary = get_folder_name(folder_name_list)
                count = 1
                for i in range(len(folder_name_list)):
                    print(f"{i + 1}. {folder_name_list[i]}")
                    count += 1
                order_num = input_order_num("请输入序号:", 1, count)
                name = folder_name_dictionary[order_num]
                file_path_list = get_image_path_list(root_directory, name)
                images_data_list = images_read(file_path_list)
                compress_data_list = images_compress(images_data_list)
                after_enhancement_list = data_enhancement(compress_data_list)
                generate_dataset(after_enhancement_list, (order_num - 1), file_name=f"{order_num}")
                if not get_is_next("是否继续生成训练集?请输入(y/n):"):
                    break
        else:
            print("错误: 根目录不存在")

        if not get_is_next("是否更换数据集根目录后继续处理数据?请输入(y/n):"):
            break


if __name__ == '__main__':
    main()
