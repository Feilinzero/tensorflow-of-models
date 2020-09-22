# 导包，加载数据集

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K

(x_train, y_train), (x_test, y_test) = mnist.load_data("mnist/mnist.npz")
# 规范化数据
img_rows, img_columns = (28, 28)
if (K.image_data_format() == "channels_last"):
    X_train = x_train.reshape(x_train[0].shape, img_rows, img_columns, 1).astype("float32")
    X_test = x_test.reshape(x_test[0].shape, img_rows, img_columns, 1).astype("float32")
    input_shape = (img_rows, img_columns, 1)
else:
    X_train = x_train.reshape(x_train[0].shape, 1, img_rows, img_columns).astype("float32")
    X_test = x_test.reshape(x_test[0].shape, 1, img_rows, img_columns).astype("float32")
    input_shape = (1, img_rows, img_columns)
X_train /= 255
X_test /= 255
# 独热编码
from tensorflow.keras.utils import to_categorical

n_classes = 10
Y_train, Y_test = (to_categorical(y_train, n_classes), to_categorical(y_test, n_classes))
# 搭建模型
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, MaxPooling2D, Conv2D

model = Sequential()
model.add(Conv2D(filter=32,
                 kernel_size=(3, 3),
                 activation="relu",
                 input_shape=input_shape))
model.add(Conv2D(filter=64,
                 kernel_size=(3, 3),
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(n_classes, activation="softmax"))

model.compile(loss=tf.losses.categorical_crossentropy, optimizer=tf.optimizers.Adam, metrics=tf.metrics.Accuracy)

model.fit(X_train, Y_train, batch_size=128, epochs=10, verbose=2, validation_split=0.25)
