import image_processing as im
import numpy as np
import matplotlib.pyplot as plt
list_path = im.get_image_path_list("./data", "test")
image_list = im.images_read(list_path)

image_list = im.images_compress(image_list, width=92, height=112)
image_list = np.array(image_list)
image_list = np.reshape(image_list[:], (-1, 112 * 92 * 3)).astype("float32") / 255
input_labels = np.array([[1, 0], [0, 1]]).astype("float32")


# 绘制一张图像
input_images = image_list[0]
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(np.reshape(input_images, (112, 92, 3)))
plt.show()

import tensorflow as tf
import os

graph_path = os.path.abspath('./model/my-gender-v1.0.meta')
model = os.path.abspath('./model/')

sess = tf.Session()
server = tf.train.import_meta_graph(graph_path)
server.restore(sess, tf.train.latest_checkpoint(model))

graph = tf.get_default_graph()

# 填充feed_dict
x = graph.get_tensor_by_name('images_input:0')
y = graph.get_tensor_by_name('labels_input:0')
feed_dict = {x: np.reshape(image_list[0], [1, 30912]), y: np.reshape(input_labels[0], [1, 2])}

# 第一层卷积+池化
relu_1 = graph.get_tensor_by_name('relu_1:0')
max_pool_1 = graph.get_tensor_by_name('max_pool_1:0')

# 第二层卷积+池化
relu_2 = graph.get_tensor_by_name('relu_2:0')
max_pool_2 = graph.get_tensor_by_name('max_pool_2:0')

# 第三层卷积+池化
relu_3 = graph.get_tensor_by_name('relu_3:0')
max_pool_3 = graph.get_tensor_by_name('max_pool_3:0')

# 全连接最后一层输出
f_softmax = graph.get_tensor_by_name('softmax_f:0')

# conv1 特征
r1_relu = sess.run(relu_1, feed_dict)
r1_tranpose = sess.run(tf.transpose(r1_relu, [3, 0, 1, 2]))
fig, ax = plt.subplots(nrows=1, ncols=16, figsize=(16, 1))
for i in range(16):
    ax[i].imshow(r1_tranpose[i][0])
plt.title('Conv1 16*112*92')
plt.show()

# pool1特征
max_pool_1 = sess.run(max_pool_1, feed_dict)
r1_tranpose = sess.run(tf.transpose(max_pool_1, [3, 0, 1, 2]))
fig, ax = plt.subplots(nrows=1, ncols=16, figsize=(16, 1))
for i in range(16):
    ax[i].imshow(r1_tranpose[i][0])
plt.title('Pool1 16*56*46')
plt.show()

print(sess.run(f_softmax, feed_dict))
