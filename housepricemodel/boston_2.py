import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# 读取数据集
df = pd.read_csv("./data/boston.csv")
# 显示数据集

# 获取df的值
df = df.values
# 转换为np的数组格式
df = np.array(df)
print(df)
# 对特征数据【0到11】列做（0-1）归一化
for i in range(12):
    # x=x/(max(x)-min(x))
    df[:, i] = (df[:, i] - df[:, i].min()) / (df[:, i].max() - df[:, i].min())
x_data = df[:, :12]  # 特征集
y_data = df[:, 12]  # 标签集

# 定义特征数据和标签数据占位
x = tf.placeholder(tf.float32, shape=[None, 12], name="X")
y = tf.placeholder(tf.float32, shape=[None, 1], name="Y")

# 定义一个命名空间,决定空间内对象和操作的所属区域
with tf.name_scope('Model'):
    # w 初始化值为shape=（12,1）的随机数
    w = tf.Variable(tf.random_normal([12, 1], stddev=0.01, name='W'))
    # b 初始化值为1.0
    b = tf.Variable(1.0, name='b')


    # w和x是矩阵相乘，用matmul，不能用mutiply或*
    def model(x, w, b):
        return tf.matmul(x, w) + b


    # 预测计算操作，前向计算节点
    pred = model(x, w, b)

# 训练模型
# learning_rate = 0.01
# 定义学习率
LEARNING_RATE_BASE = 0.01  # 学习率基数,初始值
global_step = tf.Variable(0, trainable=False)  # 训练轮数
LEARNING_RATE_STEP = 1  # 学习率更新频率,一般为:总样本数/每轮输入的样本数,即样本数输入的轮数
LEARNING_RATE_DECAY = 0.99  # 学习率衰减率
# 定义指数衰减学习率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, LEARNING_RATE_STEP, LEARNING_RATE_DECAY,
                                           staircase=False)

# 定义均方差损失函数
with tf.name_scope('LossFunction'):
    # loss_function = tf.reduce_mean(tf.pow(y-pred, 2))
    loss_function = tf.reduce_mean(tf.square(y - pred))
# 定义优化器,梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function, global_step=global_step)
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function,global_step=global_step)

# 声明会话启动会话
sess = tf.Session()
# 初始化参数
sess.run(tf.global_variables_initializer())

loss_list = []
# 训练轮数
train_epochs = 500
for epoch in range(train_epochs):
    loss_sum = 0.0
    # 抽取每行的数据xs(12,),ys()
    for xs, ys in zip(x_data, y_data):
        # 数据变形，要和placeholder的shape保持一致
        xs = xs.reshape(1, 12)
        ys = ys.reshape(1, 1)
        # 喂入数据
        _, loss = sess.run([optimizer, loss_function], feed_dict={x: xs, y: ys})
        # _, summary_str, loss = sess.run([optimizer, sum_loss_op, loss_function], feed_dict={x: xs, y: ys})
        # writer.add_summary(summary_str, epoch)
        # 计算本轮loss值得和
        loss_sum = loss_sum + loss
        loss_list.append(loss)
    # 打乱数据顺序
    xvalues, yvalues = shuffle(x_data, y_data)

    b0temp = b.eval(session=sess)
    w0temp = w.eval(session=sess)
    loss_average = loss_sum / len(y_data)

    # loss_list.append(loss_average)#每轮添加一次
    print('epoch=', epoch + 1, 'loss=', loss_average, 'b=', b0temp, 'w=', w0temp)
plt.plot(range(len(loss_list)), loss_list, color="red")
plt.show()
print(f"学习率:{sess.run(learning_rate)}")

# 模型验证

n = np.random.randint(500)  # 一共500条数据
x_test = x_data[n]
x_test = x_test.reshape(1, 12)
predict = sess.run(pred, feed_dict={x: x_test})
print('预测值：%f' % predict)
target = y_data[n]
print('标签值：%f' % target)

# loss 可视化
# 设置日志存储目录
logdir = 'log/linear/test1'
# 创建一个操作，记录损失值loss，后面在tensorboard中SCALARS栏可见
sum_loss_op = tf.summary.scalar('loss', loss_function)
# 把所有需要记录摘要日志文件的合并，方便一次性写入
merged = tf.summary.merge_all()
