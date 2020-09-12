import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 前置条件
# 数据可视化库：matplotlib 2D绘图库
#               seaborn 基于matplotlib的统计图形绘制
#               mplot3d是matplotlib的一个基础3D绘图
# 数据分析库：pandas
# 数据处理库：numpy

data0_path="./data/data0.csv"
data0_name=['square','price']

data1_path="./data/data1.csv"
data1_name=['square','bedrooms','price']

# 1.数据处理
def readdata(data_path,data_name):
    df = pd.read_csv(data_path,names=data_name)  # 读取数据
    print(f"数据结构信息:\n{df.info()}\n数据头:\n{df.head()}") # 打印数据信息
    return df

# 数据归一化处理
def normalize_feature(data):
    return data.apply(lambda column:(column - column.mean())/column.std())

# 扩列
def add_ones(data):
    ones = pd.DataFrame({'ones':np.ones(len(data))})
    print(f"""扩充的X0列，数据结构：
    {ones.info()}
    头部数据：
    {ones.head()}""")
    data = pd.concat([ones,data],axis=1)
    return data

# 2.设计模型

# 3.可视化数据流图
def view2D(data,data_name):
    sns.set(context="notebook",style="whitegrid",palette="dark")
    sns.lmplot(x=data_name[0],y=data_name[1],data=data,height=6,fit_reg=True) 
    plt.show()


def view3D(data,data_name,color='Greens'):
    fig = plt.figure()
    ax = plt.axes(projection='3d') # 创建一个3D项目
    # 设坐标轴名称
    ax.set_xlabel(data_name[0])
    ax.set_ylabel(data_name[1])
    ax.set_zlabel(data_name[2])
    # 绘制3D散点图
    ax.scatter3D(data[data_name[0]],data[data_name[1]],data[data_name[2]],c=data[data_name[2]],cmap=color)
    plt.show()
    




# 4.训练模型

# 构建模型
# 分割数据
def getData(data):
    x_data=np.array(data[data.columns[0:3]])
    y_data=np.array(data[data.columns[-1]]).reshape(len(data),1)
    # print(f"{x_data.shape()}\n{y_data.shape()}")
    return x_data,y_data

def model(data):
    x_data,y_data=getData(data)
    learn_rate=0.01 # 学习率
    epoch = 500 # 训练轮数
    # 参数占位
    X_data=tf.placeholder(tf.float32,shape=x_data.shape)
    Y_data=tf.placeholder(tf.float32,shape=y_data.shape)
    # 权重变量
    w = tf.get_variable("w",(x_data.shape[1],1),initializer=tf.constant_initializer())
    # 前置运算
    y_pre = tf.matmul(X_data,w)
    # 损失函数,转置矩阵a
    loss_op=1/(2*len(X_data))*tf.matmul((y_pre-Y_data),transpose_a=True)
    # 梯度下降优化
    train_op = tf.train.GradientDescentOptimizer(learn_rate).minizer(loss_op)
    train()

def train():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(1,epoch+1):
            sess.run(train_op,feed_dict={X_data:x_data,Y_data:y_data})
            if e % 10==0:
                loss,w=sess.run([loss_op,w],feed_dict={X_data:x_data,Y_data:y_data})
                log_str="Epoch %d \t Loss=%.4g \t Model:y = %.4gx1 + %.4gx2 + %.4g"
                print(log_str % (e,loss,w[1],w[2],w[0]))


# 运行函数
if __name__=='__main__':
    # 单变量线性数据
    df0 = readdata(data0_path,data0_name)
    view2D(df0,data0_name)
    # 多变量线性数据
    df1 = readdata(data1_path,data1_name)
    view3D(df1,data1_name)
    # 归一化后
    df1_n = normalize_feature(df1)
    view3D(df1_n,data1_name,color='Reds')
    df1 = add_ones(df1_n)
    print(df1.head())

    model(df1)
