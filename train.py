# -*- coding:utf-8 -*-

import numpy as np
import os
import tensorflow as tf
import inference
import matplotlib.pyplot as plt
import cv2

data_path = 'E:\\pycode\\DataSet\\data'
label_path = 'E:\\pycode\\DataSet\\label'
test_images_path = 'E:\\pycode\\DataSet\\test_images'
test_labels_path = 'E:\\pycode\\DataSet\\test_labels'
train_images_path = 'E:\\pycode\\DataSet\\train_images'
train_labels_path = 'E:\\pycode\\DataSet\\train_labels'


TRAIN_NUM = 60000
DATASET_SIZE = 65536
# 神经网络参数
BATCH_SIZE = 16
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZER_RATE = 0.0001
TRAINING_STEPS = TRAIN_NUM//BATCH_SIZE + 1  # 60000/16
MOVING_AVERAGE_DECAY = 0.99


# 模型
MODEL_SAVE_PATH = 'model/'
MODEL_NAME = 'model.ckpt'

def train(train_images, train_labels):
    x = tf.placeholder(tf.float32,[BATCH_SIZE,inference.IMAGE_SIZE, inference.IMAGE_SIZE, inference.NUM_CHANNELS], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, inference.NUM_LABELS], name='y-input')

    # 网络结构
    regularizer = regularizer = tf.contrib.layers.l2_regularizer(REGULARIZER_RATE)
    y = inference.inference(x, train=True, regularizer=regularizer)

    # 定义训练轮数的变量，该变量不可训练
    global_step = tf.Variable(0, trainable=False)

    # 初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step
    )
    # 把 所有可训练的变量 使用滑动平均来更新，即 神经网络的参数
    variable_averages_op = variable_averages.apply(
        tf.trainable_variables()
    )

    # 交叉熵作为损失函数。logits=神经网络不包含softmax层的前向传播结果，labels=正确答案
    # 由于groundtruth正确答案是10维向量，所以使用tf.argmax得到答案
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1)
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE  # 基础学习率
        , global_step  # 当前迭代轮数
        , TRAINING_STEPS  # 过完所有数据需要的迭代次数
        , LEARNING_RATE_DECAY  # 学习率的衰减速度
        , staircase=True
    )

    # 使用 GradientDescentOptimizer梯度下降 优化算法来优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 使用tf更新网络参数和滑动平均值，效果同tf.group()
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # tf持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 仅训练，不测试
        for i in range(TRAINING_STEPS):
            #xs, ys = mnist.train.next_batch(BATCH_SIZE)
            start = (i * BATCH_SIZE) % DATASET_SIZE
            end = min(start + BATCH_SIZE, DATASET_SIZE)
            #print(start,end)
            xs = train_images[start:end]
            ys = train_labels[start:end]
            #print(ys)
            try:
                reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                              inference.IMAGE_SIZE,
                                              inference.IMAGE_SIZE,
                                              inference.NUM_CHANNELS))
            except:
                continue
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: reshaped_xs, y_: ys})
            # 每1000轮输出一次loss保存一次模型
            if i % 100 == 0:
                print("After %d training steps, loss on traning batch is %g." % (step, loss_value))
                # if i % 10000 == 0:  # 每1000轮输出一次loss保存一次模型
                saver.save(
                    sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                    global_step=global_step  # 模型文件名末尾加上训练次数 model.ckpt-1000
                )


def main(argv=None):
    #if not os.path.exists("mnist_data"): os.mkdir("mnist_data")
    #if not os.path.exists("model"): os.mkdir("model")
    #载入mnist数据，如果目录下面没有，则会下载
    #mnist = input_data.read_data_sets("mnist_data", one_hot=True)

    train_images, train_labels = [], []
    for file in os.listdir(train_images_path):
        file_path = os.path.join(train_images_path,file)
        img = cv2.imread(file_path,0)
        train_images.append(img)
    for file in os.listdir(train_labels_path):
        file_path = os.path.join(train_labels_path,file)
        label = np.load(file_path)
        train_labels.append(label)
    print('训练数据读取完成')
    print(train_labels)
    train(train_images, train_labels)


if __name__ == '__main__':
    # tf.app.run()
    main()
