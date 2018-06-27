import os
import numpy as np
import cv2
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras import backend

data_path = 'E:\\pycode\\DataSet\\data'
label_path = 'E:\\pycode\\DataSet\\label'
test_images_path = 'E:\\pycode\\DataSet\\test_images'
test_labels_path = 'E:\\pycode\\DataSet\\test_labels'
train_images_path = 'E:\\pycode\\DataSet\\train_images'
train_labels_path = 'E:\\pycode\\DataSet\\train_labels'

num_class = 2
img_rows, img_cols = 4, 4


def get_data():
    train_images, train_labels = [], []
    for file in os.listdir(train_images_path):
        file_path = os.path.join(train_images_path, file)
        img = cv2.imread(file_path, 0)
        train_images.append(img)
    for file in os.listdir(train_labels_path):
        file_path = os.path.join(train_labels_path, file)
        label = np.load(file_path)
        train_labels.append(label)
    print('训练数据读取完成')
    test_images, test_labels = [], []
    for file in os.listdir(test_images_path):
        file_path = os.path.join(test_images_path, file)
        img = cv2.imread(file_path, 0)
        test_images.append(img)
    for file in os.listdir(test_labels_path):
        file_path = os.path.join(test_labels_path, file)
        label = np.load(file_path)
        test_labels.append(label)
    print('测试数据读取完成')
    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)


def NN():
    trainX, trainY, testX, testY = get_data()
    print('trainX.shape', trainX.shape)
    print('trainY.shape', trainY.shape)
    print('testX.shape', testX.shape)
    print('testY.shape', testY.shape)
    # 根据不同的底层。设置不同的输入层格式。
    if backend.image_data_format() == 'channels_first':
        trainX = trainX.reshape(trainX.shape[0], 1, img_rows, img_cols)
        testX = testX.reshape(testX.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        trainX = trainX.reshape(trainX.shape[0], img_rows, img_cols, 1)
        testX = testX.reshape(testX.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    trainX = trainX.astype('float32')
    testX = testX.astype('float32')
    trainX /= 255.0
    testX /= 255.0

    # 使用keras定义神经网络模型
    model = Sequential()
    # 层1： 卷积层1+relu ： 深度32 卷积核5*5
    # 28*28*1 -> 28*28*32
    model.add(Conv2D(4, kernel_size=(2, 2), activation='relu', input_shape=input_shape))
    # 层2： 池化层1  ： 2*2最大池化
    # 28*28*32 -> 14*14*32
    # model.add(MaxPool2D(pool_size=(2, 2)))
    # 层3： 卷积层2+relu ： 深度64 卷积核5*5
    # 14*14*32 -> 14*14*64
    model.add(Conv2D(8, kernel_size=(2, 2), activation='relu'))
    # 层4： 池化层2  ： 2*2最大池化
    # 14*14*64 -> 7*7*64
    # model.add(MaxPool2D(pool_size=(2, 2)))
    # 拉直
    model.add(Flatten())
    # 层5： 全连接层1+relu ： 500节点
    model.add(Dense(16, activation='relu'))
    # 层6： 全连接层2+softmax到最后的输出：num_class
    model.add(Dense(num_class, activation='softmax'))

    # 定义损失函数，优化函数，评测方法
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(),
                  metrics=['accuracy'])

    # 通过Keras的API训练模型并计算在测试数据上的准确率
    # 训练数据，batch_size，训练轮数， 验证数据
    model.fit(
        trainX, trainY,
        batch_size=16,
        epochs=4,
        validation_data=(testX, testY)
    )

    # 在测试数据上计算准确率
    score = model.evaluate(testX, testY)
    print('Test loss', score[0])
    print('Test accuracy', score[1])


if __name__ == '__main__':
    NN()
