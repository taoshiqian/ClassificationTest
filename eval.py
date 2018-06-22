# coding:
import numpy as np
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import inference
import train
import cv2

data_path = 'E:\\pycode\\DataSet\\data'
label_path = 'E:\\pycode\\DataSet\\label'
test_images_path = 'E:\\pycode\\DataSet\\test_images'
test_labels_path = 'E:\\pycode\\DataSet\\test_labels'
#train_images_path = 'E:\\pycode\\DataSet\\train_images'
#train_labels_path = 'E:\\pycode\\DataSet\\train_labels'

# 每10秒加载一次模型，并在最新模型上测试准确率
EVAL_INTERVAL_SECS = 10
BATCH_SIZE = 5536


def evaluate(test_images, test_labels):
    with tf.Graph().as_default() as g:
        # 定义输入输出的格式
        x = tf.placeholder(
            tf.float32,
            [BATCH_SIZE, inference.IMAGE_SIZE, inference.IMAGE_SIZE, inference.NUM_CHANNELS],
            name='x-input'
        )
        y_ = tf.placeholder(tf.float32, [None, inference.NUM_LABELS], name='y-input')
        #reshaped_x = np.reshape(mnist.validation.images,(None, LeNet5_inference.IMAGE_SIZE, LeNet5_inference.IMAGE_SIZE, LeNet5_inference.NUM_CHANNELS))

        xs, ys = test_images, test_labels

        reshaped_xs = np.reshape(xs, (
            BATCH_SIZE,
            inference.IMAGE_SIZE,
            inference.IMAGE_SIZE,
            inference.NUM_CHANNELS))

        validate_feed = {
            x: reshaped_xs,
            y_: ys
        }

        # 直接调用inference的网络，不需要正则化项
        y = inference.inference(x, train=False,regularizer=None)

        # 准确率。
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名的方式来加载模型。
        variable_average = tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_average.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        # 每隔EVAL_INTERVAL_SECS秒调用一次检测
        while True:
            with tf.Session() as sess:
                # get_checkpoint_state通过checkpoint文件自动找到目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名得到模型保存时迭代的次数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print(
                        "After %s training step(s), validation accuracy = %g %%" % (global_step, accuracy_score * 100))
                else:
                    print('No checkpoint file ')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    #mnist = input_data.read_data_sets('mnist_data', one_hot=True)
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
    evaluate(test_images, test_labels)


if __name__ == '__main__':
    tf.app.run()
