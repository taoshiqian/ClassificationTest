# -*- coding:utf-8 -*-

import tensorflow as tf

# 神经网络参数
INPUT_NODE = 16
OUTPUT_NODE = 2

IMAGE_SIZE = 4
NUM_CHANNELS = 1
NUM_LABELS = 2

# 第一层卷积层的深度与尺寸
CONV1_DEEP = 4
CONV1_SIZE = 2

# 第二层卷积层的深度与尺寸
CONV1_DEEP = 8
CONV1_SIZE = 2

# 全连接层节点个数
FC_SIZE = 16

# 定义网络
def inference(input_tensor, train, regularizer):
    # 1. 卷积层