import os
import numpy as np
import cv2
import shutil

images_path = 'E:\\pycode\\DataSet\\images'
labels_path = 'E:\\pycode\\DataSet\\labels'
test_images_path = 'E:\\pycode\\DataSet\\test_images'
test_labels_path = 'E:\\pycode\\DataSet\\test_labels'
train_images_path = 'E:\\pycode\\DataSet\\train_images'
train_labels_path = 'E:\\pycode\\DataSet\\train_labels'


def make_image():
    # img_rgb = cv2.imread('data/_.jpg')
    # img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)
    # img = thresh
    # print(img.shape)
    # print(img)
    # cv2.imwrite('data/__.jpg', img)
    img = cv2.imread('E:\\pycode\\DataSet\\other\\__.jpg', 0)
    print(img.shape)
    print(img)

    for num in range(6):
        for i in range(16):
            x, y = divmod(i, 4)
            if num & (1 << i):
                img[x, y] = 255
            else:
                img[x, y] = 0
        print(num)
        print(img)
        cv2.imwrite(os.path.join(images_path, str(num) + '.jpg'), img)


def make_label():
    for num in range(65536):
        inside_count = bin(num & 1632)[2:].count('1')
        outside_count = bin(num & 63903)[2:].count('1')
        print(num, inside_count, outside_count)
        label = inside_count >= outside_count
        label = [label, 1 - label]
        np.save(os.path.join(labels_path, str(num) + '.npy'), label)
        tmp = np.load(os.path.join(labels_path, str(num) + '.npy'))
        print(tmp)


def split_train_test(split_num):
    # images, labels = [], []
    # for num in range(65536):
    #     image = cv2.imread(os.path.join(data_path, str(num) + '.bmp'), 0)
    #     # print(img)
    #     images.append(image)
    #     label = np.load(os.path.join(label_path, str(num) + '.npy'))
    #     # print(label)
    #     labels.append(label)
    # print(images)
    # print('*******************')
    # print(labels)
    # 共同打乱两个数据
    file_list = []
    for i in range(65536):
        file_list.append(i)
    import random
    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(file_list)
    print(file_list)

    # train_images = images[:60000]
    # test_images = images[60000:]
    # train_labels = labels[:60000]
    # test_labels = labels[60000:]

    # 训练集与测试集存起来
    for i in range(65536):
        num = file_list[i]
        if i < split_num:
            shutil.copyfile(os.path.join(images_path, str(num) + '.bmp'),
                            os.path.join(train_images_path, str(num) + '.bmp'))
            shutil.copyfile(os.path.join(labels_path, str(num) + '.npy'),
                            os.path.join(train_labels_path, str(num) + '.npy'))
        else :
            shutil.copyfile(os.path.join(images_path, str(num) + '.bmp'),
                            os.path.join(test_images_path, str(num) + '.bmp'))
            shutil.copyfile(os.path.join(labels_path, str(num) + '.npy'),
                            os.path.join(test_labels_path, str(num) + '.npy'))
        if i % 1000 == 0:
            print('完成了',i)
    print('完成了65536份数据的切分')


if __name__ == '__main__':
    # print(bin(123)[2:].count('1'))
    # make_image()
    # make_label()
    split_train_test(split_num=60000)
