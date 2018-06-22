import os
import numpy
import cv2

def make_image():
    # img_rgb = cv2.imread('data/_.jpg')
    # img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)
    # img = thresh
    # print(img.shape)
    # print(img)
    # cv2.imwrite('data/__.jpg', img)
    img = cv2.imread('other/__.jpg',0)
    print(img.shape)
    print(img)

    for num in range(65536):
        for i in range(16):
            x, y = divmod(i, 4)
            if num & (1 << i):
                img[x, y] = 255
            else:
                img[x, y] = 0
        print(num)
        print(img)
        cv2.imwrite('data/'+str(num)+'.jpg', img)

def make_label():
    pass

if __name__ == '__main__':
    make_image()
    make_label()

