# ------------------------
# 对图像进行简单变化
# 不改变图像的像素值、在图像平面上进行像素的重新安排
# ------------------------

import cv2
import numpy as np
import math
img = cv2.imread('I:\\PycharmProjects\\pythonProject\\opencv\\rose.png')

# (一)图像的平移
# 构造移动矩阵H
# 在x轴方向移动多少距离，在y轴方向移动多少距离
H = np.float32([[1, 0, 50], [0, 1, 25]])
rows, cols = img.shape[0], img.shape[1]
print(img.shape)
print(rows, cols)
res = cv2.warpAffine(img, H, (cols, rows))
cv2.imshow('origin_picture', img)
cv2.imshow('new_picture', res)
cv2.waitKey(0)


# (二)图像的放大和缩小
# 一是通过设置图像缩放比例，即缩放因子，来对图像进行放大或缩小
res1 = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
height, width = img.shape[:2]

# 二是直接设置图像的大小，不需要缩放因子
res2 = cv2.resize(img, (int(0.8 * width), int(0.8 * height)), interpolation=cv2.INTER_AREA)

cv2.imshow('origin_picture', img)
cv2.imshow('res1', res1)
cv2.imshow('res2', res2)
cv2.waitKey(0)

# 最近邻插值算法：选择离他映射到位置最近的输入像素的灰度值为插值结果

# need_height参数不得超过原有图像高的两倍， 这是由算法本身性质所决定的
# need_height参数则可以任意选取


def pic_interpolation(img, need_height, need_width):
    height, width, channels = img.shape
    print(height, width, channels)
    emptyImage = np.zeros((need_height, need_width, channels), np.uint8)
    sh = need_height / height
    sw = need_width / width
    for i in range(need_height):
        for j in range(need_width):
            x = round(i / sh)
            y = math.floor(j / sw)
            emptyImage[i, j] = img[x, y]

    return emptyImage


zoom = pic_interpolation(img, 220, 180)
cv2.imshow('nearest neighbor', zoom)
cv2.imshow('image', img)
cv2.waitKey(0)