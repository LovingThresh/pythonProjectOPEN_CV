import cv2 as cv
import numpy as np

# 读取保存图像
img = cv.imread('I:/dataset/rose/rose.png')
print(img)
print(img.shape)
cv.namedWindow('Image')
cv.imshow('Image', img)
cv.waitKey(0)
cv.destroyWindow('Image')

cv.imwrite('../rose.png', img, [cv.IMWRITE_PNG_COMPRESSION, 2])

