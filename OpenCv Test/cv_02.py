"""
色彩空间转换
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = plt.imread('../images/lena256.jpg')
fig = plt.figure()
plt.imshow(cv.cvtColor(image,cv.COLOR_BGR2RGB)), plt.title("original")
plt.xticks([]), plt.yticks([])


fig = plt.figure()
# opencv里，COLOR_RGB2GRAY是将三通道RGB对象转换为单通道的灰度对象。
# 将单通道灰度对象转换为 RGB 时，生成的RGB对象的每个通道的值是灰度对象的灰度值
plt.subplot(2, 2, 1)
plt.imshow(cv.cvtColor(image,cv.COLOR_BGR2GRAY)),plt.title("gray")
# cv.imshow('gray',cv.cvtColor(image,cv.COLOR_BGR2GRAY))
# 通常在针对某种颜色做提取时会转换到HSV颜色空间里面来处理
# opencv里HSV色彩空间范围为： H：0-180  S: 0-255   V： 0-255
plt.subplot(2, 2, 2)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2HSV)), plt.title("hsv")
# cv.imshow('hsv', cv.cvtColor(image, cv.COLOR_BGR2HSV))
plt.subplot(2, 2, 3)
plt.imshow(cv.cvtColor(image, cv.COLOR_RGB2YUV)), plt.title("yuv")
# cv.imshow('yuv', cv.cvtColor(image, cv.COLOR_RGB2YUV))
plt.subplot(2, 2, 4)
plt.imshow(cv.cvtColor(image, cv.COLOR_RGB2YCrCb)), plt.title("YCrCb")
# cv.imshow('ycrcb', cv.cvtColor(image, cv.COLOR_RGB2YCrCb))

plt.show()