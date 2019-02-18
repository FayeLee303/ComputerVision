"""
直方图均衡化
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 画灰度直方图
def gray_hist(img):
    plt.hist(img.ravel(),256,[0,256])
    #numpy的ravel函数功能是将多维数组降为一维数组
    # plt.show()


# 画三个通道的直方图
def color_hist(img):

    color = ['b','g','r']
    for i,color in enumerate(color):
        # 计算直方图
        hist = cv.calcHist([img],[i],None,[256],[0,256])
        plt.plot(hist,color)    # 折线连起来
        plt.xlim([0,256])
    # plt.show()
"""
cv2.calcHist的原型为：calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]]) -> hist
images参数表示输入图像，传入时应该用中括号[ ]括起来
channels参数表示传入图像的通道，灰度图像只有一个通道，值为0,如果是彩色图像（有3个通道），那么值为0,1,2,中选择一个，对应着BGR各个通道。这个值也得用[ ]传入。
mask参数表示掩膜图像。如果统计整幅图，那么为None。主要是如果要统计部分图的直方图，就得构造相应的掩膜来计算。
histSize参数表示灰度级的个数，需要中括号，比如[256]
ranges参数表示像素值的范围，通常[0,256]。此外，假如channels为[0,1],ranges为[0,256,0,180],则代表0通道范围是0-256,1通道范围0-180。
hist参数表示计算出来的直方图。
"""

# equalizeHist灰度图像直方图均衡化
def gray_equalizeHist(img):
    res = cv.equalizeHist(img)
    cv.imshow('gray',res)

def color_equalizeHist(img):
    # 如果是彩色图像直方图均衡化，需要将彩色图像先用split()方法,将三个通道分别进行均衡化
    # 最后使用merge()方法将均衡化之后的三个通道进行合并
    b,g,r = cv.split(img)
    bH = cv.equalizeHist(b)
    gH = cv.equalizeHist(g)
    rH = cv.equalizeHist(r)
    dst = cv.merge([bH,gH,rH])
    cv.imshow('colorful',dst)






if __name__ == "__main__":
    img_gray = cv.imread('../images/lena256.jpg', 0)  # 读取为灰度图像
    img_color = cv.imread('../images/lena256.jpg')

    # gray_hist(img_gray)
    # color_hist(img_color)
    # plt.show()

    gray_equalizeHist(img_gray)
    color_equalizeHist(img_color)

    """Clahe"""
    clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(10, 10))
    cl1 = clahe.apply(img)
    # 绘图
    plt.subplot(131), plt.imshow(img, 'gray')
    plt.subplot(133), plt.imshow(cl1, 'gray')
    plt.show()


    cv.waitKey(0)
    cv.destroyAllWindows()