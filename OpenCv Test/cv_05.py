"""
像素运算
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

"""算术运算"""
# 要进行算术运算，两张图片的形状（shape）size必须一样
# 在相除的时候，一个很小的数除以很大的数结果必然小，所以得出的图像几乎全黑。（黑色为0，白色为255）
def add_image(m1,m2):
    result = cv.add(m1,m2)
    cv.imshow('add',result)

def subtract_image(m1,m2):
    result = cv.subtract(m1,m2)
    cv.imshow('subtract',result)

def multiply_image(m1,m2):
    result = cv.multiply(m1,m2)
    cv.imshow('multiply',result)

def divide_image(m1,m2):
    result = cv.divide(m1,m2)
    cv.imshow('divide',result)


"""逻辑运算"""
# 逻辑运算是按照像素点的各通道的值按二进制形式按位与或非进行运算的
def add_logical(m1,m2):
    # 与运算  每个像素点每个通道的值按位与
    result = cv.bitwise_and(m1,m2)
    cv.imshow('and',result)

def or_logical(m1,m2):
    # 或运算   每个像素点每个通道的值按位或
    result = cv.bitwise_or(m1,m2)
    cv.imshow('or',result)

def not_logical(m1,m2):
    # 非运算   每个像素点每个通道的值按位取反
    result = cv.bitwise_not(m1,m2)
    cv.imshow('not',result)

def xor_logical(m1,m2):
    # 异或运算   ？
    result = cv.bitwise_xor(m1,m2)
    cv.imshow('xor',result)

"""调节图片对比度和亮度"""

"""
函数addWeighted的原型：
addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]]) -> dst
src1表示需要加权的第一个数组（上述例子就是图像矩阵）
alpha表示第一个数组的权重
src2表示第二个数组（和第一个数组必须大小类型相同）
beta表示第二个数组的权重
gamma表示一个加到权重总和上的标量值
即输出后的图片矩阵：
result = src1*alpha + src2*beta + gamma;
"""
def contrast_brightness_image(m1,ratio,b):
    # 第2个参数rario为对比度  第3个参数b为亮度
    h,w,ch = m1.shape
    # 新建的一张全黑图片和img1图片shape类型一样，元素类型也一样
    new_image = np.zeros([h,w,ch],m1.dtype)
    result = cv.addWeighted(m1,ratio,new_image,1-ratio,b)
    cv.imshow('changed',result)

"""遍历访问图像的每一个像素点"""
def access_pixels(image):
    height,width,channels = image.shape[0],image.shape[1],image.shape[2]
    print("width%s, height%s,channels%s" % (width, height, channels))
    for row in range(height):
        for col in range(width):
            for c in range(channels):
                # 获取每个像素值每个通道的值
                pv = image[row,col,c]
                # 灰度值是0～255
                # 修改每个像素点每个通道的灰度值
                # 取反操作
                image[row, col, c] = 255-pv
    # return image
    cv.imshow('-pv',image)


"""调用opencv的库函数实现像素取反"""
def inverse(image):
    # 函数cv.bitwise_not可以实现像素点各通道值取反
    image = cv.bitwise_not(image)
    # return image
    cv.imshow('bitwise', image)


if __name__ == "__main__":
    img1 = cv.imread('../images/1.jpg')
    img2 = cv.imread('../images/2.jpg')
    cv.imshow('1',img1)
    cv.imshow('2',img2)

    # add_image(img1,img2)
    # subtract_image(img1, img2)
    # multiply_image(img1, img2)
    # divide_image(img1, img2)
    #
    # add_logical(img1, img2)
    # or_logical(img1, img2)
    # not_logical(img1, img2)
    # xor_logical(img1, img2)
    #
    # contrast_brightness_image(img1,0.9,90)

    # # GetTickcount函数返回从操作系统启动到当前所经过的毫秒数
    # t1 = cv.getTickCount()
    # changed_img = access_pixels(img)
    # t2 = cv.getTickCount()
    # # getTickFrequency函数返回CPU的频率,就是每秒的计时周期数
    # time = (t2 - t1) / cv.getTickFrequency()
    # print("遍历像素运行时间 : %s ms" % (time * 1000))  # 输出运行时间

    # inverse_img = inverse(img)
    access_pixels(img1)
    inverse(img1)

    cv.waitKey(0)
    cv.destroyAllWindows()

