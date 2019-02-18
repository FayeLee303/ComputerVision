"""
通道的分离与合并
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


image = plt.imread('../images/lena256.jpg')
# 三通道分离形成单通道图片
b, g, r = cv.split(image)

# 注意分离出来时单通道，图片是灰色的！！！
plt.subplot(2, 3, 1), plt.imshow(b), plt.title('b')
plt.subplot(2, 3, 2), plt.imshow(b), plt.title('g')
plt.subplot(2, 3, 3), plt.imshow(b), plt.title('r')
cv.imshow('b', b)
cv.imshow('g', g)
cv.imshow('r', r)

# 生成一个值为0的单通道数组
zeros = np.zeros(image.shape[:2], dtype = "uint8")
# 分别扩展B、G、R成为三通道。另外两个通道用上面的值为0的数组填充
bb = cv.merge([b,zeros,zeros])
gg = cv.merge([zeros,g,zeros])
rr = cv.merge([zeros,zeros,r])
cv.imshow('b', bb)
cv.imshow('g', gg)
cv.imshow('r', rr)

# 三个单通道合成一个三通道图片
src = cv.merge([b, g, r])
plt.subplot(2, 3, 4), plt.imshow(src), plt.title('merge')
cv.imshow('merge', src)
# 修改多通道里的某个通道的值
src[:, :, 2] = 0
plt.subplot(2, 3, 5), plt.imshow(src), plt.title('change')
cv.imshow('change', src)
plt.show()



"""生成图片"""
def create_image():
    """
    单通道： 此通道上值为0－255。 （255为白色，0是黑色） 只能表示灰度，不能表示彩色
    三通道：BGR （255，255，255为白色， 0,0,0是黑色 ）  可以表示彩色， 灰度也是彩色的一种。
    注意opencv里对图像储存顺序是BGR！！！不是RBG
    :return:
    """
    # 三通道
    # img = np.zeros([400,400,3],np.uint8)    # 将所有像素点的各通道数值赋0
    # img[:, :, 0] = np.ones([400, 400]) * 255  # 0通道代表B
    # img[:, :, 1] = np.ones([400, 400]) * 255   #1通道代表G
    # img[:, :, 2] = np.ones([400, 400]) * 255   #2通道代表R
    # 单通道
    img = np.ones([400, 400, 1], np.uint8)  # 该像素点只有一个通道，该函数使所有像素点的通道的灰度值为1
    img = img * 127  # 使每个像素点单通道的灰度值变为127
    return img

cv.imshow('createimg',create_image().squeeze())