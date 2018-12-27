import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


"""opencv读取并显示图片"""
img = cv.imread('./images/lena.jpg')
cv.namedWindow('img')
cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()

"""色彩空间转换"""
# opencv图像储存顺序是BGR！matplotlib和python默认显示是RGB
img_RGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
plt.imshow(img_RGB)

# 使用分离通道的方法得到RGB图像
b,g,r = cv.split(img) # 分离通道
img_RGB_2 = cv.merge([r,g,b]) # 合并通道
plt.imshow(img_RGB_2)
# cv2.split函数分离得到各个通道的灰度值(单通道图像)!
# 这样得到的bgr都是灰色的！！！b是blue通道的灰度值！！！！
# 生成一个值为0的单通道数组
zeros = np.zeros(img.shape[:2], dtype="uint8")
# 分别扩展B、G、R成为三通道。另外两个通道用上面的值为0的数组填充
cv.imshow("Blue", cv.merge([b, zeros, zeros]))
cv.imshow("Green", cv.merge([zeros, g, zeros]))
cv.imshow("Red", cv.merge([zeros, zeros, r]))



img_YUV = cv.cvtColor(img_RGB,cv.COLOR_RGB2YUV)
plt.imshow(img_YUV)

img_Gray_1 = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
plt.imshow(img_Gray_1)

# 直接读取灰度图
img_Gray_2 = cv.imread('./images/lena.jpg',0) # 0就是灰色的
plt.imshow(img_Gray_2)

# 使用彩色转灰色公式得到灰度图像
# Gray = r*0.299+g*0.587+b*0.114
b,g,r = cv.split(img) # 分离通道
gray = (r*30+g*59+b*11+50)/100
img_Gray_3 = cv.merge([gray,gray,gray])
plt.imshow(img_Gray_3)

img_HSV = cv.cvtColor(img,cv.COLOR_BGR2HSV)
plt.imshow(img_HSV)