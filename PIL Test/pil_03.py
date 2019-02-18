"""
图像轮廓与直方图
"""
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# 绘制图像的轮廓（或其他二维函数的等轮廓线）需要对每个坐标 [x, y] 的像素值施加同一个阈值
# 所以首先需要将图像灰度化

# 读取图像到数组中
img = np.array(Image.open('../images/lena512.jpg').convert('L'))

fig = plt.figure()
plt.subplot(121)
plt.gray() # 不使用颜色信息
plt.contour(img,origin='image')
plt.axis('equal')
plt.axis('off')

plt.subplot(122)
# hist函数绘制灰度直方图，第二个参数指定小区间的数目
# 因为hist()只接受一维数组作为输入，绘制图像直方图之前，必须先对图像进行压平处理
# flatten()方法将任意数组按照行优先准则转换成一维数组
plt.hist(img.flatten(),128)

plt.show()