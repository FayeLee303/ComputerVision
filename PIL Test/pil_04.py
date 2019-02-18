"""
灰度变换
"""
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

img = np.array(Image.open('../images/lena_gray.jpg'))   # 2~255

# 反相处理
img2 = 255 - img  # 0~253

# 将图像像素值变换到100～200区间内
img3 = (100.0/255)*img + 100 # 100~200

# 对图像像素值求平方后得到的图像，使得较暗的像素值变得更小
img4 = 255.0*(img/255.0) **2 # 0~255

# 使用PIL的求相反
img5 = Image.fromarray(img)

def imgresize(img,size):
    im = Image.fromarray(uint8(img))
    return np.array(im.resize(size))

img6 = imgresize(img5,(128,128))

fig = plt.figure()
plt.subplot(151)
plt.imshow(img)
plt.subplot(152)
plt.imshow(img2)
plt.subplot(153)
plt.imshow(img3)
plt.subplot(154)
plt.imshow(img4)
plt.subplot(155)
plt.imshow(img5)


plt.show()

"""
如果通过一些操作将“uint8”数据类型转换为其他数据类型
那么在创建 PIL 图像之前，需要将数据类型转换回来：
img = Image.fromarray(uint8(img))
"""