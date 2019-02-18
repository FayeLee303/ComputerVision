from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

"""
图像显示
"""

# 读取图片,返回一个PIL图像对象
img = Image.open('../images/lena512.jpg')
img2 = Image.open('../images/lena512.jpg').convert('L')

# 使用matplotlib显示图像
imgg = np.array(Image.open('../images/monkey512.bmp'))
plt.imshow(imgg)


# 把图像用数组储存
imggg = np.array(Image.open('../images/monkey512.bmp').convert('L'),'f')

# shape包括了图像数组的行列和颜色通道
# 普通打开，图像被编码成无符号八位整数uint8
# 对图像进行灰度处理，并且使用参数f转化为浮点数，图像编码为float32
print(imgg.shape,imgg.dtype)
print(imggg.shape,imggg.dtype)

print('Please click 3 points')
x = plt.ginput(3) # 鼠标交互
print('you clicked:',x)


# 创建缩略图
# img.thumbnail((128,128))

# 复制和粘贴图像区域
# 使用左上右下四元祖坐标来指定
# 坐标系左上角(0,0)
box = (100,100,400,400)
img4 = img.crop(box)    # crop 裁剪指定区域
img5 = img4.transpose(Image.ROTATE_180) # 旋转
img.paste(img5,box) # paste放回去

# 调整图像的尺寸
img6 = img.resize((128,128))
img7 = img.rotate(45)


fig = plt.figure()
plt.subplot(141)
plt.imshow(img)
plt.subplot(142)
plt.imshow(img2)
plt.subplot(143)
plt.imshow(img4)
plt.subplot(144)
plt.imshow(img7)

plt.show()