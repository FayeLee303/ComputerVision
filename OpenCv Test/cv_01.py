"""
图片读取与显示
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 使用matplotlib读取显示
img_1 = plt.imread('../images/lena256.jpg')
plt.imshow(img_1)
# plt.show()

# 使用opencv读取显示
img_2 = cv.imread('../images/lena256.jpg')
# 创建窗口并显示图像
cv.namedWindow('img2')
cv.imshow('img2',img_2)
cv.waitKey(0)
# 释放窗口
cv.destroyAllWindows( )

# 使用opencv读取，matplotlib显示
img_3 = cv.imread('../images/lena256.jpg')
# opencv图像像素的存储顺序是BRG！matplotlib和python默认显示是RGB
# plt.imshow(img_3)
plt.imshow(cv.cvtColor(img_3,cv.COLOR_BGR2RGB))
plt.show()