import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


"""
点噪声与平滑
"""
img = cv.imread('../images/lena256.jpg',0) # 读取为灰度图像
# for i in range(2000):
#     # 添加点噪声
#     temp_x = np.random.randint(0,img.shape[0])
#     temp_y = np.random.randint(0,img.shape[1])
#     img[temp_x][temp_y] = 255 # 白点
# blur_1 = cv.GaussianBlur(img,(5,5),0) # 高斯模糊
# blur_2 = cv.medianBlur(img,5) # 均值模糊

# plt.subplot(131)
# plt.imshow(img,'gray')
# plt.subplot(1,3,2)
# plt.imshow(blur_1,'gray')
# plt.subplot(1,3,3)
# plt.imshow(blur_2,'gray')
#
# plt.show()

"""拉普拉斯边缘检测"""
# laplacian = cv.Laplacian(img,cv.CV_64F)
# sobelX = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
# sobelY = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
# # 绘图
# plt.subplot(2,2,1)
# plt.imshow(img,cmap='gray'),plt.title("Original"),plt.xticks([]),plt.yticks([])
# plt.subplot(2,2,2)
# plt.imshow(laplacian,cmap='gray'),plt.title("Laplacian"),plt.xticks([]),plt.yticks([])
# plt.subplot(2,2,3)
# plt.imshow(sobelX,cmap='gray'),plt.title("SobelX"),plt.xticks([]),plt.yticks([])
# plt.subplot(2,2,4)
# plt.imshow(sobelY,cmap='gray'),plt.title("SobelY"),plt.xticks([]),plt.yticks([])
#
# plt.show()



