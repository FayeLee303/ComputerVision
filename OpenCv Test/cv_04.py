"""
视频特定颜色追踪
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

"""
Opencv的inRange函数：可实现二值化功能
函数原型：inRange(src,lowerb, upperb[, dst]) -> dst         
函数的参数意义：第一个参数为原数组，可以为单通道，多通道。第二个参数为下界，第三个参数为上界
例如：mask = cv2.inRange(hsv, lower_blue, upper_blue)      
第一个参数：hsv指的是原图（原始图像矩阵）
第二个参数：lower_blue指的是图像中低于这个lower_blue的值，图像值变为255
第三个参数：upper_blue指的是图像中高于这个upper_blue的值，图像值变为255 （255即代表黑色）
而在lower_blue～upper_blue之间的值变成0 (0代表白色)
即：Opencv的inRange函数可提取特定颜色，使特定颜色变为白色，其他颜色变为黑色，这样就实现了二值化功能
"""

capture = cv.VideoCapture("./video_example.mp4")
while True:
    ret, frame = capture.read()
    if ret == False:
        break
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  # 色彩空间由RGB转换为HSV
    lower_hsv = np.array([100, 43, 46])  # 设置要过滤颜色的最小值
    upper_hsv = np.array([124, 255, 255])  # 设置要过滤颜色的最大值
    mask = cv.inRange(hsv, lower_hsv, upper_hsv)  # 调节图像颜色信息（H）、饱和度（S）、亮度（V）区间，选择蓝色区域
    cv.imshow("video", frame)
    cv.imshow("mask", mask)
    c = cv.waitKey(40)
    if c == 27:  # 按键Esc的ASCII码为27
        break
cv.destroyAllWindows()
