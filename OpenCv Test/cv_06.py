"""
调用摄像头
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


"""图片属性"""
def get_image_info(image):
    print("image.type%s, image.shape%s, image.size%s, image.dtype%s"%
          (type(image),image.shape,image.size,image.dtype))
    pixel_data = np.array(image)
    # print(pixel_data) # 图片矩阵

"""显示图片"""
def show_image(image,string):
    cv.namedWindow(string)
    cv.imshow(string,image)  # 显示
    # cv.imwrite('./img',img)    # 保存
    cv.waitKey(0)
    # 释放窗口
    cv.destroyAllWindows()
    # plt.subplot(1,1,1)
    # plt.imshow(image)

"""调用摄像头显示"""
def video_demo():
    # 参数为视频设备的id，如果只有一个摄像头可以填0，表示打开默认的摄像头
    # 这里的参数也可以是视频文件名路径
    capture = cv.VideoCapture(0)
    # 只要没跳出循环，就会循环播放每一帧，waitKey(10)表示间隔10ms
    while True:
        # read函数读取视频/摄像头的某帧，返回两个参数
        # 第一个参数是bool类型的ret，代表有没有读到图片
        # 第二个参数是frame，是当前截取一帧的图片
        ret,frame = capture.read()
        # 翻转  0:沿X轴翻转(垂直翻转)
        # 大于0:沿Y轴翻转(水平翻转)
        # 小于0:先沿X轴翻转，再沿Y轴翻转，等价于旋转180°
        frame = cv.flip(frame,1)
        cv.imshow('video',frame)
        if cv.waitKey(10) == ord('q'):
            # 键盘输入q退出窗口，不按q点击关闭会一直关不掉
            # ord()函数返回对应字符的ASCII数值
            break


if __name__=="__main__":
    img = cv.imread('../images/lena256.jpg')
    # get_image_info(img)
    # show_image(img,'img')
    video_demo()