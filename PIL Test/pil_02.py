from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

"""
文件操作
"""
# 转换图像格式
def convertImgFormat():
    filelist = './images'
    for infile in filelist:
        outfile = os.path.splitext(infile)[0] + '.png'
        if infile != outfile:
            try:
                Image.open(infile).save(outfile)
            except IOError:
                print('cannot convert'.infile)

# convertImgFormat()


# 返回目录中所有JPG图像的文件名列表
def getImgList(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('jpg')]
path = './images/'
# print(getImgList(path))