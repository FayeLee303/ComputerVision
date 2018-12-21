import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
from PIL import Image


"""
二维DCT变换
F = A*f*A.T 
A是转换矩阵
f是N*N的图像
A[i,j] = c(i)*cos((j+0.5)*pi/N*i)
"""
# class Transform(object):
#     # def __init__(self,img):
#     #     self.img = self.pre_process(img) # 保证长款一致
#     #     self.A = np.zeros(self.img.shape)
#     #     self.create_DCT_transform_matrix(self.img) # 变换矩阵
#
#     # 做成单例模式
#     instance = None
#     init_flag = False
#
#     def __new__(cls,*args,**kwargs):
#         if cls.instance is None:
#             cls.instance = super().__new__(cls)
#         return cls.instance
#
#     def __init__(self):
#         if self.init_flag is True:
#             return
#         else:
#             self.init_flag = True
#
#     # 对图像进行预处理，如果长宽不相等，就在补最后一行或这最后一列的数据
#     def pre_process(self,img):
#         if img.shape[0] == img.shape[1]:
#             return img
#         else:
#             if img.shape[0] > img.shape[1]: # 行比列多，要补列
#                 print(img.shape)
#                 column = img[:,-1] # 最后一列的数据,这里得到的是行向量！！！
#                 # print(column)
#                 column = column.reshape((img.shape[0],1)) # 变成列向量！
#                 # print(column)
#                 deference = img.shape[0]-img.shape[1]
#                 r = np.tile(column,(1,deference))
#                 # print(r)
#                 img = np.hstack((img,r)) # 横向拼接
#                 print(img.shape)
#                 return img
#             else: # 列比行多，要补行
#                 print(img.shape)
#                 row = img[-1,:] # 最后一行的数据
#                 deference = img.shape[1]-img.shape[0]
#                 r = np.tile(row, (deference,1))
#                 img = np.vstack((img, r))  # 纵向拼接
#                 print(img.shape)
#                 return img
#
#     # 得到变换矩阵
#     def create_DCT_transform_matrix(self, matrix):
#         N = matrix.shape[0] # 图像大小
#         A = np.zeros((matrix.shape[0],matrix.shape[0])) # 变换矩阵
#         for i in range(N):
#             for j in range(N):
#                 if i == 0:
#                     c = math.sqrt(1/N)
#                 else:
#                     c = math.sqrt(2/N)
#                 # self.A[i,j] = c * math.cos((((j+0.5)*math.pi)/N)*i)
#                 A[i, j] = c * math.cos((((j + 0.5) * math.pi) / N) * i)
#         return A
#
#     def DCT(self,matrix):
#         # F = np.dot(np.dot(self.A,matrix),self.A.T)
#         A = self.create_DCT_transform_matrix(matrix)
#         F = np.dot(np.dot(A, matrix), A.T)
#         # print(F)
#         # print(F.shape)
#         return F
#
#
#     def iDCT(self,matrix):
#         # f = np.dot(np.dot(self.A.T,matrix),self.A)
#         A = self.create_DCT_transform_matrix(matrix)
#         f = np.dot(np.dot(A, matrix), A.T)
#         # print(f)
#         # print(f.shape)
#         return f
#
#     # 分块
#     def devide_blocks(self,matrix):
#         blocks = [] # 用来装划分的块，之后用来复原
#
#         img_block = np.zeros((8,8))
#         shape_1 = matrix.shape[0] # 512
#         shape_2 = img_block.shape[0] # 64
#
#         # 分块
#         m,n = 0,0
#         for i in range(0,int(shape_1/shape_2)):
#             for j in range(0,int(shape_1/shape_2)):
#                 if n + shape_2 - 1 <= shape_1 - 1:
#                     img_block = matrix[m:m+shape_2,n:n+shape_2]
#                     # print(img_block)
#                     blocks.append(img_block) # 放到数组里
#                     n += shape_2
#                 # else:
#                 #     # 到头了
#                 #     n = 0
#                 #     break
#             if m + shape_2 - 1 <= shape_1 - 1:
#                 m += shape_2
#                 n = 0
#             # else:
#             #     # 到头了
#             #     break
#         # print(blocks)
#         print(np.array(blocks).shape)
#         return blocks
#
#
#
#     # 合并块
#     def merge_blocks(self,size,blocks):
#
#         img_block = np.zeros((8,8))
#         shape_1 = size  # 512
#         shape_2 = img_block.shape[0]  # 8
#
#         # 合并
#         i = 0
#         row, rows = blocks[0], []
#         # 合并列为一行
#         while True:
#             if i > len(blocks) - int(shape_1 / shape_2):
#                 break
#             else:
#                 row = blocks[i]
#                 t = 1
#                 while True:
#                     row = np.hstack((row, blocks[i + 1]))
#                     t += 1
#                     if t >= int(shape_1 / shape_2):
#                         rows.append(row)
#                         break
#                 i += int(shape_1 / shape_2)
#         # print('rows', rows)
#         # print(len(rows))
#         # 合并行为全部图像
#         num = len(rows)-1 # 一共做这么些次合并
#         j = 0
#         img = rows[0]
#         while j<num:
#             img = np.vstack((img, rows[j + 1]))
#             j += 1
#             # print(np.array(img).shape)
#         print(np.array(img).shape)
#         return img
#
#     # z字扫描，输入矩阵得到一维数组
#     def Zigzag(self,matrix):
#         N = matrix.shape[0]
#
#
#     # 根据一维数组，反变换为矩阵
#     def iZigzag(self,array):
#         pass
#
#     # 保留一部分系数，其他部分置0
#     def retain(self,num,matrix):
#         matrix = np.array(matrix)
#         for i in range(matrix.shape[0]):
#             for j in range(matrix.shape[0]):
#                 if i > num or j > num:
#                     matrix[i, j] = 0
#         return matrix
#
#     # 量化
#     def quantify(self, matrix):
#         pass
#
#




class Transform(object):
    #做成单例模式
    instance = None
    init_flag = False

    def __new__(cls,*args,**kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self):
        if self.init_flag is True:
            return
        else:
            self.init_flag = True

    # 对图像进行预处理，如果长宽不相等，就在补最后一行或这最后一列的数据
    def pre_process(self,img):
        if img.shape[0] == img.shape[1]:
            return img
        else:
            if img.shape[0] > img.shape[1]: # 行比列多，要补列
                print(img.shape)
                column = img[:,-1] # 最后一列的数据,这里得到的是行向量！！！
                # print(column)
                column = column.reshape((img.shape[0],1)) # 变成列向量！
                # print(column)
                deference = img.shape[0]-img.shape[1]
                r = np.tile(column,(1,deference))
                # print(r)
                img = np.hstack((img,r)) # 横向拼接
                print(img.shape)
                return img
            else: # 列比行多，要补行
                print(img.shape)
                row = img[-1,:] # 最后一行的数据
                deference = img.shape[1]-img.shape[0]
                r = np.tile(row, (deference,1))
                img = np.vstack((img, r))  # 纵向拼接
                print(img.shape)
                return img

    Y_quantify_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                  [12, 12, 14, 19, 26, 58, 60, 55],
                                  [14, 13, 16, 24, 40, 57, 69, 56],
                                  [14, 17, 22, 29, 51, 87, 80, 62],
                                  [18, 22, 37, 56, 68, 109, 103, 77],
                                  [24, 35, 55, 64, 81, 104, 113, 92],
                                  [49, 64, 78, 87, 103, 121, 120, 101],
                                  [72, 92, 95, 98, 112, 100, 103, 99]
                                  ])

    U_quantify_matrix = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                                  [18, 21, 26, 66, 99, 99, 99, 99],
                                  [24, 26, 56, 99, 99, 99, 99, 99],
                                  [47, 99, 99, 99, 99, 99, 99, 99],
                                  [99, 99, 99, 99, 99, 99, 99, 99],
                                  [99, 99, 99, 99, 99, 99, 99, 99],
                                  [99, 99, 99, 99, 99, 99, 99, 99],
                                  [99, 99, 99, 99, 99, 99, 99, 99]
                                  ])

    # 保留一部分系数，其他部分置0
    # type=1是对矩阵取左上角
    # type=2是对z字扫描后的一维数组取前几位再变成矩阵
    # type=3是对矩阵分块后取左上角
    # type=4是对矩阵分块后z字扫描后的一维数组取前几位再变成矩阵
    def retain(self,num,type=1,matrix=None,split_size=8):
        img_size = matrix.shape[0]

        if type == 1:
            matrix = np.array(matrix) # 整张图的dct系数矩阵
            # 取左上角
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[0]):
                    if i > num-1 or j > num-1:
                        matrix[i, j] = 0
            return matrix   # 丢弃高频系数后的整张图的dct系数矩阵
        elif type == 2:
            matrix = np.array(matrix)  # 整张图的dct系数矩阵
            z_list = self.zigzag(matrix) # Z字扫描
            # 只保留Z扫描的前num位
            for i in range(len(z_list)):
                if i > num:
                    z_list[i] = 0
            matrix = self.izigzag(z_list)  # 丢弃高频系数后的整张图的dct系数矩阵
            return matrix
        elif type == 3:
            # matrix 整张图的像素矩阵！
            blocks = self.split(matrix,split_size=split_size) # 切分成块,blocks是list,block是array
            blocks_dct, blocks_idct = [],[]

            for i in range(len(blocks)):
                block_dct = cv.dct(blocks[i]) # 对每一块做DCT变换

                # 取左上角
                for i in range(block_dct.shape[0]):
                    for j in range(block_dct.shape[0]):
                        if i > num - 1 or j > num - 1:
                            block_dct[i, j] = 0
                # # 量化
                # block_dct = quantify(block_dct,Y_quantify_matrix)

                block_idct = cv.dct(block_dct)  # 对每一块做反DCT变换
                blocks_dct.append(block_dct)
                blocks_idct.append(block_idct)

            matrix_dct = self.merge(blocks_dct,img_size,split_size=split_size) # 丢弃高频系数后的整张图的dct系数矩阵
            matrix_idct = self.merge(blocks_idct, img_size,split_size=split_size)  # 丢弃高频系数后的整张图的反dct变换后的像素矩阵
            return matrix_dct,matrix_idct
        elif type == 4:
            # matrix 整张图的像素矩阵！
            blocks = self.split(matrix, split_size=split_size)  # 切分成块,blocks是list,block是array
            blocks_dct, blocks_idct = [], []

            for i in range(len(blocks)):
                block_dct = cv.dct(blocks[i])  # 对每一块做DCT变换

                z_list = self.zigzag(block_dct)  # 对每一块做Z字扫描
                # 只保留Z扫描的前num位
                for i in range(len(z_list)):
                    if i > num:
                        z_list[i] = 0
                block_dct = self.izigzag(z_list)  # 丢弃高频系数的每一块的dct系数矩阵

                # # 量化
                # block_dct = quantify(block_dct, Y_quantify_matrix)

                block_idct = cv.dct(block_dct)  # 对每一块做反DCT变换
                blocks_dct.append(block_dct)
                blocks_idct.append(block_idct)

            matrix_dct = self.merge(blocks_dct, img_size, split_size=split_size)  # 丢弃高频系数后的整张图的dct系数矩阵
            matrix_idct = self.merge(blocks_idct, img_size, split_size=split_size)  # 丢弃高频系数后的整张图的反dct变换后的像素矩阵
            return matrix_dct, matrix_idct

    # 输入array,返回list列表,list的每一个元素是array
    def split(self,matrix,split_size=8):
        blocks = []  # 用来装划分的块，之后用来复原

        img_block = np.zeros((split_size,split_size))
        shape_1 = matrix.shape[0] # 512
        # shape_1 = len(matrix[0]) # 512
        shape_2 = img_block.shape[0] # 64

        # 分块
        m,n = 0,0
        for i in range(0,int(shape_1/shape_2)):
            for j in range(0,int(shape_1/shape_2)):
                if n + shape_2 - 1 <= shape_1 - 1:
                    img_block = matrix[m:m+shape_2,n:n+shape_2]
                    blocks.append(img_block) # 放到数组里
                    n += shape_2
            if m + shape_2 - 1 <= shape_1 - 1:
                m += shape_2
                n = 0
        return blocks

    # 输入list，返回合并后的array
    def merge(self,blocks,img_size,split_size=8):
        img_block = np.zeros((split_size, split_size))
        shape_1 = img_size  # 512
        shape_2 = img_block.shape[0]  # 8
        num = int(shape_1/shape_2)
        # print('num',num)
        # 合并
        # 一行做num-1次合并
        # 一共做num行
        rows = []
        for i in range(0,num):
            row = blocks[i * num]
            for j in range(1,num):
                row = np.hstack((row, blocks[i * num + j]))
                # print(row)
            rows.append(row)
            # print(rows)
        # 合并行为全部图像
        num = len(rows)-1 # 一共做这么些次合并
        j = 0
        img = rows[0]
        while j<num:
            img = np.vstack((img, rows[j + 1]))
            j += 1
        return img

    def zigzag(self,array):
        matrix = list(array)
        n = len(matrix[0])

        result = []

        i = 0
        j = 0
        flag = ""
        # print(matrix[i][j], end=" ")
        result.append(matrix[i][j])
        flag = "right"
        while True:
            if flag == "right":
                j += 1
                if i == 0:
                    flag = "left-down"
                if i == n - 1 and j == n - 1:
                    # print(matrix[i][j], end=" ")
                    result.append(matrix[i][j])
                    break
                if i == n - 1:
                    flag = "right-up"
                # print(matrix[i][j], end=" ")
                result.append(matrix[i][j])
            if flag == "down":
                i += 1
                if j == 0:
                    flag = "right-up"
                if j == n - 1:
                    flag = "left-down"
                # print(matrix[i][j], end=" ")
                result.append(matrix[i][j])
            if flag == "left-down":
                i += 1
                j -= 1
                if j == 0:
                    flag = "down"
                if i == n - 1:
                    flag = "right"
                if j == 0 and i == n - 1:
                    flag = "right"
                if i != n - 1 and j != 0:
                    flag = "left-down"
                # print(matrix[i][j], end=" ")
                result.append(matrix[i][j])
            if flag == "right-up":
                i -= 1
                j += 1
                if j == n - 1:
                    flag = "down"
                if i == 0:
                    flag = "right"
                if (i == 0 and j == n - 1):
                    flag = "down"
                if (i != 0 and j != n - 1):
                    flag = "right-up"
                # print(matrix[i][j], end=" ")
                result.append(matrix[i][j])
        # print(result)
        return result

    def izigzag(self,list):
        n = int(math.sqrt(len(list)))
        matrix = np.zeros((n,n))

        i = 0
        j = 0
        t = 0
        flag = ""
        # print(matrix[i][j], end=" ")
        matrix[i][j] = list[0]
        flag = "right"
        while True:
            if flag == "right":
                j += 1
                if i == 0:
                    flag = "left-down"
                if i == n - 1 and j == n - 1:
                    # print(matrix[i][j], end=" ")
                    t +=1
                    matrix[i][j] = list[t]
                    break
                if i == n - 1:
                    flag = "right-up"
                # print(matrix[i][j], end=" ")
                t += 1
                matrix[i][j] = list[t]
            if flag == "down":
                i += 1
                if j == 0:
                    flag = "right-up"
                if j == n - 1:
                    flag = "left-down"
                # print(matrix[i][j], end=" ")
                t += 1
                matrix[i][j] = list[t]
            if flag == "left-down":
                i += 1
                j -= 1
                if j == 0:
                    flag = "down"
                if i == n - 1:
                    flag = "right"
                if j == 0 and i == n - 1:
                    flag = "right"
                if i != n - 1 and j != 0:
                    flag = "left-down"
                # print(matrix[i][j], end=" ")
                t += 1
                matrix[i][j] = list[t]
            if flag == "right-up":
                i -= 1
                j += 1
                if j == n - 1:
                    flag = "down"
                if i == 0:
                    flag = "right"
                if (i == 0 and j == n - 1):
                    flag = "down"
                if (i != 0 and j != n - 1):
                    flag = "right-up"
                # print(matrix[i][j], end=" ")
                t += 1
                matrix[i][j] = list[t]
        # print(matrix)
        return matrix

    def quantify(self,target_matrix,quantify_matrix):
        # A/B = A*B的逆
        # rint 四舍五入
        return np.rint(np.dot(target_matrix,np.linalg.inv(quantify_matrix)))

    def method1(self,img,num):
        # 1 把整张图做DCT变换，去掉一部分系数，反DCT变换
        img_dct_1 = cv.dct(img)  # 对整张图做DCT变换
        img_idct_1 = cv.idct(img_dct_1)  # 对整张图做反DCT变换

        # num = 200
        img_dct_1_drop = self.retain(num, type=1, matrix=img_dct_1)  # 去掉一部分系数
        img_idct_1_drop = cv.idct(img_dct_1_drop)  # 反DCT变换
        return img_dct_1_drop,img_idct_1_drop

    def method2(self,img,num):
        # 2 把整张图做DCT变换，Z字扫描，去掉一部分系数，合并成整个数组，反DCT变换
        # num = 1000
        img_dct_1 = cv.dct(img)  # 对整张图做DCT变换
        img_dct_2_drop = self.retain(num, type=2, matrix=img_dct_1)  # 去掉一部分系数
        img_idct_2_drop = cv.idct(img_dct_2_drop)  # 反DCT变换
        return img_dct_2_drop,img_idct_2_drop

    def method3(self,img,num,split_size):
        # 3 把整张图分块，每块做DCT变换，每块去掉一部分系数，合成整张图，对每一块反DCT变换
        # num = 48
        img_dct_3_drop, img_idct_3_drop = self.retain(num, type=3, matrix=img, split_size=split_size)  # 去掉一部分系数
        return img_dct_3_drop, img_idct_3_drop

    def method4(self,img,num,split_size):
        # 4 把整张图分块，每块做DCT变换，每块Z字扫描，去掉一部分系数，合并整个数组，合成整张图，反DCT变换
        # num = 48
        img_dct_4_drop, img_idct_4_drop =self.retain(num, type=4, matrix=img, split_size=split_size)  # 去掉一部分系数
        return img_dct_4_drop, img_idct_4_drop

if __name__ == "__main__":
    img = cv.imread('./images/lena256.jpg',0) # 直接读取灰度图像
    cv.imshow('original',img)
    img = np.float32(img)/255.0 # 转成浮点数
    # print(img_1)
    # img = np.array(img,dtype=np.float32)

    # array = np.arange(0,64).reshape((8,8))
    # array = np.array(array,dtype=np.float32)
    # print(array)
    # dct,idct = retain(2,type=3,matrix=array,split_size=2)
    # print('dct',dct)
    # print('idct',idct)

    # blocks = split(array,split_size=2)
    # print('blocks',type(blocks),blocks)
    # img = merge(blocks,8,split_size=2)
    # print('img', type(img), img)

    trans = Transform()

    img_dct_1 = cv.dct(img)  # 对整张图做DCT变换
    img_idct_1 = cv.idct(img_dct_1)  # 对整张图做反DCT变换
    # 展示
    imgs = np.hstack([img_dct_1, img_idct_1])
    cv.imshow("dct and idct for whole image", imgs)

    # 1 把整张图做DCT变换，去掉一部分系数，反DCT变换
    num = 200
    img_dct_1_drop,img_idct_1_drop = trans.method1(img,num=200)
    # 展示
    imgs = np.hstack([img_dct_1_drop, img_idct_1_drop])
    cv.imshow("dct and idct for whole image with abandon", imgs)

    # 2 把整张图做DCT变换，Z字扫描，去掉一部分系数，合并成整个数组，反DCT变换
    num = 1000
    img_dct_2_drop,img_idct_2_drop= trans.method2(img,num=1000)
    # 展示
    imgs = np.hstack([img_dct_2_drop, img_idct_2_drop])
    cv.imshow("dct and idct for whole image with zigzag abandon", imgs)

    # 3 把整张图分块，每块做DCT变换，每块去掉一部分系数，合成整张图，对每一块反DCT变换
    num = 48
    img_dct_3_drop,img_idct_3_drop = trans.method3(img,num=48,split_size=16)
    # 展示
    imgs = np.hstack([img_dct_3_drop, img_idct_3_drop])
    cv.imshow("dct and idct for block with abandon", imgs)

    # 4 把整张图分块，每块做DCT变换，每块Z字扫描，去掉一部分系数，合并整个数组，合成整张图，反DCT变换
    num = 48
    img_dct_4_drop,img_idct_4_drop = trans.method4(img,num=48,split_size=16)
    # 展示
    imgs = np.hstack([img_dct_4_drop, img_idct_4_drop])
    cv.imshow("dct and idct for block with zigzag abandon", imgs)

    cv.waitKey(0)
    cv.destroyAllWindows()