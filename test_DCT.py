import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
from PIL import Image


"""
二维DCT变换
F = A*f*A.T
f = A.T*F*A 
A是转换矩阵
f是N*N的图像
A[i,j] = c(i)*cos((j+0.5)*pi/N*i)
"""

"""
    # 使用try来对YUV和灰度图像分别处理
    try:
        y, u, v = cv.split(img)
    except ValueError:
        # print("不能分离通道")
        # 此处对不能分离通道的灰度图像做处理
    else:
        # 没有异常才会执行的代码
        # 此处对分离后的yuv三个图像做处理
    finally:
        # 无论是否有异常都会执行的代码
        # 返回处理过的图像
"""

class Transform(object):

    # 折叠
    # region 单例模式
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
    # endregion

    # 亮度量化矩阵
    Y_quantify_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                  [12, 12, 14, 19, 26, 58, 60, 55],
                                  [14, 13, 16, 24, 40, 57, 69, 56],
                                  [14, 17, 22, 29, 51, 87, 80, 62],
                                  [18, 22, 37, 56, 68, 109, 103, 77],
                                  [24, 35, 55, 64, 81, 104, 113, 92],
                                  [49, 64, 78, 87, 103, 121, 120, 101],
                                  [72, 92, 95, 98, 112, 100, 103, 99]
                                  ])

    # 色度量化矩阵
    U_quantify_matrix = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                                  [18, 21, 26, 66, 99, 99, 99, 99],
                                  [24, 26, 56, 99, 99, 99, 99, 99],
                                  [47, 99, 99, 99, 99, 99, 99, 99],
                                  [99, 99, 99, 99, 99, 99, 99, 99],
                                  [99, 99, 99, 99, 99, 99, 99, 99],
                                  [99, 99, 99, 99, 99, 99, 99, 99],
                                  [99, 99, 99, 99, 99, 99, 99, 99]
                                  ])

    # 对图像进行预处理，如果长宽不相等，就在补最后一行或这最后一列的数据
    # 不是图像长宽不相等，而是不能整除8！！
    # 输出 长宽相等，并且都能整除8的方阵
    def pre_process(self,img,split_size=8):
        # 转成浮点数
        # img = np.array(img, dtype=np.float32) / 255.0

        m,n = img.shape[0],img.shape[1]

        # 如果行列小于8，不管能不能整除都补成8*8
        # 行补最后一行，列补最后一列
        if m < split_size and n < split_size:
            temp_r = split_size - m
            temp_c = split_size - n

            column = img[:, -1]  # 最后一列的数据,这里得到的是行向量！！！
            column = column.reshape((img.shape[0], 1))  # 变成列向量！
            c = np.tile(column, (1, temp_c))
            img = np.hstack((img, c))  # 横向拼接

            row = img[-1, :]  # 最后一行的数据
            r = np.tile(row, (temp_r, 1))
            img = np.vstack((img, r))  # 纵向拼接

            # print(img.shape)
            return np.array(img)
        else:
            # 行列相等
            if m == n:
                # 行列都能整除8
                if m%split_size == 0:
                    return img.shape

                # 不能整除8，行列补一样的
                else:
                    temp = split_size - m % split_size
                    # print('行列都补%d',temp)

                    row = img[-1, :]  # 最后一行的数据
                    r = np.tile(row, (temp, 1))
                    img = np.vstack((img, r))  # 纵向拼接

                    column = img[:, -1]  # 最后一列的数据,这里得到的是行向量！！！
                    column = column.reshape((img.shape[0], 1))  # 变成列向量！
                    c = np.tile(column, (1, temp))
                    img = np.hstack((img, c))  # 横向拼接

                    # print(img.shape)
                    return np.array(img)
            # 行列不相等
            else:
                # 行能整除，列不能整除
                if m%split_size==0 and n%split_size!=0:
                    # 行比列多，补 行-列 个列
                    if m > n:
                        temp_c = m - n
                        # print('补%d个列',temp_c)

                        column = img[:, -1]  # 最后一列的数据,这里得到的是行向量！！！
                        column = column.reshape((img.shape[0], 1))  # 变成列向量！
                        c = np.tile(column, (1, temp_c))
                        img = np.hstack((img, c))  # 横向拼接

                        # print(img.shape)
                        return np.array(img)

                    # 列比行多，补列也要补行
                    else:
                        temp_c = split_size - n % split_size
                        temp_r = n + temp_c - m
                        # print('补%d个列',temp_c)
                        # print('补%d个行',temp_r)

                        column = img[:, -1]  # 最后一列的数据,这里得到的是行向量！！！
                        column = column.reshape((img.shape[0], 1))  # 变成列向量！
                        c = np.tile(column, (1, temp_c))
                        img = np.hstack((img, c))  # 横向拼接

                        row = img[-1, :]  # 最后一行的数据
                        r = np.tile(row, (temp_r, 1))
                        img = np.vstack((img, r))  # 纵向拼接

                        # print(img.shape)
                        return np.array(img)

                # 列能整除，行不能整除
                elif m%split_size!=0 and n%split_size==0:
                    # 列比行多，补 列-行 个行
                    if n > m:
                        temp_r = n - m
                        # print('补%d个行', temp_r)

                        row = img[-1, :]  # 最后一行的数据
                        r = np.tile(row, (temp_r, 1))
                        img = np.vstack((img, r))  # 纵向拼接

                        # print(img.shape)
                        return np.array(img)

                    # 列比行少，补行也要补列
                    else:
                        temp_r = split_size - m % split_size
                        temp_c = m + temp_r - n

                        # print('补%d个列', temp_c)
                        # print('补%d个行', temp_r)

                        column = img[:, -1]  # 最后一列的数据,这里得到的是行向量！！！
                        column = column.reshape((img.shape[0], 1))  # 变成列向量！
                        c = np.tile(column, (1, temp_c))
                        img = np.hstack((img, c))  # 横向拼接

                        row = img[-1, :]  # 最后一行的数据
                        r = np.tile(row, (temp_r, 1))
                        img = np.vstack((img, r))  # 纵向拼接

                        # print(img.shape)
                        return np.array(img)

                # 行不能整除，列也不能整除
                elif m%split_size!=0 and n%split_size!=0:
                    temp_r,temp_c = 0,0
                    # 看多的
                    if m > n:
                        temp_r = split_size - m % split_size
                        temp_c = m + temp_r - n
                    elif m < n:
                        temp_c = split_size - n % split_size
                        temp_r = n + temp_c - m

                        # print('补%d个列', temp_c)
                        # print('补%d个行', temp_r)

                    column = img[:, -1]  # 最后一列的数据,这里得到的是行向量！！！
                    column = column.reshape((img.shape[0], 1))  # 变成列向量！
                    c = np.tile(column, (1, temp_c))
                    img = np.hstack((img, c))  # 横向拼接

                    row = img[-1, :]  # 最后一行的数据
                    r = np.tile(row, (temp_r, 1))
                    img = np.vstack((img, r))  # 纵向拼接

                    # print(img.shape)
                    return np.array(img)


    # 得到变换矩阵
    def create_DCT_transform_matrix(self, matrix):
        N = matrix.shape[0] # 图像大小
        A = np.zeros((N,N)) # 变换矩阵
        for i in range(N):
            for j in range(N):
                if i == 0:
                    c = math.sqrt(1/N)
                else:
                    c = math.sqrt(2/N)
                A[i, j] = c * math.cos((((j + 0.5) * math.pi) / N) * i)
        return A

    def DCT(self,matrix):
        A = self.create_DCT_transform_matrix(matrix)
        F = np.dot(np.dot(A, matrix), A.T)
        return F

    def iDCT(self,matrix):
        A = self.create_DCT_transform_matrix(matrix)
        f = np.dot(np.dot(A.T, matrix), A)
        return f

    # 保留一部分系数，其他部分置0
    # type=1是对矩阵取左上角
    # type=2是对z字扫描后的一维数组取前几位再变成矩阵
    def retain(self,num,type=1,matrix=None):
        if type == 1:
            matrix = np.array(matrix)  # dct系数矩阵
            # 取左上角
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[0]):
                    if i > num - 1 or j > num - 1:
                        matrix[i, j] = 0
            return matrix  # 丢弃高频系数后的dct系数矩阵

        if type == 2:
            matrix = np.array(matrix)  # dct系数矩阵
            z_list = self.zigzag(matrix)  # Z字扫描
            # 只保留Z扫描的前num位
            for i in range(len(z_list)):
                if i > num:
                    z_list[i] = 0
            matrix = self.izigzag(z_list)  # 丢弃高频系数后的dct系数矩阵
            return matrix


    # 输入array,返回list列表,list的每一个元素是array
    def split(self,matrix,split_size=8):
        blocks = []  # 用来装划分的块，之后用来复原

        img_block = np.zeros((split_size,split_size))
        shape_1 = matrix.shape[0]
        shape_2 = img_block.shape[0]

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
        # 合并
        # 一行做num-1次合并
        # 一共做num行
        rows = []
        for i in range(0,num):
            row = blocks[i * num]
            for j in range(1,num):
                row = np.hstack((row, blocks[i * num + j]))
            rows.append(row)
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
        i, j = 0,0
        flag = ""
        result.append(matrix[i][j])
        flag = "right"
        while True:
            if flag == "right":
                j += 1
                if i == 0:
                    flag = "left-down"
                if i == n - 1 and j == n - 1:
                    result.append(matrix[i][j])
                    break
                if i == n - 1:
                    flag = "right-up"
                result.append(matrix[i][j])
            if flag == "down":
                i += 1
                if j == 0:
                    flag = "right-up"
                if j == n - 1:
                    flag = "left-down"
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
                result.append(matrix[i][j])
        return result

    def izigzag(self,list):
        n = int(math.sqrt(len(list)))
        matrix = np.zeros((n,n))

        i = 0
        j = 0
        t = 0
        flag = ""
        matrix[i][j] = list[0]
        flag = "right"
        while True:
            if flag == "right":
                j += 1
                if i == 0:
                    flag = "left-down"
                if i == n - 1 and j == n - 1:
                    t +=1
                    matrix[i][j] = list[t]
                    break
                if i == n - 1:
                    flag = "right-up"
                t += 1
                matrix[i][j] = list[t]
            if flag == "down":
                i += 1
                if j == 0:
                    flag = "right-up"
                if j == n - 1:
                    flag = "left-down"
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
                t += 1
                matrix[i][j] = list[t]
        return matrix

    # 量化
    def quantify(self,target_matrix,quantify_matrix):
        # A/B = A*B的逆
        # rint 四舍五入
        target_matrix = np.rint(np.dot(target_matrix,np.linalg.inv(quantify_matrix)))
        return target_matrix

    # 取整
    def round(self,matrix):
        return np.rint(matrix)


    # 返回降维后提取的特征向量feature_vector，是一维序列！！
    # 例如原来28*28，保留15*15的参数，得到1*225的特征向量！
    def extract_feature_vector(self,matrix):
        print(matrix)

        feature_vector = []
        # 把不为0的部分加到特征向量里去！！
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[0]):
                if matrix[i, j] != 0:
                    feature_vector.append(matrix[i, j])
        # 要得到一个行向量！
        return np.reshape(np.array(feature_vector),(1,-1))


    """
    # num是保留DCT系数的位数
    # isQuantify是是否进行量化，默认是，量化就是除以量化矩阵
    """

    # 1 把整张图做DCT变换，去掉一部分系数，再反DCT变换
    def method1(self,img,num,isQuantify=False):
        # 预处理
        self.pre_process(img)
        N = img.shape[0]

        # 初始化
        img_dct_1_drop, img_idct_1_drop = np.zeros((N,N)), np.zeros((N,N))

        # 使用try来对YUV和Gray图像分别处理
        try:
            y,u,v = cv.split(img)
            # img = cv.merge([y,u,v])
        except ValueError:
            # 出现异常，也就是不能分离通道的单通道灰色图像
            # print("不能分离通道")
            # 此处对不能分离通道的灰度图像做处理

            # print("处理单通道灰色图像")
            img_dct_1 = self.DCT(img)  # 对整张图做DCT变换
            img_dct_1_drop = self.retain(num, type=1, matrix=img_dct_1)  # 去掉一部分系数
            img_idct_1_drop = self.iDCT(img_dct_1_drop)  # 反DCT变换

        else:
            # 没有异常才会执行的代码
            # 此处对分离后的yuv三个图像做处理
            # print("处理3通道YUV图像")

            img_dct_1_drop_list, img_idct_1_drop_list = [], []
            for matrix in [y,u,v]:
                # 对分离后的每个分量做DCT变换
                matrix_dct = self.DCT(matrix)
                matrix_dct_drop = self.retain(num, type=1, matrix=matrix_dct)  # 去掉一部分系数
                matrix_idct_drop = self.iDCT(matrix_dct_drop)  # 反DCT变换

                img_dct_1_drop_list.append(matrix_dct_drop)
                img_idct_1_drop_list.append(matrix_idct_drop)

            # 合并
            img_dct_1_drop = cv.merge(img_dct_1_drop_list)
            img_idct_1_drop = cv.merge(img_idct_1_drop_list)

            # # # 为了方便看，从YUV再转为BGR
            # img_dct_2_drop = cv.cvtColor(img_dct_2_drop, cv.COLOR_YCrCb2BGR)
            # img_idct_2_drop = cv.cvtColor(img_idct_2_drop, cv.COLOR_YCrCb2BGR)

        finally:
            # 无论是否有异常都会执行的代码
            return img_dct_1_drop, img_idct_1_drop

    # 2 把整张图做DCT变换，Z字扫描，去掉一部分系数，合并成整个数组，反DCT变换
    def method2(self,img,num,isQuantify=False):
        # 预处理
        self.pre_process(img)
        N = img.shape[0]

        # 初始化
        img_dct_2_drop, img_idct_2_drop = np.zeros((N,N)), np.zeros((N,N))

        # 使用try来对YUV和Gray图像分别处理
        try:
            y,u,v = cv.split(img)
            # img = cv.merge([y,u,v])
        except ValueError:
            # 出现异常，也就是不能分离通道的单通道灰色图像
            # print("不能分离通道")
            # 此处对不能分离通道的灰度图像做处理

            # print("处理单通道灰色图像")
            img_dct_2 = self.DCT(img)  # 对整张图做DCT变换
            img_dct_2_drop = self.retain(num, type=2, matrix=img_dct_2)  # 去掉一部分系数
            img_idct_2_drop = self.iDCT(img_dct_2_drop)  # 反DCT变换

        else:
            # 没有异常才会执行的代码
            # 此处对分离后的yuv三个图像做处理
            # print("处理3通道YUV图像")

            img_dct_2_drop_list, img_idct_2_drop_list = [], []
            for matrix in [y,u,v]:
                # 对分离后的每个分量做DCT变换
                matrix_dct = self.DCT(matrix)
                matrix_dct_drop = self.retain(num, type=2, matrix=matrix_dct)  # 去掉一部分系数
                matrix_idct_drop = self.iDCT(matrix_dct_drop)  # 反DCT变换

                img_dct_2_drop_list.append(matrix_dct_drop)
                img_idct_2_drop_list.append(matrix_idct_drop)

            # 合并
            img_dct_2_drop = cv.merge(img_dct_2_drop_list)
            img_idct_2_drop = cv.merge(img_idct_2_drop_list)

            # # # 为了方便看，从YUV再转为BGR
            # img_dct_2_drop = cv.cvtColor(img_dct_2_drop, cv.COLOR_YCrCb2BGR)
            # img_idct_2_drop = cv.cvtColor(img_idct_2_drop, cv.COLOR_YCrCb2BGR)

        finally:
            # 无论是否有异常都会执行的代码
            return img_dct_2_drop, img_idct_2_drop

    # 3 把整张图分块，每块做DCT变换，每块去掉一部分系数，合成整张图，对每一块反DCT变换
    def method3(self,img,num,split_size=8,isQuantify=False):
        # 预处理
        self.pre_process(img)
        N = img.shape[0]

        # 初始化
        img_dct_3_drop, img_idct_3_drop = np.zeros((N,N)), np.zeros((N,N))

        # 使用try来对YUV和Gray图像分别处理
        try:
            y, u, v = cv.split(img)
            # img = cv.merge([y,u,v])
        except ValueError:
            # 出现异常，也就是不能分离通道的单通道灰色图像
            # print("不能分离通道")
            # 此处对不能分离通道的灰度图像做处理

            # print("处理单通道灰色图像")
            blocks = self.split(img, split_size=split_size)  # 切分成块,blocks是list,block是array
            blocks_dct, blocks_idct = [], []
            for i in range(len(blocks)):
                block_dct = self.DCT(blocks[i])  # 对每一块做DCT变换
                block_dct = self.retain(num, type=1, matrix=block_dct)  # 丢弃一部分系数
                block_idct = self.iDCT(block_dct)  # 对每一块做反DCT变换

                blocks_dct.append(block_dct)
                blocks_idct.append(block_idct)

            img_dct_3_drop = self.merge(blocks_dct, N, split_size=split_size)  # 丢弃高频系数后的整张图的dct系数矩阵
            img_idct_3_drop = self.merge(blocks_idct, N, split_size=split_size) # 反变换后的像素矩阵

        else:
            # 没有异常才会执行的代码
            # 此处对分离后的yuv三个图像做处理
            # print("处理3通道YUV图像")

            # 每个分量的dct系数矩阵和反变换后的像素矩阵
            y_matrix_dct, y_matrix_idct = None, None
            u_matrix_dct, u_matrix_idct = None, None
            v_matrix_dct, v_matrix_idct = None, None

            # 对每个通道做处理
            for matrix in [y, u, v]:
                # matrix 整张图的像素矩阵！
                blocks = self.split(matrix, split_size=split_size)  # 切分成块,blocks是list,block是array
                blocks_dct, blocks_idct = [], []
                for i in range(len(blocks)):
                    block_dct = self.DCT(blocks[i])  # 对每一块做DCT变换
                    block_dct = self.retain(num, type=1, matrix=block_dct)  # 丢弃一部分系数

                    # 量化
                    if isQuantify is True:
                        if matrix is y:
                            block_dct = self.quantify(block_dct,self.Y_quantify_matrix)
                        elif matrix is u:
                            block_dct = self.quantify(block_dct, self.U_quantify_matrix)
                        elif matrix is v:
                            block_dct = self.quantify(block_dct, self.U_quantify_matrix)

                    block_idct = self.iDCT(block_dct)  # 对每一块做反DCT变换

                    blocks_dct.append(block_dct)
                    blocks_idct.append(block_idct)
                # 合并矩阵
                if matrix is y:
                    y_matrix_dct = self.merge(blocks_dct, N, split_size=split_size)
                    y_matrix_idct = self.merge(blocks_idct, N,split_size=split_size)
                elif matrix is u:
                    u_matrix_dct = self.merge(blocks_dct, N, split_size=split_size)
                    u_matrix_idct = self.merge(blocks_idct, N, split_size=split_size)
                elif matrix is v:
                    v_matrix_dct = self.merge(blocks_dct, N, split_size=split_size)
                    v_matrix_idct = self.merge(blocks_idct, N, split_size=split_size)

            # 合并通道
            img_dct_3_drop = cv.merge([y_matrix_dct, u_matrix_dct, v_matrix_dct])  # 合并通道
            img_idct_3_drop = cv.merge([y_matrix_idct, u_matrix_idct, v_matrix_idct])  # 合并通道

        finally:
            # 无论是否有异常都会执行的代码
            return img_dct_3_drop, img_idct_3_drop

    # 4 把整张图分块，每块做DCT变换，每块Z字扫描，去掉一部分系数，合并整个数组，合成整张图，反DCT变换
    def method4(self,img,num,split_size=8,isQuantify=False):
        # 预处理
        self.pre_process(img)
        N = img.shape[0]

        # 初始化
        img_dct_4_drop, img_idct_4_drop = np.zeros((N,N)), np.zeros((N,N))

        # 使用try来对YUV和Gray图像分别处理
        try:
            y, u, v = cv.split(img)
            # img = cv.merge([y,u,v])
        except ValueError:
            # 出现异常，也就是不能分离通道的单通道灰色图像
            # print("不能分离通道")
            # 此处对不能分离通道的灰度图像做处理

            # print("处理单通道灰色图像")
            blocks = self.split(img, split_size=split_size)  # 切分成块,blocks是list,block是array
            blocks_dct, blocks_idct = [], []
            for i in range(len(blocks)):
                block_dct = self.DCT(blocks[i])  # 对每一块做DCT变换
                block_dct = self.retain(num, type=2, matrix=block_dct)  # 丢弃一部分系数
                block_idct = self.iDCT(block_dct)  # 对每一块做反DCT变换

                blocks_dct.append(block_dct)
                blocks_idct.append(block_idct)

            img_dct_4_drop = self.merge(blocks_dct, N, split_size=split_size)  # 丢弃高频系数后的整张图的dct系数矩阵
            img_idct_4_drop = self.merge(blocks_idct, N, split_size=split_size) # 反变换后的像素矩阵

        else:
            # 没有异常才会执行的代码
            # 此处对分离后的yuv三个图像做处理
            # print("处理3通道YUV图像")

            # 每个分量的dct系数矩阵和反变换后的像素矩阵
            y_matrix_dct, y_matrix_idct = None, None
            u_matrix_dct, u_matrix_idct = None, None
            v_matrix_dct, v_matrix_idct = None, None

            # 对每个通道做处理
            for matrix in [y, u, v]:
                # matrix 整张图的像素矩阵！
                blocks = self.split(matrix, split_size=split_size)  # 切分成块,blocks是list,block是array
                blocks_dct, blocks_idct = [], []
                for i in range(len(blocks)):
                    block_dct = self.DCT(blocks[i])  # 对每一块做DCT变换
                    block_dct = self.retain(num, type=2, matrix=block_dct)  # 丢弃一部分系数

                    # 量化
                    if isQuantify is True:
                        if matrix is y:
                            block_dct = self.quantify(block_dct,self.Y_quantify_matrix)
                        elif matrix is u:
                            block_dct = self.quantify(block_dct, self.U_quantify_matrix)
                        elif matrix is v:
                            block_dct = self.quantify(block_dct, self.U_quantify_matrix)

                    block_idct = self.iDCT(block_dct)  # 对每一块做反DCT变换

                    blocks_dct.append(block_dct)
                    blocks_idct.append(block_idct)
                # 合并矩阵
                if matrix is y:
                    y_matrix_dct = self.merge(blocks_dct, N, split_size=split_size)
                    y_matrix_idct = self.merge(blocks_idct, N,split_size=split_size)
                elif matrix is u:
                    u_matrix_dct = self.merge(blocks_dct, N, split_size=split_size)
                    u_matrix_idct = self.merge(blocks_idct, N, split_size=split_size)
                elif matrix is v:
                    v_matrix_dct = self.merge(blocks_dct, N, split_size=split_size)
                    v_matrix_idct = self.merge(blocks_idct, N, split_size=split_size)

            # 合并通道
            img_dct_4_drop = cv.merge([y_matrix_dct, u_matrix_dct, v_matrix_dct])  # 合并通道
            img_idct_4_drop = cv.merge([y_matrix_idct, u_matrix_idct, v_matrix_idct])  # 合并通道

        finally:
            # 无论是否有异常都会执行的代码
            return img_dct_4_drop, img_idct_4_drop

if __name__ == "__main__":
    # img = cv.imread('./images/lena256.jpg',0) # 直接读取灰度图像
    img = cv.imread('./images/lena256.jpg')
    # img = cv.imread('./images/monkey512.bmp')
    cv.imshow('original', img)
    # b, g, r = cv.split(img)
    # # 生成一个值为0的单通道数组
    # zeros = np.zeros(img.shape[:2], dtype="uint8")
    # # 分别扩展成为三通道。另外两个通道用上面的值为0的数组填充
    # # 注意opencv显示是BGR！！！
    # r = cv.merge([ zeros, zeros,r])
    # g = cv.merge([zeros, g, zeros])
    # b = cv.merge([b,zeros, zeros])
    # imgs = np.hstack([r,g,b])
    # cv.imshow('r,g,b', imgs)

    # img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    # y, cb, cr = cv.split(img)
    # # 生成一个值为0的单通道数组
    # zeros = np.zeros(img.shape[:2], dtype="uint8")
    # # 分别扩展成为三通道。另外两个通道用上面的值为0的数组填充
    # y = cv.merge([zeros, zeros, y])
    # cb = cv.merge([zeros, cb, zeros])
    # cr = cv.merge([cr, zeros, zeros])
    # imgs = np.hstack([y, cb, cr])
    # cv.imshow('y,cb,cr', imgs)

    img = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    # cv.imshow('YUV', img)
    # img = np.float32(img)/255.0 # 转成浮点数
    img = np.array(img,dtype=np.float32)/255.0

    trans = Transform()

    # y,u,v = cv.split(img)
    # test = cv.merge([y,u,v])
    # test = cv.cvtColor(test, cv.COLOR_YUV2BGR)
    # cv.imshow('test',test)

    # img_dct = trans.DCT(img)  # 对整张图做DCT变换
    # img_idct = trans.iDCT(img_dct)  # 对整张图做反DCT变换
    #
    # # 展示
    # imgs = np.hstack([img_dct, img_idct])
    # cv.imshow("dct and idct for whole image", imgs)



    # 1 把整张图做DCT变换，去掉一部分系数，反DCT变换
    num = 180
    img_dct_1_drop,img_idct_1_drop = trans.method1(img,num=num)
    # 展示
    imgs = np.hstack([img_dct_1_drop, img_idct_1_drop])
    cv.imshow("dct and idct for whole image with abandon", imgs)


    # 2 把整张图做DCT变换，Z字扫描，去掉一部分系数，合并成整个数组，反DCT变换
    num = 180*180
    img_dct_2_drop,img_idct_2_drop= trans.method2(img,num=num)
    # 展示
    imgs = np.hstack([img_dct_2_drop, img_idct_2_drop])
    cv.imshow("dct and idct for whole image with zigzag abandon", imgs)

    # 3 把整张图分块，每块做DCT变换，每块去掉一部分系数，合成整张图，对每一块反DCT变换
    num = 4
    img_dct_3_drop,img_idct_3_drop = trans.method3(img,num=num,isQuantify=False)
    # 展示
    imgs = np.hstack([img_dct_3_drop, img_idct_3_drop])
    cv.imshow("dct and idct for block with abandon", imgs)

    # 4 把整张图分块，每块做DCT变换，每块Z字扫描，去掉一部分系数，合并整个数组，合成整张图，反DCT变换
    num = 4*4
    img_dct_4_drop,img_idct_4_drop = trans.method4(img,num=num,isQuantify=False)
    # 展示
    imgs = np.hstack([img_dct_4_drop, img_idct_4_drop])
    cv.imshow("dct and idct for block with zigzag abandon", imgs)

    cv.waitKey(0)
    cv.destroyAllWindows()