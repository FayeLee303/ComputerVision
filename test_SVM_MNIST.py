from test_DCT import Transform # 导入模块

"""支持向量机SMO算法"""
import numpy as np
import pandas as pd
import cv2 as cv
# from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
import struct
import collections

from sklearn.svm import SVC


# 训练集文件
train_images_idx3_ubyte_file = "./MNIST_data/train-images-idx3-ubyte"
# 训练集标签文件
train_labels_idx1_ubyte_file = "./MNIST_data/train-labels-idx1-ubyte"

# 测试集文件
test_images_idx3_ubyte_file = "./MNIST_data/t10k-images-idx3-ubyte"
# 测试集标签文件
test_labels_idx1_ubyte_file = "./MNIST_data/t10k-labels-idx1-ubyte"


def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii' #因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)  #获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
    # print(offset)
    fmt_image = '>' + str(image_size) + 'B'  #图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
    # print(fmt_image,offset,struct.calcsize(fmt_image))
    images = np.empty((num_images, num_rows, num_cols))
    #plt.figure()
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            # print('已解析 %d' % (i + 1) + '张')
            # print(offset)
            pass
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        #print(images[i])
        offset += struct.calcsize(fmt_image)
    print('解析images完毕！')
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            # print ('已解析 %d' % (i + 1) + '张')
            pass
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    print('解析labels完毕！')
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    """
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)

# def create_data():
#     iris = load_iris()
#     df = pd.DataFrame(iris.data, columns=iris.feature_names)
#     df['label'] = iris.target
#     df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
#     data = np.array(df.iloc[:100, [0, 1, -1]])
#     for i in range(len(data)):
#         if data[i, -1] == 0:
#             data[i, -1] = -1 # 把原数据里的0改成-1
#     # print(data)
#     return data[:, :2], data[:, -1] # 返回x,label



"""使用非线性可分分类器"""
"""
SMO算法
输入训练数据集，二分类，精度e
输出近似解alpha'
    取初始值alpha=0,k=0
    选取最优变化量alpha1_k,alpha2_k
        alpha1:外层循环找违反KKT条件最严重的样本点
            首先遍历所有满足0<aplhai<C的样本点，如果都满足就遍历整个训练集                
        alpha2:内层循环，找使|E1=E2|最大的点
            启发式
    求解两个变量的最优化问题，求得最优解alpha1_k+1,alpha2_k+1
    更新alpha = alpha_k+1
    更新Ei保存在列表里
    if在精度e范围内满足停止条件
        取alpha' = alpha_k+1
    else
        k = k+1
        重复
"""
class SVM_SMO(object):
    def __init__(self,positive_class_label,negetive_class_label,max_iter=100,kernel='linear'):
        self.max_iter = max_iter # 迭代次数
        self._kernel = kernel # 内部访问

        self.positive_class_label = positive_class_label # 正例标签
        self.negetive_class_label = negetive_class_label # 负例标签

    # 初始化变量
    def init_args(self,features,labels):
        self.m, self.n = features.shape # m*n的图？
        # self.m, self.n = features.shape[1], features.shape[2] # m*n的图？
        self.X = features # 输入特征
        self.Y = labels # 标签
        self.b = 0.0 # 截距

        self.alpha = np.ones(self.m) # 初始化拉格朗日乘子
        self.E = [self._E(i) for i in range(self.m)] # 把Ei保存在一个列表里
        # C是惩罚参数
        # C越大，相当于希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况
        # 对训练集测试时准确率很高，但泛化能力弱
        # C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强
        self.C = 1.0

    # KKT条件
    """
    g(X[i]) = sum(alpha[i]*Y[i]*K(X,X[i])) + b
    alpha[i]=0 等价于 Y[i]*g(X[i]) >=1 分对的？
    0<alpha[i]<C 等价于 Y[i]*g(X[i]) =1 在间隔和超平面之间
    alpha[i]=C 等价于 Y[i]*g(X[i]) <=1 分错的？
    """
    def _KKT(self,i):
        y_g = self._g(i) * self.Y[i]
        if self.alpha[i] == 0:
            return y_g >= 1
        elif 0 < self.alpha[i] < self.C:
            return y_g == 1
        else:
            return y_g <= 1

    # w是唯一的，b是不唯一的
    # g(x)预测值，输入xi，输出预测的y
    def _g(self,i):
        # r = self.b
        r = 0
        for j in range(self.m): # 对所有的样本Xi
            r += self.alpha[j] * self.Y[j] * self.kernel(self.X[i],self.X[j])
        return r + self.b

    # 核函数
    def kernel(self,x1,x2):
        if self._kernel == 'linear':
            return sum([x1[k] * x2[k] for k in range(self.n)]) # 内积
        elif self._kernel == 'ploy':
            # 多项式核函数k(x,z) = (x*z+1)的p次幂，对应p次多项式分类器
            # 决策函数是f(x)=sign(sum(alpha[i]*Y[i]*K(x[i],x[j]))+b)
            return (sum([x1[k] * x2[k] for k in range(self.n)]) + 1)**2
        elif self._kernel == 'gaussian':
            # 高斯核函数K(x,z)=exp(-(x-z的范数)/2*方差)
            # 决策函数f(x)=sign(sum(alpha[i]*Y[i]*K(x[i],x[j]))+b)
            mean = sum(self.X) / float(len(self.X))
            variance = math.sqrt(sum([pow(xi-mean,2)for xi in self.X]) / float(len(self.X)))
            return math.exp(sum(-np.linalg.norm(x1[k],x2[k],ord=2)for k in range(self.n)) / 2*variance)


    # E(x)为g(x)对输入x的预测值和y的差，可以理解成损失
    def _E(self,i):
        return self._g(i) - self.Y[i]


    # SMO算法在每个子问题中选择两个变量进行优化
    # 固定其他变量不变
    def _init_alpha(self):
        # 外层循环，首先遍历所有满足0<alpha<C的样本点，检查是否满足KKT条件
        index_list = [i for i in range(self.m) if 0<self.alpha[i]<self.C]
        # 否则遍历整个训练集
        non_satisfy_list = [i for i in range(self.m) if i not in index_list]
        index_list.extend(non_satisfy_list) # 得到了所有的训练集样本点
        for i in index_list:
            if self._KKT(i):
                continue # 直到找到不满足KKT条件的点
            E1 = self.E[i]
            # 第一轮把alpha1确定，E1随之确定
            # alpha2使得|E1-E2|最大
            # 如果E2是正的，就选最小的，如果E2是负的，就选最大的
            # 体会利用lamba排序
            if E1 >=0:
                j = min(range(self.m),key=lambda i:self.E[i])
            else:
                j = max(range(self.m),key=lambda i:self.E[i])
            return i,j # 就是要优化的两个变量alpha1 alpha2


    """
    不等式约束0<=alpha[i]<=C使得alpha1和alpha2在[0,C]*[0,C]的正方形里
    条件约束alpha1*y1 + alpha2*y2 =-sum(y[i]*alpha[i]) = 某个常数
    也就是说alpha1和alpha2在一条直线上
    用这条直线和正方形相交，得到两个交点L H
    超过的部分就取端点值，中间就取alpha
    """
    def _compare(self,_alpha,L,H):
        if _alpha > H:
            return H
        elif _alpha < L:
            return L
        else:
            return _alpha

    # 训练
    def fit(self,features,labels):
        self.init_args(features,labels) # 初始化参数
        for t in range(self.max_iter):
            i1,i2 = self._init_alpha() # 要优化的两个变量的index
            # 找到直线和正方形相交的两个端点的值
            # 这是alpha取值的边界
            # 就是把alpha的取值范围从[0,C]缩小到了[L,H]
            if self.Y[i1] == self.Y[i2]:
                L = max(0, self.alpha[i1] + self.alpha[i2] - self.C)
                H = min(self.C, self.alpha[i1] + self.alpha[i2])
            else:
                L = max(0, self.alpha[i2] - self.alpha[i1])
                H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])

            E1 = self.E[i1]
            E2 = self.E[i2]
            # SMO最优化问题的子问题沿着约束方向的未经剪辑的解是alpha2_new_unc= alpha2_old+y2(E1-E2)/h
            # eta = K11 + K22 - 2K12 = (φ(x1)-φ(x2)) 的范数！
            eta = self.kernel(self.X[i1],self.X[i1]) + \
                  self.kernel(self.X[i2],self.X[i2]) - \
                  2*self.kernel(self.X[i1],self.X[i2])
            if eta <= 0:
                # 求的范数距离应该是大于0的，如果不是就说明错了
                # 重新找alpha1 alpha2
                continue

            # 沿着约束方向的未经剪辑的解
            alpha2_new_unc = self.alpha[i2] + self.Y[i2] * (E2-E1) / eta
            # 进行剪辑，就是限制在[0,C]范围之内
            alpha2_new = self._compare(alpha2_new_unc,L,H)
            # 根据alpha2_new来求alpha1_new
            alpha1_new = self.alpha[i1] + self.Y[i1] * self.Y[i2] * (self.alpha[i2]-alpha2_new)

            # 计算阈值b
            b1_new = -E1 - self.Y[i1] * self.kernel(self.X[i1],self.X[i1]) * (alpha1_new-self.alpha[i1]) - \
                     self.Y[i2] * self.kernel(self.X[i2],self.X[i1]) * (alpha2_new-self.alpha[i2]) + self.b
            b2_new = -E2 - self.Y[i1] * self.kernel(self.X[i1], self.X[i2]) * (alpha1_new - self.alpha[i1]) - \
                     self.Y[i2] * self.kernel(self.X[i2], self.X[i2]) * (alpha2_new - self.alpha[i2]) + self.b

            # alpha1_new alpha2_new同时满足在区间[0,C],b1_new = b2_new = b
            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                # alpha1_new alpha2_new是0或者C，就是都落在端点上了
                # 选择b1_new b2_new的中点作为新的b
                b_new = (b1_new + b2_new) / 2

            # 更新参数
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new

            self.E[i1] = self._E(i1)
            self.E[i2] = self._E(i2) # 更新Ei保存在列表中

        print('train done!')


    # 预测
    def predict(self,data):
        r = self.b
        for i in range(self.m):
            # 决策函数f(x) = self.alpha[i] * self.Y[i] * self.kernel(data,self.X[i])
            r += self.alpha[i] * self.Y[i] * self.kernel(data,self.X[i])
        # return 1 if r > 0 else -1
        # 大于0返回正类标签，小于0返回负类标签
        return self.positive_class_label if r>0 else self.negetive_class_label

    # # 准确率
    # def score(self,x_test,y_test):
    #     right = 0
    #     for i in range(len(x_test)):
    #         result = self.predict(x_test[i]) # 做预测
    #         if result == y_test[i]:
    #             right += 1
    #     print('accuracy rate:',right / len(x_test))
    #     return right / len(x_test)
    #
    # # 权重
    # def _weight(self):
    #     # linear model
    #     yx = self.Y.reshape(-1,1)*self.X
    #     self.w = np.dot(yx.T,self.alpha)
    #     return self.w


# 对数据进行第一种方法处理
def create_data(Transform, images, labels,method_num=1):
    # train_images的形状N*28*28
    # 要输出为N*特征向量长度的矩阵

    # 提取特征向量
    feature_vector = []
    for i in range(images.shape[0]):
        img = images[i]  # 得到图像

        img = np.array(img, dtype=np.float32) / 255.0  # 转成浮点数

        fv = None

        if method_num ==1:
            img_dct_1 = (Transform.method1(img, num=15))[0]  # 处理
            fv = trans.extract_feature_vector(img_dct_1)  # 提取特征向量
        elif method_num ==2:
            img_dct_2 = (Transform.method1(img, num=100))[0]  # 处理
            fv = trans.extract_feature_vector(img_dct_2)  # 提取特征向量
        elif method_num == 3:
            img_dct_3 = (Transform.method1(img, num=256))[0]  # 处理
            fv = trans.extract_feature_vector(img_dct_3)  # 提取特征向量
        elif method_num == 4:
            img_dct_4 = (Transform.method1(img, num=4))[0]  # 处理
            fv = trans.extract_feature_vector(img_dct_4)  # 提取特征向量

        feature_vector.append(fv)  # 加到列表里
        if i % 5000 == 0:
            print('提取%d张图片的特征向量' % i)

    print('extract_feature_vector is done!')

    temp = len(feature_vector[0][0])

    feature_vector = np.reshape(feature_vector, (images.shape[0], temp))  # reshape

    labels = np.reshape(labels, (-1,1))  # reshape
    return feature_vector,labels

# 分成十堆
def divide_into_10(feature_vector,train_labels):
    # 分成十堆
    # 注意使用append得到的label是行向量，用的时候要转为列向量！！
    inputs_0, label_0 = [], []
    inputs_1, label_1 = [], []
    inputs_2, label_2 = [], []
    inputs_3, label_3 = [], []
    inputs_4, label_4 = [], []
    inputs_5, label_5 = [], []
    inputs_6, label_6 = [], []
    inputs_7, label_7 = [], []
    inputs_8, label_8 = [], []
    inputs_9, label_9 = [], []

    inputs = [inputs_0, inputs_1, inputs_2, inputs_3, inputs_4, inputs_5, inputs_6, inputs_7, inputs_8, inputs_9]
    labels = [label_0, label_1, label_2, label_3, label_4, label_5, label_6, label_7, label_8, label_9]
    for i in range(60000):
        if train_labels[i] == 0.0:
            inputs[0].append(feature_vector[i])
            labels[0].append('0')
        elif train_labels[i] == 1.0:
            inputs[1].append(feature_vector[i])
            labels[1].append('1')
        elif train_labels[i] == 2.0:
            inputs[2].append(feature_vector[i])
            labels[2].append('2')
        elif train_labels[i] == 3.0:
            inputs[3].append(feature_vector[i])
            labels[3].append('3')
        elif train_labels[i] == 4.0:
            inputs[4].append(feature_vector[i])
            labels[4].append('4')
        elif train_labels[i] == 5.0:
            inputs[5].append(feature_vector[i])
            labels[5].append('5')
        elif train_labels[i] == 6.0:
            inputs[6].append(feature_vector[i])
            labels[6].append('6')
        elif train_labels[i] == 7.0:
            inputs[7].append(feature_vector[i])
            labels[7].append('7')
        elif train_labels[i] == 8.0:
            inputs[8].append(feature_vector[i])
            labels[8].append('8')
        elif train_labels[i] == 9.0:
            inputs[9].append(feature_vector[i])
            labels[9].append('9')

    return inputs, labels

"""
使用SVM处理多分类问题
做法是在任意两类样本之间设计一个SVM，因此k个类别的样本就需要设计k(k-1)/2个SVM分类器
当对一个未知样本进行分类时，最后得票最多的类别即为该未知样本的类别
  优点：不需要重新训练所有的SVM，只需要重新训练和增加语音样本相关的分类器。在训练单个模型时，相对速度较快。
  缺点：所需构造和测试的二值分类器的数量关于k成二次函数增长，总训练时间和测试时间相对较慢。
如何投票：
    假设有10个类，就有10*9/2-45个分类器
    对每个样本就有45个分类结果，统计分类结果里哪个票数最多就把该样本分为哪类
"""

def OVO_SVMS(Transform,train_images,train_labels,test_images,test_labels,method_num=1):
    train_inputs, train_labels = None,None
    test_inputs, test_labels = None,None
    train_10_inputs, train_10_labels = None,None

    if method_num ==1:
        # 对图片进行处理
        train_inputs,train_labels = create_data(Transform,train_images,train_labels,method_num=1)
        test_inputs, test_labels = create_data(Transform,test_images,test_labels,method_num=1)

        # 把训练样本和标签分十类，测试的不用分类
        train_10_inputs,train_10_labels = divide_into_10(train_inputs,train_labels)


    # 从0到9取两个数字
    index_list = []
    for i in range(10):
        for j in range(10):
            if i != j and [i, j] not in index_list and [j, i] not in index_list:
                index_list.append([i, j])


    svm_list = []   # 用列表存45个分类器

    # 训练
    for k in range(len(index_list)):

        i,j = index_list[k][0],index_list[k][1] # 两个类的序号

        temp = len(train_10_inputs[i][0]) # 特征向量长度！！

        # 两类数据
        X = np.vstack([train_10_inputs[i],train_10_inputs[j]])
        Y = np.hstack([train_10_labels[i],train_10_labels[j]])
        Y = np.reshape(Y,(-1,1)) # 1列
        X = np.array(X).reshape(-1,temp)

        positive_lable = train_10_labels[i][0]
        negetive_label = train_10_labels[j][0]
        svm = SVM_SMO(positive_lable,negetive_label,max_iter=200, )  # 创建分类器

        svm.fit(X,Y) # 训练
        print('第%d个svm分类器训练完毕'%k)
        svm_list.append(svm) # 把分类器加到列表里

    print('所有svm分类器训练完毕！')

    # 测试
    predict_labels = np.zeros((10000, 1))  # 初始化预测标签值

    for i in range(10000):
        # 每个样本用45个分类器做预测，得到45个结果，按出现次数排序
        # 哪个结果的出现次数最高，就把该样本分到这个类里
        temp_result = [] # 用来装45个结果
        for j in range(len(svm_list)):
            temp_result.append(svm_list[j].predict(test_inputs[i]))
        # Counter是计数器，用于追踪值的出现次数
        # count_pairs是个字典，存了值和值出现的次数
        # 把count_pairs排序，找到出现次数最多的，就把xx分到这个类
        count_pairs = Counter(temp_result)
        max_count = sorted(count_pairs, key=lambda x: x)[-1]
        predict_labels[i] = max_count

    print('所有样本测试完毕！')

    # 得分
    right = 0
    for i in range(10000):
        if predict_labels[i] == test_labels[i]:
            right += 1
    print('accuracy rate:',right / 10000)

if __name__ == "__main__":
    # inputs,labels = create_data()
    # x_train,x_test,y_train,y_test = train_test_split(inputs,labels,test_size=0.25)
    #
    # plt.scatter(inputs[:50, 0], inputs[:50, 1], label='0')
    # plt.scatter(inputs[50:100, 0], inputs[50:100, 1], label='1')
    # plt.legend()
    # # plt.show()
    #
    # svm = SVM_SMO(max_iter=200)
    # svm.fit(x_train,y_train)
    # # svm.predict([4.4,3.2,1.3,0.2])
    # svm.score(x_test,y_test)

    # # 使用sklearn自带的函数
    # clf = SVC()
    # clf.fit(x_train,y_train)
    # print(clf.score(x_test,y_test))

    train_images = load_train_images()
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()
    trans = Transform()
    OVO_SVMS(trans,train_images,train_labels,test_images,test_labels,method_num=1)

    # 查看前几个数据及其标签以读取是否正确
    # for i in range(5):
    #     print(train_labels[i])
    #     img_original = train_images[i]
    #     # 转成浮点数
    #     img = np.array(train_images[i], dtype=np.float32) / 255.0
    #
        # img_dct_1, img_idct_1 = trans.method1(img, num=15)
        # img_dct_2, img_idct_2 = trans.method2(img, num=100)
        # img_dct_3, img_idct_3 = trans.method3(img, num=4, split_size=7)
        # img_dct_4, img_idct_4 = trans.method4(img, num=16, split_size=7)
    #
    #     fig = plt.figure()
    #     plt.subplot(151)
    #     plt.imshow(img_original,cmap='gray'),plt.title('original'),plt.xticks([]),plt.yticks([])
    #     plt.subplot(152)
    #     plt.imshow(img_idct_1, cmap='gray'), plt.title('DCT ULC'), plt.xticks([]), plt.yticks([])
    #     plt.subplot(153)
    #     plt.imshow(img_idct_2, cmap='gray'), plt.title('DCT zigzag'), plt.xticks([]), plt.yticks([])
    #     plt.subplot(154)
    #     plt.imshow(img_idct_3, cmap='gray'), plt.title('block based DCT ULC'), plt.xticks([]), plt.yticks([])
    #     plt.subplot(155)
    #     plt.imshow(img_idct_4, cmap='gray'), plt.title('block based DCT zigzag'), plt.xticks([]), plt.yticks([])
    #
    #     # imgs = np.hstack([img_original,img_idct_1,img_idct_2,img_idct_3,img_idct_4])
    #     # cv.imshow(str(train_labels[i]),imgs)

    # 查看变换后的特征向量的大小
    # print(train_labels[0])
    # img_original = train_images[0]
    #
    # # 转成浮点数
    # img = np.array(img_original, dtype=np.float32) / 255.0
    #
    # fv_original = np.reshape(img, (1,-1)) # 原始图像特征向量
    #
    # img_dct_1, img_idct_1 = trans.method1(img, num=15)
    # img_dct_2, img_idct_2 = trans.method2(img, num=100)
    # img_dct_3, img_idct_3 = trans.method3(img, num=4, split_size=7)
    # img_dct_4, img_idct_4 = trans.method4(img, num=16, split_size=7)
    #
    # fv_1 = trans.extract_feature_vector(img_dct_1)
    # fv_2 = trans.extract_feature_vector(img_idct_2)
    # fv_3 = trans.extract_feature_vector(img_idct_3)
    # fv_4 = trans.extract_feature_vector(img_idct_4)
    #
    # print('fv_original',fv_original.shape)
    # print('fv_1', fv_1.shape)
    # print('fv_2', fv_2.shape)
    # print('fv_3', fv_3.shape)
    # print('fv_4', fv_4.shape)
    #
    # fig = plt.figure()
    # plt.subplot(151)
    # plt.imshow(img_original, cmap='gray'), plt.title('original'), plt.xticks([]), plt.yticks([])
    # plt.subplot(152)
    # plt.imshow(img_idct_1, cmap='gray'), plt.title('DCT ULC'), plt.xticks([]), plt.yticks([])
    # plt.subplot(153)
    # plt.imshow(img_idct_2, cmap='gray'), plt.title('DCT zigzag'), plt.xticks([]), plt.yticks([])
    # plt.subplot(154)
    # plt.imshow(img_idct_3, cmap='gray'), plt.title('block based DCT ULC'), plt.xticks([]), plt.yticks([])
    # plt.subplot(155)
    # plt.imshow(img_idct_4, cmap='gray'), plt.title('block based DCT zigzag'), plt.xticks([]), plt.yticks([])
    #
    # plt.show()

    # cv.waitKey(0)
    # cv.destroyAllWindows()