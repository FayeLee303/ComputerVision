from test_DCT import Transform # 导入模块

"""支持向量机SMO算法"""
import numpy as np
import pandas as pd
# from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
import struct

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
    # plt.imshow(images[i],'gray')
    # plt.pause(0.00001)
    # plt.show()
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
    def __init__(self,max_iter=100,kernel='linear'):
        self.max_iter = max_iter # 迭代次数
        self._kernel = kernel # 内部访问

    # 初始化变量
    def init_args(self,features,labels):
        # self.m, self.n = features.shape # m*n的图？
        self.m, self.n = features.shape[1], features.shape[2] # m*n的图？
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
            # h = K11 + K22 - 2K12 = (φ(x1)-φ(x2)) 的范数！
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
        return 1 if r >0 else -1

    # 准确率
    def score(self,x_test,y_test):
        right = 0
        for i in range(len(x_test)):
            result = self.predict(x_test[i]) # 做预测
            if result == y_test[i]:
                right += 1
        print('accuracy rate:',right / len(x_test))
        return right / len(x_test)

    # 权重
    def _weight(self):
        # linear model
        yx = self.Y.reshape(-1,1)*self.X
        self.w = np.dot(yx.T,self.alpha)
        return self.w


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

    # print('train_images',train_images.shape,train_images[0])

    print('='*50)
    trans = Transform()
    for i in range(train_images.shape[0]):
        img_dct_4_drop, img_idct_4_drop = trans.method4(train_images[i], num=20, split_size=7)
        train_images[i] = img_idct_4_drop
    print('dct transform done')

    # # 查看前十个数据及其标签以读取是否正确
    # for i in range(10):
    #     print(train_labels[i])
    #     plt.imshow(train_images[i], cmap='gray')
    #     # plt.pause(0.000001)
    # plt.show()
    #
    # print('done')

    # svm = SVM_SMO(max_iter=200)
    # svm.fit(train_images,train_labels)
    # # svm.predict([4.4,3.2,1.3,0.2])
    # svm.score(test_images,test_labels)
