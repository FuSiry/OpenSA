"""
    -*- coding: utf-8 -*-
    @Time   :2022/04/12 17:10
    @Author : Pengyou FU
    @blogs  : https://blog.csdn.net/Echo_Code?spm=1000.2115.3001.5343
    @github : https://github.com/FuSiry/OpenSA
    @WeChat : Fu_siry
    @License：Apache-2.0 license

"""



from sklearn.cluster import KMeans
import numpy as np

def Kmeans(data, n_clusters=10, iter_num=30):

    cluster = KMeans(n_clusters=n_clusters, random_state=0, max_iter=iter_num)
    cluster.fit(data)
    label = cluster.labels_  # 对原数据表进行类别标记

    return label

class FCM:
    def __init__(self, data, clust_num, iter_num=10, m=2) :
        self.data = data
        self.cnum = clust_num
        self.sample_num=data.shape[0]
        self.m = m
        self.dim = data.shape[-1]  # 数据最后一维度数
        Jlist=[]   # 存储目标函数计算值的

        U = self.Initial_U(self.sample_num, self.cnum)
        for i in range(0, iter_num): # 迭代次数默认为10
            C = self.Cen_Iter(self.data, U, self.cnum)
            U = self.U_Iter(U, C)
            print("第%d次迭代" %(i+1) ,end="")
            print("聚类中心",C)
            J = self.J_calcu(self.data, U, C)  # 计算目标函数
            Jlist = np.append(Jlist, J)
        self.label = np.argmax(U, axis=0)  # 所有样本的分类标签
        self.Clast = C    # 最后的类中心矩阵
        self.Jlist = Jlist  # 存储目标函数计算值的矩阵

    # 初始化隶属度矩阵U
    def Initial_U(self, sample_num, cluster_n):
        U = np.random.rand(sample_num, cluster_n)  # sample_num为样本个数, cluster_n为分类数
        row_sum = np.sum(U, axis=1)  # 按行求和 row_sum: sample_num*1
        row_sum = 1 / row_sum    # 该矩阵每个数取倒数
        U = np.multiply(U.T, row_sum)  # 确保U的每列和为1 (cluster_n*sample_num).*(sample_num*1)
        return U   # cluster_n*sample_num

    # 计算类中心
    def Cen_Iter(self, data, U, cluster_n, m):
        c_new = np.empty(shape=[0, self.dim])  # self.dim为样本矩阵的最后一维度
        for i in range(0, cluster_n):          # 如散点的dim为2，图片像素值的dim为1
            u_ij_m = U[i, :] ** m  # (sample_num,)
            sum_u = np.sum(u_ij_m)
            ux = np.dot(u_ij_m, data)  # (dim,)
            ux = np.reshape(ux, (1, self.dim))  # (1,dim)
            c_new = np.append(c_new, ux / sum_u, axis=0)   # 按列的方向添加类中心到类中心矩阵
        return c_new  # cluster_num*dim

    # 隶属度矩阵迭代
    def U_Iter(self, U, c, m):
        for i in range(0, self.cnum):
            for j in range(0, self.sample_num):
                sum = 0
                for k in range(0, self.cnum):
                    temp = (np.linalg.norm(self.data[j, :] - c[i, :]) /
                            np.linalg.norm(self.data[j, :] - c[k, :])) ** (
                                2 / (m - 1))
                    sum = temp + sum
                U[i, j] = 1 / sum

        return U

    # 计算目标函数值
    def J_calcu(self, data, U, c, m):
        temp1 = np.zeros(U.shape)
        for i in range(0, U.shape[0]):
            for j in range(0, U.shape[1]):
                temp1[i, j] = (np.linalg.norm(data[j, :] - c[i, :])) ** 2 * U[i, j] ** m

        J = np.sum(np.sum(temp1))
        print("目标函数值:%.2f" %J)
        return J

def Fcm(data, n_clusters=10, iter_num=30):

    Fcm = FCM(data, n_clusters, iter_num)
    label =Fcm.U_Iter()

    return  label

def Cluster(method, data):
    if method == "Kmeans":
        label = Kmeans(data)
    if method == "Fcm":
        label = Fcm(data)
    return label



