"""
    -*- coding: utf-8 -*-
    @Time   :2022/04/12 17:10
    @Author : Pengyou FU
    @blogs  : https://blog.csdn.net/Echo_Code?spm=1000.2115.3001.5343
    @github : https://github.com/FuSiry/OpenSA
    @WeChat : Fu_siry
    @License：Apache-2.0 license

"""


from sklearn import linear_model
import numpy as np

def Lar(X, y, nums=40):
    '''
           X : 预测变量矩阵
           y ：标签
           nums : 选择的特征点的数目，默认为40
           return ：选择变量集的索引
    '''
    Lars = linear_model.Lars()
    Lars.fit(X, y)
    corflist = np.abs(Lars.coef_)

    corf = np.asarray(corflist)
    SpectrumList = corf.argsort()[-1:-(nums+1):-1]
    SpectrumList = np.sort(SpectrumList)

    return SpectrumList