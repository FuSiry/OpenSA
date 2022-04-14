import numpy as np
"""
    -*- coding: utf-8 -*-
    @Time   :2022/04/12 17:10
    @Author : Pengyou FU
    @blogs  : https://blog.csdn.net/Echo_Code?spm=1000.2115.3001.5343
    @github : https://github.com/FuSiry/OpenSA
    @WeChat : Fu_siry
    @License：Apache-2.0 license

"""


from numpy.linalg import norm
# from skimage.measure import compare_psnr, compare_ssim
# from skimage.metrics import structural_similarity as compare_ssim
# from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def Simcalculation(type,data1, data2):
    """
       :param method: 计算相似度的方法
       :param data1: 光谱数据：格式：(1，length),高光谱图像：格式：(H, W, C)
       :param data2: 光谱数据：格式：(1，length),高光谱图像：格式：(H, W, C)
       :return: 计算原始光谱数据与目标光谱数据的相似度，float
    """

    if type == 'SAM':
        return SAM(data1, data2)
    elif type == 'SID':
        return SID(data1,data2)
    elif type == 'HsiSam':
        return HsiSam(data1,data2)
    elif type == 'mssim':
        return mssim(data1,data2)
    elif type == 'mpsnr':
        return mpsnr(data1,data2)
    else:
        print("no this method of Simcalculation!")

def SAM(x,y):
    """
       :param x_true: 光谱数据：格式：(1，length)
       :param x_pred: 光谱数据：格式：(1，length)
       :return: 计算原始光谱数据与目标光谱数据的光谱角差异
    """
    s = np.sum(np.dot(x,y))
    t1 = (norm(x)) * (norm(y))
    val = s/t1
    sam = 1.0/np.cos(val)

    return sam

# 计算SID
def SID(x,y):
    """
       :param x_true: 光谱数据：格式：(1，length)
       :param x_pred: 光谱数据：格式：(1，length)
       :return: 计算原始光谱数据与目标光谱数据的光谱角差异
       References
       :param x_true: 光谱数据：格式：(1，length)
       :param x_pred: 光谱数据：格式：(1，length)
       :return: 计算原始光谱数据与目标光谱数据的光谱角差异
       References
       ----------
       ----------
       """
    p = np.zeros_like(x,dtype=np.float)
    q = np.zeros_like(y,dtype=np.float)
    Sid = 0
    for i in range(len(x)):
        p[i] = np.around((x[i]/np.sum(x)), 8)
        q[i] = np.around((y[i]/np.sum(y)), 8)
    for j in range(len(x)):
        Sid += p[j]*np.log10(p[j]/q[j])+q[j]*np.log10(q[j]/p[j])
    return Sid

def mpsnr(x_true, x_pred):
    """
        :param x_true: 高光谱图像：格式：(H, W, C)
        :param x_pred: 高光谱图像：格式：(H, W, C)
        :return: 计算原始高光谱数据与重构高光谱数据的均方误差
    """
    n_bands = x_true.shape[2]
    p = [compare_psnr(x_true[:, :, k], x_pred[:, :, k], data_range=(0, 10000)) for k in range(n_bands)]
    return np.mean(p)


def HsiSam(x_true, x_pred):
    """
        :param x_true: 高光谱图像：格式：(H, W, C)
        :param x_pred: 高光谱图像：格式：(H, W, C)
        :return: 计算原始高光谱数据与重构高光谱数据的光谱角相似度
    """
    assert x_true.ndim ==3 and x_true.shape == x_pred.shape
    sam_rad = np.zeros(x_pred.shape[0, 1])
    for x in range(x_true.shape[0]):
        for y in range(x_true.shape[1]):
            tmp_pred = x_pred[x, y].ravel()
            tmp_true = x_true[x, y].ravel()
            sam_rad[x, y] = np.arccos(tmp_pred / (norm(tmp_pred) * tmp_true / norm(tmp_true)))
    sam_deg = sam_rad.mean() * 180 / np.pi
    return sam_deg


def mssim(x_true,x_pred):
    """
        :param x_true: 高光谱图像：格式：(H, W, C)
        :param x_pred: 高光谱图像：格式：(H, W, C)
        :return: 计算原始高光谱数据与重构高光谱数据的结构相似度
    """
    SSIM = compare_ssim(im1=x_true, im2=x_pred, multichannel=True)
    return SSIM