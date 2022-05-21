"""
    -*- coding: utf-8 -*-
    @Time   :2022/04/12 17:10
    @Author : Pengyou FU
    @blogs  : https://blog.csdn.net/Echo_Code?spm=1000.2115.3001.5343
    @github : https://github.com/FuSiry/OpenSA
    @WeChat : Fu_siry
    @License：Apache-2.0 license

"""

from sklearn.preprocessing import scale,MinMaxScaler,Normalizer,StandardScaler
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.neural_network import MLPRegressor
import numpy as np


def ModelRgsevaluate(y_pred, y_true):

    mse = mean_squared_error(y_true,y_pred)
    R2  = r2_score(y_true,y_pred)
    mae = mean_absolute_error(y_true,y_pred)

    return np.sqrt(mse), R2, mae

def ModelRgsevaluatePro(y_pred, y_true, yscale):

    yscaler = yscale
    y_true = yscaler.inverse_transform(y_true)
    y_pred = yscaler.inverse_transform(y_pred)

    mse = mean_squared_error(y_true,y_pred)
    R2  = r2_score(y_true,y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return np.sqrt(mse), R2, mae