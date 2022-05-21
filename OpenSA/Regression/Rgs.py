"""
    -*- coding: utf-8 -*-
    @Time   :2022/04/12 17:10
    @Author : Pengyou FU
    @blogs  : https://blog.csdn.net/Echo_Code?spm=1000.2115.3001.5343
    @github : https://github.com/FuSiry/OpenSA
    @WeChat : Fu_siry
    @License：Apache-2.0 license

"""

from Regression.ClassicRgs import Pls, Anngression, Svregression, ELM
from Regression.CNN import CNNTrain

def  QuantitativeAnalysis(model, X_train, X_test, y_train, y_test):

    if model == "Pls":
        Rmse, R2, Mae = Pls(X_train, X_test, y_train, y_test)
    elif model == "ANN":
        Rmse, R2, Mae = Anngression(X_train, X_test, y_train, y_test)
    elif model == "SVR":
        Rmse, R2, Mae = Svregression(X_train, X_test, y_train, y_test)
    elif model == "ELM":
        Rmse, R2, Mae = ELM(X_train, X_test, y_train, y_test)
    elif model == "CNN":
        Rmse, R2, Mae = CNNTrain("AlexNet",X_train, X_test, y_train, y_test,  150)
    else:
        print("no this model of QuantitativeAnalysis")

    return Rmse, R2, Mae 