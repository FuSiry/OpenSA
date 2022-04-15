"""
    -*- coding: utf-8 -*-
    @Time   :2022/04/12 17:10
    @Author : Pengyou FU
    @blogs  : https://blog.csdn.net/Echo_Code?spm=1000.2115.3001.5343
    @github : https://github.com/FuSiry/OpenSA
    @WeChat : Fu_siry
    @License：Apache-2.0 license

"""

from Classification.ClassicCls import ANN, SVM, PLS_DA, RF
from Classification.CNN import CNN
from Classification.SAE import SAE

def  QualitativeAnalysis(model, X_train, X_test, y_train, y_test):

    if model == "PLS_DA":
        acc = PLS_DA(X_train, X_test, y_train, y_test)
    elif model == "ANN":
        acc = ANN(X_train, X_test, y_train, y_test)
    elif model == "SVM":
        acc = SVM(X_train, X_test, y_train, y_test)
    elif model == "RF":
        acc = RF(X_train, X_test, y_train, y_test)
    elif model == "CNN":
        acc = CNN(X_train, X_test, y_train, y_test, 16, 160, 4)
    elif model == "SAE":
        acc = SAE(X_train, X_test, y_train, y_test)
    else:
        print("no this model of QuantitativeAnalysis")

    return acc