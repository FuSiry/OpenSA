"""
    -*- coding: utf-8 -*-
    @Time   :2022/04/12 17:10
    @Author : Pengyou FU
    @blogs  : https://blog.csdn.net/Echo_Code?spm=1000.2115.3001.5343
    @github : https://github.com/FuSiry/OpenSA
    @WeChat : Fu_siry
    @License：Apache-2.0 license

"""


from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import sklearn.svm as svm
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
import pandas  as pd

def ANN(X_train, X_test, y_train, y_test, StandScaler=None):

    if StandScaler:
        scaler = StandardScaler() # 标准化转换
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # 神经网络输入为2，第一隐藏层神经元个数为5，第二隐藏层神经元个数为2，输出结果为2分类。
    # solver='lbfgs',  MLP的求解方法：L-BFGS 在小数据上表现较好，Adam 较为鲁棒，
    # SGD在参数调整较优时会有最佳表现（分类效果与迭代次数）,SGD标识随机梯度下降。
    #clf =  MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(8,8), random_state=1, activation='relu')
    clf =  MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9,
                  beta_2=0.999, early_stopping=False, epsilon=1e-08,
                  hidden_layer_sizes=(10, 8), learning_rate='constant',
                  learning_rate_init=0.001, max_iter=200, momentum=0.9,
                  nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                  solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
                  warm_start=False)

    clf.fit(X_train,y_train.ravel())
    predict_results=clf.predict(X_test)
    acc = accuracy_score(predict_results, y_test.ravel())

    return acc

def SVM(X_train, X_test, y_train, y_test):

    clf = svm.SVC(C=1, gamma=1e-3)
    clf.fit(X_train, y_train)

    predict_results = clf.predict(X_test)
    acc = accuracy_score(predict_results, y_test.ravel())

    return acc

def PLS_DA(X_train, X_test, y_train, y_test):

    y_train = pd.get_dummies(y_train)
    # 建模
    model = PLSRegression(n_components=228)
    model.fit(X_train, y_train)
    # 预测
    y_pred = model.predict(X_test)
    # 将预测结果（类别矩阵）转换为数值标签
    y_pred = np.array([np.argmax(i) for i in y_pred])
    acc = accuracy_score(y_test, y_pred)

    return acc

def RF(X_train, X_test, y_train, y_test):

    RF = RandomForestClassifier(n_estimators=15,max_depth=3,min_samples_split=3,min_samples_leaf=3)
    RF.fit(X_train, y_train)
    y_pred = RF.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return acc
