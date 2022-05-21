
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
# import hpelm

"""
    -*- coding: utf-8 -*-
    @Time   :2022/04/12 17:10
    @Author : Pengyou FU
    @blogs  : https://blog.csdn.net/Echo_Code?spm=1000.2115.3001.5343
    @github : https://github.com/FuSiry/OpenSA
    @WeChat : Fu_siry
    @License：Apache-2.0 license

"""

from sklearn.svm import SVR
from Evaluate.RgsEvaluate import ModelRgsevaluate

def Pls( X_train, X_test, y_train, y_test):


    model = PLSRegression(n_components=8)
    # fit the model
    model.fit(X_train, y_train)

    # predict the values
    y_pred = model.predict(X_test)

    Rmse, R2, Mae = ModelRgsevaluate(y_pred, y_test)

    return Rmse, R2, Mae


def Svregression(X_train, X_test, y_train, y_test):


    model = SVR(C=2, gamma=1e-07, kernel='linear')
    model.fit(X_train, y_train)

    # predict the values
    y_pred = model.predict(X_test)
    Rmse, R2, Mae = ModelRgsevaluate(y_pred, y_test)

    return Rmse, R2, Mae

def Anngression(X_train, X_test, y_train, y_test):


    model = MLPRegressor(
        hidden_layer_sizes=(20, 20), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
        learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=400, shuffle=True,
        random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.fit(X_train, y_train)

    # predict the values
    y_pred = model.predict(X_test)
    Rmse, R2, Mae = ModelRgsevaluate(y_pred, y_test)

    return Rmse, R2, Mae

def ELM(X_train, X_test, y_train, y_test):

    model = hpelm.ELM(X_train.shape[1], 1)
    model.add_neurons(20, 'sigm')


    model.train(X_train, y_train, 'r')
    y_pred = model.predict(X_test)


    Rmse, R2, Mae = ModelRgsevaluate(y_pred, y_test)

    return Rmse, R2, Mae