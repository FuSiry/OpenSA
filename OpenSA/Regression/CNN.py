"""
    Create on 2021-1-21
    Author：Pengyou Fu
    Describe：this for train NIRS with use 1-D Resnet model to transfer
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision
import torch.nn.functional as F
from sklearn.preprocessing import scale,MinMaxScaler,Normalizer,StandardScaler
import torch.optim as optim
from Regression.CnnModel import ConvNet, DeepSpectra, AlexNet
import os
from datetime import datetime
from Evaluate.RgsEvaluate import ModelRgsevaluate, ModelRgsevaluatePro
import matplotlib.pyplot  as plt


LR = 0.001
BATCH_SIZE = 16
TBATCH_SIZE = 240


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#自定义加载数据集
class MyDataset(Dataset):
    def __init__(self,specs,labels):
        self.specs = specs
        self.labels = labels

    def __getitem__(self, index):
        spec,target = self.specs[index],self.labels[index]
        return spec,target

    def __len__(self):
        return len(self.specs)



###定义是否需要标准化
def ZspPocessnew(X_train, X_test, y_train, y_test, need=True): #True:需要标准化，Flase：不需要标准化

    global standscale
    global yscaler

    if (need == True):
        standscale = StandardScaler()
        X_train_Nom = standscale.fit_transform(X_train)
        X_test_Nom = standscale.transform(X_test)

        #yscaler = StandardScaler()
        yscaler = MinMaxScaler()
        y_train = yscaler.fit_transform(y_train.reshape(-1, 1))
        y_test = yscaler.transform(y_test.reshape(-1, 1))

        X_train_Nom = X_train_Nom[:, np.newaxis, :]
        X_test_Nom = X_test_Nom[:, np.newaxis, :]

        ##使用loader加载测试数据
        data_train = MyDataset(X_train_Nom, y_train)
        data_test = MyDataset(X_test_Nom, y_test)
        return data_train, data_test
    elif((need == False)):
        yscaler = StandardScaler()
        # yscaler = MinMaxScaler()

        X_train_new = X_train[:, np.newaxis, :]  #
        X_test_new = X_test[:, np.newaxis, :]

        y_train = yscaler.fit_transform(y_train)
        y_test = yscaler.transform(y_test)

        data_train = MyDataset(X_train_new, y_train)
        ##使用loader加载测试数据
        data_test = MyDataset(X_test_new, y_test)

        return data_train, data_test




def CNNTrain(NetType, X_train, X_test, y_train, y_test, EPOCH):


    data_train, data_test = ZspPocessnew(X_train, X_test, y_train, y_test, need=True)
    # data_train, data_test = ZPocess(X_train, X_test, y_train, y_test)

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=TBATCH_SIZE, shuffle=True)

    if NetType == 'ConNet':
        model = ConvNet().to(device)
    elif NetType == 'AlexNet':
        model = AlexNet().to(device)
    elif NetType == 'DeepSpectra':
        model = DeepSpectra().to(device)



    criterion = nn.MSELoss().to(device)  # 损失函数为焦损函数，多用于类别不平衡的多分类问题
    optimizer = optim.Adam(model.parameters(), lr=LR)#,  weight_decay=0.001)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    # # initialize the early_stopping object
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=1, eps=1e-06,
                                                           patience=20)
    print("Start Training!")  # 定义遍历数据集的次数
    # to track the training loss as the model trains
    for epoch in range(EPOCH):
        train_losses = []
        model.train()  # 不训练
        train_rmse = []
        train_r2 = []
        train_mae = []
        for i, data in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
            inputs, labels = data  # 输入和标签都等于data
            inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
            labels = Variable(labels).type(torch.FloatTensor).to(device)  # batch y
            output = model(inputs)  # cnn output
            loss = criterion(output, labels)  # MSE
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            pred = output.detach().cpu().numpy()
            y_true = labels.detach().cpu().numpy()
            train_losses.append(loss.item())
            rmse, R2, mae = ModelRgsevaluatePro(pred, y_true, yscaler)
            # plotpred(pred, y_true, yscaler))
            train_rmse.append(rmse)
            train_r2.append(R2)
            train_mae.append(mae)
        avg_train_loss = np.mean(train_losses)
        avgrmse = np.mean(train_rmse)
        avgr2 = np.mean(train_r2)
        avgmae = np.mean(train_mae)
        print('Epoch:{}, TRAIN:rmse:{}, R2:{}, mae:{}'.format((epoch+1), (avgrmse), (avgr2), (avgmae)))
        print('lr:{}, avg_train_loss:{}'.format((optimizer.param_groups[0]['lr']), avg_train_loss))

        with torch.no_grad():  # 无梯度
            model.eval()  # 不训练
            test_rmse = []
            test_r2 = []
            test_mae = []
            for i, data in enumerate(test_loader):
                inputs, labels = data  # 输入和标签都等于data
                inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
                labels = Variable(labels).type(torch.FloatTensor).to(device)  # batch y
                outputs = model(inputs)  # 输出等于进入网络后的输入
                pred = outputs.detach().cpu().numpy()
                y_true = labels.detach().cpu().numpy()
                rmse, R2, mae = ModelRgsevaluatePro(pred, y_true, yscaler)
                test_rmse.append(rmse)
                test_r2.append(R2)
                test_mae.append(mae)
            avgrmse = np.mean(test_rmse)
            avgr2   = np.mean(test_r2)
            avgmae = np.mean(test_mae)
            print('EPOCH：{}, TEST: rmse:{}, R2:{}, mae:{}'.format((epoch+1), (avgrmse), (avgr2), (avgmae)))
            # 将每次测试结果实时写入acc.txt文件中
            scheduler.step(rmse)

    return avgrmse, avgr2, avgmae











#
# def CNN(X_train, X_test, y_train, y_test, BATCH_SIZE, n_epochs):
#
#     CNNTrain(X_train, X_test, y_train, y_test,BATCH_SIZE,n_epochs)
