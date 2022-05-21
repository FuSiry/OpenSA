"""
    -*- coding: utf-8 -*-
    @Time   :2022/04/15 9:36
    @Author : Pengyou FU
    @blogs  : https://blog.csdn.net/Echo_Code?spm=1000.2115.3001.5343
    @github : https://github.com/FuSiry/OpenSA
    @WeChat : Fu_siry
    @License：Apache-2.0 license

"""

import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score,auc,roc_curve,precision_recall_curve,f1_score
import torch.optim as optim
# from EarlyStop import EarlyStopping
from sklearn.preprocessing import scale,MinMaxScaler,Normalizer,StandardScaler
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def conv_k(in_chs, out_chs, k=1, s=1, p=1):
    """ Build size k kernel's convolution layer with padding"""
    return nn.Conv1d(in_chs, out_chs, kernel_size=k, stride=s, padding=p, bias=False)

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
def ZspPocess(X_train, X_test,y_train,y_test,need=True): #True:需要标准化，Flase：不需要标准化
    if (need == True):
        # X_train_Nom = scale(X_train)
        # X_test_Nom = scale(X_test)
        standscale = StandardScaler()
        X_train_Nom = standscale.fit_transform(X_train)
        X_test_Nom = standscale.transform(X_test)

        X_train_Nom = X_train_Nom[:, np.newaxis, :]
        X_test_Nom = X_test_Nom[:, np.newaxis, :]
        data_train = MyDataset(X_train_Nom, y_train)
        ##使用loader加载测试数据
        data_test = MyDataset(X_test_Nom, y_test)
        return data_train, data_test
    else:
        X_train = X_train[:, np.newaxis, :]  # （483， 1， 2074）
        X_test = X_test[:, np.newaxis, :]
        data_train = MyDataset(X_train, y_train)
        ##使用loader加载测试数据
        data_test = MyDataset(X_test, y_test)
        return data_train, data_test

class CNN3Lyaers(nn.Module):
    def __init__(self, nls):
        super(CNN3Lyaers, self).__init__()
        self.CONV1 = nn.Sequential(
            nn.Conv1d(1, 64, 21, 1),
            nn.BatchNorm1d(64),  # 对输出做均一化
            nn.ReLU(),
            nn.MaxPool1d(3, 3)
        )
        self.CONV2 = nn.Sequential(
            nn.Conv1d(64, 64, 19, 1),
            nn.BatchNorm1d(64),  # 对输出做均一化
            nn.ReLU(),
            nn.MaxPool1d(3, 3)
        )
        self.CONV3 = nn.Sequential(
            nn.Conv1d(64, 64, 17, 1),
            nn.BatchNorm1d(64),  # 对输出做均一化
            nn.ReLU(),
            nn.MaxPool1d(3, 3),
        )
        self.fc = nn.Sequential(
            # nn.Linear(4224, nls)
            nn.Linear(384, nls)
        )

    def forward(self, x):
        x = self.CONV1(x)
        x = self.CONV2(x)
        x = self.CONV3(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        out = self.fc(x)
        out = F.softmax(out,dim=1)
        return out

class mlpmodel(nn.Module):
    def __init__(self, inputdim, outputdim):
        super(mlpmodel, self).__init__()
        self.fc1 = nn.Linear(inputdim, inputdim//2)
        self.fc2= nn.Linear(inputdim//2, inputdim // 4)
        self.fc3 = nn.Linear(inputdim//4, outputdim)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # x = F.softmax(x, dim=1)
        return x


def CNNTrain(X_train, X_test,y_train,y_test, BATCH_SIZE, n_epochs, nls):


    data_train, data_test = ZspPocess(X_train, X_test,y_train,y_test,need=True)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=True)

    store_path = ".//model//all//CNN18"

    model = CNN3Lyaers(nls=nls).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=0.0001,weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=1, eps=1e-06,
                                                           patience=10)
    criterion = nn.CrossEntropyLoss().to(device)  # 损失函数为焦损函数，多用于类别不平衡的多分类问题
    #early_stopping = EarlyStopping(patience=30, delta=1e-4, path=store_path, verbose=False)

    for epoch in range(n_epochs):
        train_acc = []
        for i, data in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
            model.train()  # 不训练
            inputs, labels = data  # 输入和标签都等于data
            inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
            labels = Variable(labels).type(torch.LongTensor).to(device)  # batch y
            output = model(inputs)  # cnn output
            trian_loss = criterion(output, labels)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            trian_loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            _, predicted = torch.max(output.data, 1)
            y_predicted = predicted.detach().cpu().numpy()
            y_label = labels.detach().cpu().numpy()
            acc = accuracy_score(y_label, y_predicted)
            train_acc.append(acc)

        with torch.no_grad():  # 无梯度
            test_acc = []
            testloss = []
            for i, data in enumerate(test_loader):
                model.eval()  # 不训练
                inputs, labels = data  # 输入和标签都等于data
                inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
                labels = Variable(labels).type(torch.LongTensor).to(device)  # batch y
                outputs = model(inputs)  # 输出等于进入网络后的输入
                test_loss = criterion(outputs, labels)  # cross entropy loss
                _, predicted = torch.max(outputs.data,1)
                predicted = predicted.cpu().numpy()
                labels = labels.cpu().numpy()
                acc = accuracy_score(labels, predicted)
                test_acc.append(acc)
                testloss.append(test_loss.item())
                avg_loss = np.mean(testloss)

            scheduler.step(avg_loss)
            # early_stopping(avg_loss, model)
            # if early_stopping.early_stop:
            #     print(f'Early stopping! Best validation loss: {early_stopping.get_best_score()}')
            #     break

def CNNtest(X_train, X_test, y_train, y_test, BATCH_SIZE, nls):
    # data_train, data_test = DataLoad(tp, test_ratio, 0, 404)

    data_train, data_test = ZspPocess(X_train, X_test, y_train, y_test, need=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=True)

    store_path = ".//model//all//CNN18"

    model = CNN3Lyaers(nls=nls).to(device)

    model.load_state_dict(torch.load(store_path))
    test_acc = []
    for i, data in enumerate(test_loader):
        model.eval()  # 不训练
        inputs, labels = data  # 输入和标签都等于data
        inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
        labels = Variable(labels).type(torch.LongTensor).to(device)  # batch y
        outputs = model(inputs)  # 输出等于进入网络后的输入
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu().numpy()
        labels = labels.cpu().numpy()
        acc = accuracy_score(labels, predicted)
        test_acc.append(acc)
    return np.mean(test_acc)


def CNN(X_train, X_test, y_train, y_test, BATCH_SIZE, n_epochs,nls):

    CNNTrain(X_train, X_test, y_train, y_test,BATCH_SIZE,n_epochs,nls)
    acc = CNNtest(X_train, X_test, y_train, y_test,BATCH_SIZE,nls)

    return acc