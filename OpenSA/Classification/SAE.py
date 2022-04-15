"""
    -*- coding: utf-8 -*-
    @Time   :2022/04/15 9:36
    @Author : Pengyou FU
    @blogs  : https://blog.csdn.net/Echo_Code?spm=1000.2115.3001.5343
    @github : https://github.com/FuSiry/OpenSA
    @WeChat : Fu_siry
    @License：Apache-2.0 license

"""



import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import torch.utils.data as data
import numpy as np
import time
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyDataset(data.Dataset):
    def __init__(self,specs,labels):
        self.specs = specs
        self.labels = labels
    def __getitem__(self, index):
        spec,target = self.specs[index],self.labels[index]
        return spec,target
    def __len__(self):
        return len(self.specs)


class AutoEncoder(nn.Module):

    def __init__(self, inputDim, hiddenDim):
        super().__init__()
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.encoder = nn.Linear(inputDim, hiddenDim, bias=True)
        self.decoder = nn.Linear(hiddenDim, inputDim, bias=True)
        self.act = F.relu

    def forward(self, x, rep=False):

        hidden = self.encoder(x)
        hidden = self.act(hidden)
        if rep == False:
            out = self.decoder(hidden)
            #out = self.act(out)
            return out
        else:
            return hidden


class SAE(nn.Module):

    def __init__(self, encoderList):

        super().__init__()

        self.encoderList = encoderList
        self.en1 = encoderList[0]
        self.en2 = encoderList[1]
        # self.en3 = encoderList[2]

        self.fc = nn.Linear(128, 4, bias=True)

    def forward(self, x):

        out = x
        out = self.en1(out, rep=True)
        out = self.en2(out, rep=True)
        #out = self.en3(out, rep=True)
        out = self.fc(out)
        # out = F.log_softmax(out)

        return out


class SAE_net(object):
    def __init__(self, AE_epoch = 200, SAE_epoch = 200,
                 input_dim = 404, hidden1_dim = 512,
                 hidden2_dim = 128, output_dim = 4,
                 batch_size = 128):
        self.AE_epoch = AE_epoch
        self.SAE_epoch = SAE_epoch
        self.input_dim = input_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.train_loader = None

        encoder1 = AutoEncoder(self.input_dim, self.hidden1_dim)
        encoder2 = AutoEncoder(self.hidden1_dim, self.hidden2_dim)
        self.encoder_list = [encoder1, encoder2]


    def trainAE(self, x_train, y_train, encoderList, trainLayer, batchSize, epoch, useCuda=False):
        if useCuda:
            for i in range(len(encoderList)):
                encoderList[i].to(device)

        optimizer = optim.Adam(encoderList[trainLayer].parameters())
        ceriation = nn.MSELoss()

        data_train = MyDataset(x_train, y_train)
        self.train_loader = torch.utils.data.DataLoader(data_train, batch_size=batchSize, shuffle=True)

        for i in range(epoch):
            sum_loss = 0
            if trainLayer != 0:  # 单独处理第0层，因为第一个编码器之前没有前驱的编码器了
                for i in range(trainLayer):  # 冻结要训练前面的所有参数
                    for param in encoderList[i].parameters():
                        param.requires_grad = False

            for batch_idx, (x, target) in enumerate(self.train_loader):
                optimizer.zero_grad()
                if useCuda:
                    x, target = x.to(device), target.to(device)
                x, target = Variable(x).type(torch.FloatTensor), Variable(target).type(torch.LongTensor)
                # x = x.view(-1, 404)
                x = x.view(x.size(0), -1)
                # 产生需要训练层的输入数据
                # inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
                # labels = Variable(labels).type(torch.LongTensor).to(device)  # batch y
                out = x
                if trainLayer != 0:
                    for i in range(trainLayer):
                        out = encoderList[i](out, rep=True)

                # 训练指定的自编码器
                pred = encoderList[trainLayer](out, rep=False).cpu()

                loss = ceriation(pred, out)
                sum_loss += loss.item()
                loss.backward()
                optimizer.step()

    def trainClassifier(self, model, epoch, useCuda=False):
        if useCuda:
            model = model.to(device)

        # 解锁参数
        for param in model.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(model.parameters())
        ceriation = nn.CrossEntropyLoss()

        for i in range(epoch):
            # trainning
            sum_loss = 0
            for batch_idx, (x, target) in enumerate(self.train_loader):
                optimizer.zero_grad()
                if useCuda:
                    x, target = x.to(device), target.to(device)
                x, target = Variable(x).type(torch.FloatTensor), Variable(target).type(torch.LongTensor)
                x = x.view(-1, 404)

                out = model(x)

                loss = ceriation(out, target)
                sum_loss += loss.item()
                loss.backward()
                optimizer.step()
        self.model = model

    def fit(self, x_train = None, y_train = None):
        x_train = x_train[:, np.newaxis, :]
        x_train = torch.from_numpy(x_train)
        x_train = x_train.float()

        # pre-train
        for i in range(2):
            self.trainAE(x_train=x_train, y_train=y_train,
                        encoderList = self.encoder_list, trainLayer=i, batchSize=self.batch_size,
                         epoch = self.AE_epoch)
        model = SAE(encoderList=self.encoder_list)
        self.trainClassifier(model=model, epoch=self.SAE_epoch)

    def predict_proba(self, x_test):
        x_test = torch.from_numpy(x_test)
        x_test = x_test.float()
        x_test = x_test[:, np.newaxis, :]
        x_test = Variable(x_test)
        x_test = x_test.view(-1, 404)

        out = self.model(x_test)
        outdata = out.data
        self.y_proba = outdata
        y_proba = outdata.numpy()
        return y_proba

    def predict(self, x_test):
        _, y_out = torch.max(self.y_proba, 1)
        y_pred = []
        for i in y_out:
            y_pred.append(i)
        return y_pred

def SAE(X_train, y_train, X_test, y_test):

    clf = SAE_net()
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    # ACC
    acc = accuracy_score(y_test, y_pred)

    return acc