import sys
import pandas as pd
import numpy as np
import torch
import torchkeras as tk
import torch.nn as nn

torch.random.manual_seed(0)

#读取数据
data = pd.read_csv('E:/资料/学习/deeplearning/lihongyi/数据/hw1/train.csv', encoding='big5',)

data = data.replace('NR',0)
#去掉数据的前3列
data = data.iloc[:, 3:]
# 将值为NR的全部变为0
# data[data == 'NR'] = 0
# 变为numpy
raw_data = data.to_numpy()

np.random.seed(0)
month_data = {}
#12个月
for month in range(12):
    #每月18个特征,480条数据
    sample = np.empty([18, 480])
    #每月20天的数据
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    month_data[month] = sample

#数据总条数为 12*(480 - 9), 特征数为 18个 * 9 天
x = np.empty([10 * 471, 18 * 9], dtype = float)

y = np.empty([10 * 471, 1], dtype = float)
for month in range(1,10):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9] #value
print(x)
print(y)
#最终数据样式是 471*12 条 ，每条是 之前9天的特征平铺的结果


mean_x = np.mean(x, axis = 0) #18 * 9
std_x = np.std(x, axis = 0) #18 * 9
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #18 * 9
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]


eval_x = np.empty([1 * 471, 18 * 9], dtype = float)
eval_y = np.empty([1 * 471, 1], dtype = float)

for day in range(20):
    for hour in range(24):
        if day == 19 and hour > 14:
            continue
        eval_x[day * 24 + hour, :] = month_data[11][:, day * 24 + hour: day * 24 + hour + 9].reshape(1,
                                                                                                                 -1)  # vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
        eval_y[day * 24 + hour, 0] = month_data[11][9, day * 24 + hour + 9]  # value

for i in range(len(eval_x)): #12 * 471
    for j in range(len(eval_x[0])): #18 * 9
        if std_x[j] != 0:
            eval_x[i][j] = (eval_x[i][j] - mean_x[j]) / std_x[j]

eval_x = np.concatenate((np.ones([471, 1]), eval_x), axis = 1).astype(float)
# #训练数据 ： 验证数据 = 8 ： 2
# import math
# x_train_set = x[: math.floor(len(x) * 0.8), :]
# y_train_set = y[: math.floor(len(y) * 0.8), :]
# x_validation = x[math.floor(len(x) * 0.8): , :]
# y_validation = y[math.floor(len(y) * 0.8): , :]
# print(x_train_set)
# print(y_train_set)
# print(x_validation)
# print(y_validation)
# print(len(x_train_set))
# print(len(y_train_set))
# print(len(x_validation))
# print(len(y_validation))

class Model(tk.Model):
    def __init__(self, feature_num):
        super(Model, self).__init__()
        self.liner = nn.Linear(feature_num, 1, bias=False)
        nn.init.normal_(self.liner.weight)

    def forward(self, x):
        return self.liner(x)


dim = 18 * 9 + 1
iter_time = 100000
x = np.concatenate((np.ones([10 * 471, 1]), x), axis = 1).astype(float)
x = torch.from_numpy(x).float()
x.requires_grad_(True)
y = torch.from_numpy(y).float()
y.requires_grad_(True)

m = Model(dim)
lossFn = nn.MSELoss()
optim = torch.optim.SGD(m.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optim,1,gamma=0.9)

lastLoss = 300000
for t in range(iter_time):
    optim.zero_grad()
    y_pred = m(x)
    loss = lossFn(y_pred,y)
    if t%100==0:
        print(str(t) + ":" + str(torch.sqrt(loss).item()))

    if(lastLoss - loss < 0):
        scheduler.step()
    lastLoss = loss
    loss.backward()
    optim.step()

print(torch.sqrt(lossFn(m(torch.from_numpy(eval_x).float()), torch.from_numpy(eval_y).float())).item())


# w = np.random.randn(dim, 1)
# x = np.concatenate((np.ones([11 * 471, 1]), x), axis = 1).astype(float)
# learning_rate = 100
# iter_time = 100000
# adagrad = np.zeros([dim, 1])
# eps = 0.0000000001
# lastLoss = 10000
# for t in range(iter_time):
#     loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/11)#rmse
#     if(t%100==0):
#         print(str(t) + ":" + str(loss))
#         if loss - lastLoss < 0:
#             lastLoss = loss
#             # if lastLoss - loss < 0.00001 :
#             #     learning_rate = learning_rate * 1.1
#         else:
#             learning_rate = learning_rate * 0.9
#         lastLoss = loss
#     gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #dim*1
#     adagrad += gradient ** 2
#     w = w - learning_rate * gradient / np.sqrt(adagrad/(t + 1) + eps)
# np.save('weight.npy', w)
#
# y_pred = np.dot(eval_x, w) - eval_y
#
# # print(y_pred)
#
# print(np.sqrt(np.sum(np.power(y_pred, 2))/471))

# testdata = pd.read_csv('E:/资料/学习/deeplearning/lihongyi/数据/hw1/test.csv', header = None, encoding = 'big5')
# testdata = testdata.replace('NR',0)
# test_data = testdata.iloc[:, 2:]
# # test_data[testdata == 'NR'] = 0
# # test_data[:].replace('NR',0)
# test_data = test_data.to_numpy()
# test_x = np.empty([240, 18*9], dtype = float)
# for i in range(240):
#     test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
# for i in range(len(test_x)):
#     for j in range(len(test_x[0])):
#         if std_x[j] != 0:
#             test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
# test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
#
# test_y = test_x[9:, 9]
# print(test_y)
# test_x = test_x[]
#
# w = np.load('weight.npy')
# ans_y = np.dot(test_x, w)
# print(ans_y.shape)
# print(std_x.shape)
# print(mean_x.shape)

# print("====================================")
# result = ans_y * std_x + mean_x
# print(result)
# print(result.shape)