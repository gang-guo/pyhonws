import sys
import pandas as pd
import numpy as np

#写在前面，主要改动为：
# 丢弃第一个月的数据，因为测试下来第一个月的数据是坏掉的
# 改动的adagrad的公式，改动为学习速率随时间变更不再使用公式，而是自己控制，当发现loss变大的时候学习速率按比例下降
# 增加epoch
# 修改学习速率
# 训练集loss 5.263607216532271
# 测试集loss 5.410681155407726


#读取数据
data = pd.read_csv('E:/资料/学习/deeplearning/lihongyi/数据/hw1/train.csv', encoding='big5',)

data = data.replace('NR',0)
#去掉数据的前3列
data = data.iloc[:, 3:]
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


#训练数据使用前10个月
#数据总条数为 10*(480 - 9), 特征数为 18个 * 9 天
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


#使用最后一个月的数据为测试数据
eval_x = np.empty([1 * 471, 18 * 9], dtype = float)
eval_y = np.empty([1 * 471, 1], dtype = float)

for day in range(20):
    for hour in range(24):
        if day == 19 and hour > 14:
            continue
        eval_x[day * 24 + hour, :] = month_data[11][:, day * 24 + hour: day * 24 + hour + 9].reshape(1, -1)  # vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
        eval_y[day * 24 + hour, 0] = month_data[11][9, day * 24 + hour + 9]  # value

for i in range(len(eval_x)): #12 * 471
    for j in range(len(eval_x[0])): #18 * 9
        if std_x[j] != 0:
            eval_x[i][j] = (eval_x[i][j] - mean_x[j]) / std_x[j]

eval_x = np.concatenate((np.ones([471, 1]), eval_x), axis = 1).astype(float)



dim = 18 * 9 + 1
w = np.random.randn(dim, 1)
x = np.concatenate((np.ones([10 * 471, 1]), x), axis = 1).astype(float)
learning_rate = 10
iter_time = 100000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
lastLoss = 10000
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/10)#rmse
    if(t%100==0):
        print(str(t) + ":" + str(loss))
        if loss - lastLoss > 0:
            learning_rate = learning_rate * 0.95
        lastLoss = loss
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad/(t + 1) + eps)
np.save('weight.npy', w)


y_pred = np.dot(eval_x, w) - eval_y
print(np.sqrt(np.sum(np.power(y_pred, 2))/471))