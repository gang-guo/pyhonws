import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

data = pd.read_csv('E:/资料/学习/deeplearning/lihongyi/数据/hw1/train.csv', encoding='big5')

data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()
# 根据最近3个小时，预测第4个小时
hour_num = 12

month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24: (day + 1) * 24] = raw_data[18 * (20 * month + day): 18 * (20 * month + day + 1), :]
    month_data[month] = sample

x = np.empty([12 * (480 - hour_num), 18 * hour_num], dtype=float)
y = np.empty([12 * (480 - hour_num), 1], dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 23 - hour_num:
                continue
            x[month * (480 - hour_num) + day * 24 + hour, :] = month_data[month][:, day * 24 + hour: day * 24 + hour + hour_num].reshape(1, -1)
            y[month * (480 - hour_num) + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + hour_num]


# 删除RAINFALL属性
delarr = np.array(np.arange(10 * hour_num, 11*hour_num, 1))
x = np.delete(x, delarr, axis=1)

# 归一化
mean_x = np.mean(x, axis=0)
std_x = np.std(x, axis=0)
for i in range(len(x)):
    for j in range(len(x[0])):
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

# 分为训练集和验证集
x_train_set = x[: math.floor(len(x) * 0.9), :]
y_train_set = y[: math.floor(len(y) * 0.9), :]
x_validation = x[math.floor(len(x) * 0.9):, :]
y_validation = y[math.floor(len(y) * 0.9):, :]


# 模型训练
dim = 17 * hour_num + 1
w = np.zeros([dim, 1])
print(x_train_set.shape)
x = np.concatenate((np.ones([x_train_set.shape[0], 1]), x_train_set), axis=1).astype(float)
x_validation = np.concatenate((np.ones([x_validation.shape[0], 1]), x_validation), axis=1).astype(float)
y = y_train_set

train_loss = []
learning_rate = 1
iter_time = 1000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
for t in range(iter_time):
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)  # dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/x.shape[0])
    if t % 100 == 0:
        print(str(t) + ":" + str(loss))
        train_loss.append(loss)
plt.plot(train_loss)
plt.title('Loss')
plt.legend(['train'])
plt.show()

print('dev_loss:', np.sqrt(np.sum(np.power(np.dot(x_validation, w) - y_validation, 2))/x_validation.shape[0]))
