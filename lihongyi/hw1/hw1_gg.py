import numpy as np
import pandas as pd

datapath='E:/资料/学习/deeplearning/lihongyi/数据/hw1/train.csv'
df = pd.read_csv(datapath, encoding='big5')
df = df.iloc[:,3:]
df[df=='NR'] = 0
x_data = df.to_numpy()
print(x_data.shape)

month={}
for m in range(12):
    m_d = np.empty([18, 20*24])
    for d in range(20):
        m_d[:,d*24:(d+1)*24]=x_data[(m*20+d)*18:(m*20+d+1)*18,:]
    month[m] = m_d

x_train=np.empty([12*(20*24 -9), 9*18])
y_train=np.empty([12 * 471,1])
for m in range(12):
    for n in range(471):
        x_train[m*471+n] = month[m][:,n:n+9].reshape(1,-1)
        y_train[m * 471 + n] = month[m][9, n+9]

print(x_train.shape)
print(y_train.shape)

dim = 9*18+1
w = np.random.randn([dim,1])
learning_rate=1
epochs=1000
for epoch in range(epochs):
    loss = np.sum(np.power(np.dot(x_train, w)-y,2)) / x_train.shap[0]
    if epoch % 100 ==0