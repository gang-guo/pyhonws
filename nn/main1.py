import torch
import time
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-10, 10, 20), dim=1)
print(x)
y = x.pow(2)
# math.pow(100, 2) :  10000.0
# t.unsqueeze(di)     //在di个维度处升维
# t=T.rand(t.size())    //均匀分布
# t=T.randn(t.size())   //标准正态分布
# torch.randn(*sizes, out=None)
# 返回一个张量，包含了从标准正态分布(mean=0, std=1)中抽取一组随机数，形状             #由可变参数sizes定义。
# t=T.linspace(m,n,step_num)  //[m,n]中以m为首项，n为末项，均分区间为step_num段
x, y = Variable(x), Variable(y)


# Variable就是 变量 的意思。实质上也就是可以变化的量，区别于int变量，它是一种可以变化的变量，这正好就符合了反向传播，参数更新的属性
# 把鸡蛋放到篮子里, requires_grad是参不参与误差反向传播, 要不要计算梯度
# tensor不能反向传播，variable可以反向传播。
class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)


    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.predict(x)
        return x


model = Net(1, 200, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
loss_func = torch.nn.MSELoss()

for t in range(10000):
    prediction = model(x)
    loss = loss_func(prediction, y)
    print(t,loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 5 == 0:  # 每五步打印一次
        plt.cla()   # 清除当前图形中的当前活动轴。其他轴不受影响
        plt.scatter(x.data.numpy(), y.data.numpy())  # 打印原始数据
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)  # 打印预测数据
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data, fontdict={'size': 20, 'color': 'red'})  # 打印误差值
        plt.pause(0.1)  # 每次停顿0.1
# plt.ioff()
# plt.show()
print(model(torch.unsqueeze(torch.linspace(1, 10, 10), dim=1)))

print(model(torch.unsqueeze(torch.tensor([-2.5,2.5,3.5],dtype=torch.float32),dim=1)))
