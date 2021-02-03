import torch
import torchkeras
from torch import nn

x = torch.arange(0, 1000).float()
x = x.reshape(-1, 1)
x.requires_grad_(True)
print(x.size())
y = (x ** 2).float()
y.requires_grad_(True)

class NetWork(torchkeras.Model):
    def __init__(self):
        super(NetWork, self).__init__()
        # self.line1 = nn.Linear(1, 1)
        self.line1 = nn.Linear(1, 10)
        self.active1 = nn.LeakyReLU()
        self.line2 = nn.Linear(10, 100)
        self.active2 = nn.LeakyReLU()
        self.line3 = nn.Linear(100, 1)
        # self.active3 = nn.LeakyReLU()
        # self.line4 = nn.Linear(10, 1)
        # self.active4 = nn.LeakyReLU()

        nn.init.normal_(self.line1.weight)
        nn.init.normal_(self.line2.weight)
        nn.init.normal_(self.line3.weight)
        # nn.init.normal_(self.line4.weight)

    def forward(self, x):
        # c = 10
        # for i in range(c):
        #     re = self.line1(x)
        #     if(i != c - 1):
        #         re = self.active1(re)
        re = self.line1(x)
        re = self.active1(re)
        re = self.line2(re)
        re = self.active2(re)
        re = self.line3(re)
        # re = self.active3(re)
        # re = self.line4(re)
        return re


net = NetWork()
lr = 1e-10
optimer = torch.optim.SGD(net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimer, 0.5)
loss_fn = nn.MSELoss()

# print(scheduler.get_lr())
# optimer.zero_grad()
# y_pred = net(x)
# loss = loss_fn(y_pred, y)
# # print(i, loss.item())
# loss.backward(retain_graph=True)
# # print(net.line1.weight)
# optimer.step()
# scheduler.step()
#
# optimer.zero_grad()
# y_pred = net(x)
# loss = loss_fn(y_pred, y)
# # print(i, loss.item())
# loss.backward(retain_graph=True)
# # print(net.line1.weight)
# optimer.step()
# scheduler.step()
#
# optimer.zero_grad()
# y_pred = net(x)
# loss = loss_fn(y_pred, y)
# # print(i, loss.item())
# loss.backward(retain_graph=True)
# # print(net.line1.weight)
# optimer.step()
# scheduler.step()
# print(scheduler.get_lr())
last_loss = 0
for i in range(1000000):
    optimer.zero_grad()
    y_pred = net(x)
    loss = loss_fn(y_pred, y)
    print(i, loss.item())
    loss.backward(retain_graph=True)
    # print(net.line1.weight)
    optimer.step()
    if(last_loss != 0 and loss.item() > last_loss):
        scheduler.step()

    last_loss = loss.item()
    if(last_loss < 0.000001):
        break


print(net(torch.FloatTensor([1000.])).item())