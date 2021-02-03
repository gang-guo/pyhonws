import common as c
import torch
import torch.nn as nn
import torchkeras as tk
import torch.utils.data as tud

x = torch.linspace(-10,10,21).float()
x = x.view(-1, 1)
y = x*1
x.requires_grad_(True)
y.requires_grad_(True)

torch.manual_seed(0)

print(x)
print(y)

# active = nn.Sigmoid()
active = nn.ReLU()

class NetWork(tk.Model):
    def __init__(self):
        super(NetWork, self).__init__()
        self.line = nn.Linear(1, 20)
        # self.line2 = nn.Linear(20, 20)
        # self.line3 = nn.Linear(20, 20)
        self.out = nn.Linear(20,1)

        nn.init.normal_(self.line.weight)
        # nn.init.normal_(self.line2.weight)
        # nn.init.normal_(self.line3.weight)
        # for i in range(2000):
        #     self.line.weight[i] = x[i]
        # with torch.no_grad():
        #     self.line.weight.set_(torch.arange(-1000, 1000).float().view(-1,1))
        #     self.line.bias.zero_()
        # print(x.shape)
        # print(self.line.weight.shape)
        nn.init.normal_(self.out.weight)

    def forward(self, x):
        re = self.line(x)
        # re = active(re)
        # re = self.line2(re)
        # re = active(re)
        # re = self.line3(re)
        re = active(re)
        re = self.out(re)
        return re

model = NetWork()
loss_fn = nn.MSELoss()
optimer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10000):
    model.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print(epoch, loss.item())
    loss.backward(retain_graph=True)
    optimer.step()

print(model(torch.FloatTensor([[5.], [10.], [20.], [100.]])))
