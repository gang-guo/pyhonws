import common as c
import torch
import torch.nn as nn
import torchkeras as tk
import torch.utils.data as tud

epochs=10000
LOAD = False
TRAIN = True
SAVE = True


x,y = c.gen_data()

class NetWork(tk.Model):
    def __init__(self):
        super(NetWork, self).__init__()
        self.line = nn.Linear(1, 1000)
        self.active = nn.ReLU()
        self.line2 = nn.Linear(1000,1)

        nn.init.normal_(self.line.weight)
        # for i in range(2000):
        #     self.line.weight[i] = x[i]
        # with torch.no_grad():
        #     self.line.weight.set_(torch.arange(-1000, 1000).float().view(-1,1))
        #     self.line.bias.zero_()
        # print(x.shape)
        # print(self.line.weight.shape)
        nn.init.normal_(self.line2.weight)

    def forward(self, x):
        re = self.line(x)
        re = self.active(re)
        re = self.line2(re)
        return re


dataset = c.Data(x, y)
dataLoader = tud.DataLoader(dataset, batch_size=20, shuffle=True)

model = NetWork()
loss_fn = nn.MSELoss()
optimer = torch.optim.Adam(model.parameters(), lr=1e-4)
# schuler = torch.optim.lr_scheduler.ExponentialLR(optimer, 0.5)
loss_arr = []

def train(epochs):
    time = 0
    last_loss = -1
    for epoch in range(1,epochs + 1):
        model.train()
        losses = []
        for i,(x_,y_) in enumerate(dataLoader):
            optimer.zero_grad()
            y_pred = model(x_)
            loss = loss_fn(y_pred, y_)
            losses.append(loss.item())
            loss.backward(retain_graph=True)
            optimer.step()
        total = sum(losses)
        # if total > last_loss:
        #     if last_loss != -1:
        #         time += 1

        # if time > 3:
        #     print("step")
        #     schuler.step()
        #     time = 0
        #     last_loss = total
        if epoch % 10 == 0:
            print(epoch, total)
        loss_arr.append(total)

def save():
    torch.save(model.state_dict(), 'model.pt')

def load():
    model.load_state_dict(torch.load('model.pt'))

if TRAIN:
    train(epochs)

if SAVE:
    save()

if LOAD:
    load()

print(model(torch.FloatTensor([[1],[5],[10],[100],[1000]])))

print(model(torch.FloatTensor([[2.5],[-2.5]])))

# print(model.line.weight)