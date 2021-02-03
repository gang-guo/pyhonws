import common as c
import torch
import torch.nn as nn
import torchkeras as tk
import torch.utils.data as tud

epochs=10000
LOAD = False
TRAIN = True
SAVE = False

torch.manual_seed(0)

x,y = c.gen_data()

class NetWork3(tk.Model):
    def __init__(self):
        super(NetWork3, self).__init__()
        self.line = nn.Linear(1, 20)
        self.active = nn.ReLU()
        self.line2 = nn.Linear(20, 20)
        self.active2 = nn.ReLU()
        self.line3 = nn.Linear(20, 20)
        self.active3 = nn.ReLU()
        self.out = nn.Linear(20,1)

        nn.init.normal_(self.line.weight)
        nn.init.normal_(self.line2.weight)
        nn.init.normal_(self.line3.weight)
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
        re = self.active(re)
        re = self.line2(re)
        re = self.active2(re)
        re = self.line3(re)
        re = self.active3(re)
        re = self.out(re)
        return re


dataset = c.Data(x, y)
dataLoader = tud.DataLoader(dataset, batch_size=20, shuffle=True)

model = NetWork3()
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
    torch.save(model.state_dict(), 'model3.pt')

def load():
    model.load_state_dict(torch.load('model3.pt'))

if TRAIN:
    train(epochs)

if SAVE:
    save()

if LOAD:
    load()

print(model(torch.FloatTensor([[-1],[1],[3],[-3],[-5],[5],[-10],[10],[2.5],[-2.5]])).reshape(-1))
print(model(torch.FloatTensor([[-100],[1000],[500]])).reshape(-1))