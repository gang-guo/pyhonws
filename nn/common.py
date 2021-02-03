import torch
import torch.utils.data as tud
def gen_data():
    x = torch.arange(-10, 10).float()
    x = x.view(-1, 1)
    x.requires_grad_(True)
    y = x**2
    y.requires_grad_(True)
    return x,y

class Data(tud.Dataset):
    def __init__(self, x, y):
        super(Data, self).__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        return self.x[item], self.y[item]

