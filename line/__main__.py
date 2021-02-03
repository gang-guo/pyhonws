import numpy as np
import sys

class Liner:

    def __init__(self, wlen):
        self.w = np.zeros([wlen, 1])
        self.b = 0

    def forward(self, x, y):
        t = np.dot(x, self.w) + self.b
        error = t - y
        return error

    def backend(self, x, error):
        dw = 2 * error * x
        db = 2 * error
        return np.mean(dw, axis=0),np.mean(db)

    def train(self, x, y, epoch, rate):
        for i in range(epoch):
            error = self.forward(x, y)
            dw,db = self.backend(x, error)
            dw = dw.reshape([x.shape[1],1])
            self.w = self.w - rate * dw
            self.b = self.b - rate * db
            loss = np.sum(error * error)
            print(loss)
        return self.w


ln = Liner(3)
X = np.random.randn(5000, 3)
Y = np.matmul(X, np.array([2, 1, 3]).T) + 3
Y = Y.reshape([Y.shape[0], 1]) + np.random.randn(Y.shape[0], 1)
w = ln.train(X, Y, 5000, 0.01)
print(w)