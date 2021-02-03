import numpy as np

def sigmod(x):
    if isinstance(x, np.ndarray):
        result = []
        for i in range(x.shape[0]):
            result.append(sigmod(x[i]))
        return np.array(result)
    else:
        if x >= 0:
            return 1 / (1 + np.exp(-x))
        else:
            return np.exp(x) / (1 + np.exp(x))

class Logic:
    def __init__(self, wlen):
        self.w = np.zeros([wlen, 1])
        self.b = 0

    def forward(self, x, y):
        t = np.dot(x, self.w) + self.b
        z = sigmod(t)
        error = z - y
        return z,error

    def backend(self, x, error):
        dw = error * x
        db = error
        return np.mean(dw, axis=0), np.mean(db)

    def train(self, x, y, epoch, rate):
        for i in range(epoch):
            z, error = self.forward(x, y)
            dw, db = self.backend(x, error)
            dw = dw.reshape([x.shape[1], 1])
            self.w = self.w - rate * dw
            self.b = self.b - rate * db
            loss = np.sum(-(np.dot(y.T , np.log(z)) + np.dot((1 - y.T) , np.log(1 - z))))
            print(loss)
        return self.w

ln = Logic(3)
X = np.random.randn(5000, 3)
Y = np.matmul(X, np.array([2, 1, 3]).T)
Y = Y.reshape([Y.shape[0], 1]) + np.random.rand(Y.shape[0], 1)
Y = sigmod(Y)

# Y = np.random.rand(X.shape[0], 1)
for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
        if Y[i][j] >= 0.5 :
            Y[i][j] = 1
        else:
            Y[i][j] = 0

w = ln.train(X, Y, 500, 0.1)
print(Y.shape)
print(Y.sum())
print(w)
print(ln.b)