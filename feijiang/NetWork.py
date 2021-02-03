class NetWork(object):
    def __init__(self, numofweight):
        np.random.seed(0)
        self.w=np.random.randn(numofweight,1)
        self.b=0
    def forword(self, x):
        return np.dot(x, self.w) + self.b
    def loss(self, t, y):
        error = t - y
        cost = error ** 2
        return np.mean(cost)
    def gradient(self, x, y):
        t = self.forword(x)
        gradient_w = (t - y) * x
        gradient_w = np.mean(gradient_w, axis = 0)
        gradient_w = gradient_w.reshape([x.shape[1],y.shape[1]])
        gradient_b = np.mean(t - y)
        return gradient_w, gradient_b
    def train(self, x, y, iterations=100, eta=0.1):
        ws = []
        losses = []
        for i in range(iterations):
            ws.append(self.w)
            t = self.forword(x)
            loss = self.loss(t,y)
            losses.append(loss)
            gradient_w, gradient_b = self.gradient(x,y)
            self.w = self.w - eta * gradient_w
            self.b = self.b - eta * gradient_b
            if(i % 50 == 0):
                print('第{}次：w=[{}], loss={}'.format(i,ws[len(ws) - 1], loss))
        return ws, losses