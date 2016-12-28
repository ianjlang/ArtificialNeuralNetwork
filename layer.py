import numpy as np

class layer(object):

    def __init__(self, in_size, out_size, step = .001, momentum = .9):
        self.w = 2 * np.random.random((in_size, out_size)) - 1
        self.b = np.random.random((1, out_size))
        self.count = 0
        self.inputs = []
        self.outputs = []
        self.step = step
        self.momentum = momentum
        self.prev_bias_update = 0
        self.prev_weight_update = 0

    def forward(self, x):
        self.count += 1
        self.inputs.append(x)
        self.outputs.append(np.dot(x, self.w) + self.b)
        return self.activation(self.outputs[-1])

    def backward(self, delta):
        newdelts = []
        weights = []
        bias = []
        for i in range(self.count):
            dag = np.multiply(delta[i], self.activation_grad(self.outputs[i]))
            newdelts.append(np.dot(self.w, dag))
            weights.append(np.dot(self.inputs[i].T, dag))
            bias.append(dag)
        newdelt = sum(newdelts) / self.count
        weights = sum(weights) / self.count
        bias = sum(bias) / self.count
        self.update_weight(weights)
        self.update_bias(bias)
        self.inputs = []
        self.outputs = []
        self.count = 0
        self.step *= .97
        return newdelt.T

    def predict(self, x):
        return self.activation(np.dot(x, self.w) + self.b)

    def activation(self, val):
        return val

    def activation_grad(self, val):
        return np.ones_like(val)

    def update_weight(self, dx):
        self.prev_weight_update = (self.step * dx) + (self.prev_weight_update * self.momentum)
        self.w -= self.prev_weight_update

    def update_bias(self, dx):
        self.prev_bias_update = (self.step * dx) + (self.prev_bias_update * self.momentum)
        self.b -= self.prev_bias_update

    def mse(self, expected):
        #returns non-array singular value for plotting
        try:
            val = self.outputs - expected
        except:
            try:
                val = [self.outputs[i] - expected[i] for i in range(len(expected))]
            except:
                raise("layers.mse error")
        return (np.sum(np.power(val, 2), axis = 0) / 2)[0][0]

    def firstdelta(self, expected):
        try:
            return self.outputs - expected
        except:
            try:
                return [self.outputs[i] - expected[i] for i in range(len(expected))]
            except: raise("layers.firstdelta error")
