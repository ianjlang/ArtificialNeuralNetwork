import numpy as np
from layer import layer
import matplotlib.pyplot as plt

class Network(object):

    def __init__(self, in_size, out_size, hidden_layers = []):
        hidden_layers.insert(0, in_size)
        hidden_layers.append(out_size)
        self.i = in_size
        self.o = out_size
        if self.i < 1 or self.o < 1:
            raise("I/O size error")
        self.layers = []
        for i in range(1, len(hidden_layers)):
            if hidden_layers[i] < 1:
                raise("Layer size error")
            self.layers.append(layer(hidden_layers[i-1], hidden_layers[i]))
        self.train_error = []
        self.test_error = []
        self.trained = False

    def train(self, data, out):
        for i in range(0, len(data)):
            if len(data[i].shape) != 2:
                val = np.reshape(data[i], (1, len(data[i])))
            else:
                val = data[i]
            for layer in self.layers:
                val = layer.forward(val)
            self.train_error.append(self.layers[-1].mse(out[i]))
            delta = self.layers[-1].firstdelta(out[i])
            for layer in reversed(self.layers):
                delta = layer.backward(delta)
        self.trained = True

    def test(self, data, out):
        if not self.trained:
            raise("Not yet trained")
        for i in range(0, len(data)):
            if len(data[i].shape) != 2:
                val = np.reshape(data[i], (1, len(data[i])))
            else:
                val = data[i]
            for layer in self.layers:
                val = layer.forward(val)
            self.test_error.append(self.layers[-1].mse(out[i]))
        AverageError = np.mean(self.test_error)
        return AverageError

    def predict(self, data):
        if not self.trained:
            raise("Not yet trained")
        ret = []
        for i in range(0, len(data)):
            if len(data[i].shape) != 2:
                val = np.reshape(data[i], (1, len(data[i])))
            else:
                val = data[i]
            for layer in self.layers:
                val = layer.forward(val)
            ret.append(val)
        return ret

    def graph(wtm = "train"):
        if wtm.lower() == "train":
            if not self.trained:
                raise("Not yet trained")
            plt.plot(range(len(self.train_error)), self.train_error)
            plt.title("Train Error")
            plt.xlabel("Iteration")
            plt.ylabel("Error")
            plt.show()
