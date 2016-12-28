import numpy as np
import layer as ann
import network
import matplotlib.pyplot as plt

##### network.py #####

def trainave()
    raise("not implemented")






##### layer.py #####

def trainsum(layer):
    x = np.random.randint(low = 1, high = 10, size = (1000, 2))
    y = np.sum(x, axis=1)

    error = []

    for i in range(0, 1000, 10):
        for j in range(0, 10):
            layer.forward(np.reshape(x[i+j], (1, 2)))
        error.append(layer.mse(y[i:i+10]))
        layer.backward(layer.firstdelta(y[i:i+10]))

    plt.plot(range(len(error)), error)
    plt.show()
    return layer
