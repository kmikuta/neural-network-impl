import numpy as np
from neural_network.layer import ActivationLayer, LinearLayer
from neural_network.network import Network
from neural_network.activations import tanh, tanh_derivative
from neural_network.loss import mse, mse_derivative


def test_training_results():
    # training set
    x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

    # network
    net = Network()
    net.add(LinearLayer(2, 3))
    net.add(ActivationLayer(tanh, tanh_derivative))
    net.add(LinearLayer(3, 1))
    net.add(ActivationLayer(tanh, tanh_derivative))

    # training
    net.use(mse, mse_derivative)
    net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

    # results
    out = net.predict(x_train)
    print(out)
