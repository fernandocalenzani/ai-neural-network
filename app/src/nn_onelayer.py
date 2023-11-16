import numpy as np

xi = np.array([[0, 0],
               [1, 0],
               [0, 1],
               [1, 1]]
              )
yi = np.array([[0], [1], [1], [0]])
wi = [
    [-0.424, -0.740, -0.961],
    [0.358, -0.577, -0.469],
]
w_hide = [
    [-0.017, -0.893, 0.148],
]
bias = [1, 0]
epoch = 100


def summation(x, w):
    return np.dot(x, w)


def activation_function(params):
    return 1 / (1 + np.exp(-params))


def calc(x, y):
    return x * y


for j in range(epoch):
    xi_layer = xi
    sum_synapse = summation(xi_layer, wi)
    hide_layer = activation_function(sum_synapse)

print('END')
