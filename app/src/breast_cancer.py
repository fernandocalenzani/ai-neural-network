import numpy as np
from sklearn import datasets


def act_function(u):
    return 1 / (1 + np.exp(-u))


def dydx_act_function(f):
    return f * (1 - f)


dataset = datasets.load_breast_cancer()
y = dataset.target

x_in = dataset.data
y_out = np.empty([569, 1], dtype=int)

for i in range(569):
    y_out[i] = y[i]

w_0 = 2 * np.random.random((30, 3)) - 1
w_1 = 2 * np.random.random((3, 1)) - 1
bias = [1, 1]
epoch = 10000
momentum = 1
learn_rate = 0.6
erro_min = 0.2
average_erro = 100


while (average_erro > erro_min):

    x_layer_0 = x_in
    sum_l0 = np.dot(x_layer_0, w_0)
    f_act_l0 = act_function(sum_l0)

    sum_l1 = np.dot(f_act_l0, w_1)
    y_out = act_function(sum_l1)

    erro = y - y_out
    average_erro = np.mean(np.abs(erro))

    dydx_f_act_l1 = dydx_act_function(y_out)
    delta_y_out = erro * dydx_f_act_l1

    w_delta_y_out = np.dot(delta_y_out, np.transpose(w_1))
    delta_l1 = w_delta_y_out * dydx_act_function(f_act_l0)

    w_new_hide = np.dot(np.transpose(f_act_l0), delta_y_out)

    w_1 = w_1 * momentum + w_new_hide * learn_rate

    wi_new = np.dot(np.transpose(x_layer_0), delta_l1)

    w_0 = w_0 * momentum + wi_new * learn_rate

print('END')
