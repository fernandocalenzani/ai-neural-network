import numpy as np

x_in = np.array([[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]]
                )
w_0 = 2 * np.random.random((2, 3)) - 1

w_1 = 2 * np.random.random((3, 1)) - 1

y = np.array([[0], [1], [1], [0]])
bias = [1, 1]
epoch = 10000
momentum = 1
learn_rate = 0.6
erro_min = 0.02
average_erro = 100


def summation(x, w):
    return np.dot(x, w)


def act_function(u):
    return 1 / (1 + np.exp(-u))


def dydx_act_function(f):
    return f * (1 - f)


def calc(x, y):
    return x * y


while (average_erro > erro_min):

    x_layer_0 = x_in
    sum_l0 = summation(x_layer_0, w_0)
    f_act_l0 = act_function(sum_l0)

    sum_l1 = summation(f_act_l0, w_1)
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
