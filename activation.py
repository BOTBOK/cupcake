import numpy as np


def array_forward(input_val, activation_obj):
    deep, row, col = input_val.shape
    ret = np.zeros((deep, row, col), dtype=float)
    for index_deep in range(deep):
        for i in range(row):
            for j in range(col):
                ret[index_deep][i][j] = activation_obj.forward(input_val[index_deep][i][j])
    return ret


def array_backward(input_val, activation_obj):
    deep, row, col = input_val.shape
    ret = np.zeros((deep, row, col), dtype=float)
    for index_deep in range(deep):
        for i in range(row):
            for j in range(col):
                ret[index_deep][i][j] = activation_obj.backward(input_val[index_deep][i][j])
    return ret


class ReluActivation(object):
    @staticmethod
    def forward(in_val):
        return max(0, in_val)

    @staticmethod
    def backward(in_val):
        return 1 if in_val > 0 else 0


class SigmoidActivation(object):
    @staticmethod
    def forward(in_val):
        return 1 / (1 + np.exp(-in_val))

    @staticmethod
    def backward(out_val):
        return out_val * (1 - out_val)
