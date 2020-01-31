import numpy as np

from cnn2.activation import ReluActivation
from cnn2.list_struct import CnnListStruct
from cnn2.regulation import L2Regulation, L1Regulation


def check_gradient(layer_param, check_weight_or_bias=1, step=10 ** (-5), reg=10 ** (-1)):
    num_class = 10
    im_height = 32
    im_width = 32
    im_dims = 1

    num_samples = num_class * 20
    data = np.random.randn(num_samples, im_height, im_width, im_dims)
    labels = np.random.randint(num_class, size=num_samples)

    regulation = L1Regulation()
    activation = ReluActivation()

    cnn = CnnListStruct(layer_param, num_class, reg, regulation, activation)

    cnn.feature_maps(data)
    cnn.init_param()

    for layer in range(len(cnn.map_shape)):
        if check_weight_or_bias:
            weight = cnn.weights[layer]
            if weight.size == 0:
                continue
            else:
                row = np.random.randint(weight.shape[0])
                col = np.random.randint(weight.shape[1])
                param = weight[row][col]
        else:
            bias = cnn.biases[layer]
            if bias.size == 0:
                continue
            else:
                row = np.random.randint(bias.shape[1])
                param = bias[0][row]

        cnn.forward(data)
        data_loss = cnn.data_loss(labels)
        reg_loss = cnn.reg_loss()
        cnn.backward(labels)

        if check_weight_or_bias:
            danalytic = cnn.dweights[-1 - layer][row][col]
        else:
            danalytic = cnn.dbiases[-1 - layer][0][row]

        if check_weight_or_bias:
            cnn.weights[layer][row][col] = param - step
        else:
            cnn.biases[layer][0][row] = param - step
        cnn.forward(data)
        data_loss1 = cnn.data_loss(labels)
        reg_loss = cnn.reg_loss()
        loss1 = data_loss1 + reg_loss

        if check_weight_or_bias:
            cnn.weights[layer][row][col] = param + step
        else:
            cnn.biases[layer][0][row] = param + step
        cnn.forward(data)
        data_loss2 = cnn.data_loss(labels)
        reg_loss = cnn.reg_loss()
        loss2 = data_loss2 + reg_loss

        dnumeric = (loss2 - loss1) / (2 * step)

        print(layer, data_loss1, data_loss2)
        error_relative = np.abs(danalytic - dnumeric) / np.maximum(danalytic, dnumeric)

        print(danalytic, dnumeric, error_relative)


def t1():
    struct = ['conv_32_5_1_0'] + ['pool'] + ['conv_64'] + ['pool'] + ['conv_128'] * 2 + ['pool'] + ['conv_256'] + [
        'FC_100']
    layer_param = [(32, 5, 1, 0, 'conv'), (32, 'pool'), (64, 3, 1, 1, 'conv'), (64, 'pool'), (128, 3, 1, 1, 'conv'),
                   (128, 3, 1, 1, 'conv'), (128, 'pool'), (256, 3, 1, 1, 'conv'), (100, 'FC')]

    check_gradient(layer_param, check_weight_or_bias=1, step=10 ** (-5), reg=10 ** (-50))


if __name__ == '__main__':
    t1()
