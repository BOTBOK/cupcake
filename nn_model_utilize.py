# copy 卷积神经网络python实现代码
import numpy as np
import matplotlib.pyplot as plt

from data_process import gen_toy_data, split_data, normalize, PCA_white
from nn_model import initialize_parameters, forward, decores_softmax, gradient_backprop, data_loss_softmax, reg_L2_loss, \
    predict, nesterov_momentumSGD


def check_gradient(X, lables, layer_param, check_weight_or_bias):
    # (X, lables) = gen_random_data(dim)
    (weights, biases, vweights, vbiases) = initialize_parameters(layer_param)
    reg = 10 ** (-9)
    step = 10 ** (-5)
    for layer in range(len(weights)):
        if check_weight_or_bias:
            row = np.random.randint(weights[layer].shape[0])
            col = np.random.randint(weights[layer].shape[1])
            param = weights[layer][row][col]
        else:
            row = np.random.randint(biases[layer].shape[1])
            param = biases[layer][0][row]

        hiddens, scores = forward(X, layer_param, weights, biases)

        dscores = decores_softmax(scores, lables)
        dweights, dbiases = gradient_backprop(dscores, hiddens, weights, biases, reg)

        if check_weight_or_bias:
            danalytic = dweights[-1 - layer][row][col]
        else:
            danalytic = dbiases[-1 - layer][0][row]

        if check_weight_or_bias:
            weights[layer][row][col] = param - step
        else:
            biases[layer][0][row] = param - step

        hiddens, scores = forward(X, layer_param, weights, biases)
        data_loss1 = data_loss_softmax(scores, lables)
        reg_loss1 = reg_L2_loss(weights, reg)
        loss1 = data_loss1 + reg_loss1

        if check_weight_or_bias:
            weights[layer][row][col] = param + step
        else:
            biases[layer][0][row] = param + step

        hiddens, scores = forward(X, layer_param, weights, biases)
        data_loss2 = data_loss_softmax(scores, lables)
        reg_loss2 = reg_L2_loss(weights, reg)
        loss2 = data_loss2 + reg_loss2

        dnumeric = (loss2 - loss1) / (2 * step)

        error_relative = np.abs(danalytic - dnumeric) / np.maximum(danalytic, dnumeric)


def train_net(X_train, labels_train, layer_param, lr, lr_decay, reg, mu, max_epoch, X_val, labels_val):
    weights, biases, vweights, vbiases = initialize_parameters(layer_param)
    epoch = 0
    data_losses = []
    reg_losses = []

    val_accuracy = []
    train_accuracy = []
    weights_update_ratio = []
    biases_update_ratio = []
    while epoch < max_epoch:
        (hiddens, scores) = forward(X_train, layer_param, weights, biases)

        val_accuracy.append(predict(X_val, labels_val, layer_param, weights, biases))

        train_accuracy.append(predict(X_train, labels_train, layer_param, weights, biases))

        data_loss = data_loss_softmax(scores, labels_train)

        reg_loss = reg_L2_loss(weights, reg)

        data_losses.append(data_loss)
        reg_losses.append(reg_loss)

        dscores = decores_softmax(scores, labels_train)
        (dweights, dbiases) = gradient_backprop(dscores, hiddens, weights, biases, reg)

        weights_update_ratio.append(nesterov_momentumSGD(vweights, weights, dweights, lr, mu))
        biases_update_ratio.append(nesterov_momentumSGD(vbiases, biases, dbiases, lr, mu))

        epoch += 1
        lr *= lr_decay

    plt.close()
    fig = plt.figure('loss')
    ax = fig.add_subplot(2, 1, 1)
    ax.grid(True)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.grid(True)
    plt.xlabel('test')
    plt.ylabel('accuracy log10(data loss)', fontsize=14)
    ax.scatter(np.arange(len(data_losses)), np.log10(data_losses), c='b', marker='.')
    ax2.scatter(np.arange(len(val_accuracy) - 0), val_accuracy[0:], c='r', marker='*')
    ax2.scatter(np.arange(len(train_accuracy) - 0), train_accuracy[0:], c='g', marker='.')
    plt.show()

    return data_losses, reg_losses, weights, biases, val_accuracy


def overfit_tinydata(X, labels, layer_param, lr=10 ** (-2.1), lr_decay=1, mu=0.9, reg=0, max_epoch=100):
    data_losses, reg_losses, weights, biases, val_accuracy = train_net(X, labels, layer_param, lr, lr_decay, reg, mu,
                                                                       max_epoch, X, labels)


def t1():
    num_samp_per_class = 2
    dim = 2
    N_class = 4

    X, labels = gen_toy_data(dim, N_class, num_samp_per_class)

    X_norm, mean, std = normalize(X)

    X_norm, mean, U, S = PCA_white(X_norm)

    layer_param = [dim, 100, 100, N_class]

    overfit_tinydata(X_norm, labels, layer_param)

    X_train, labels_train, X_val, labels_val, X_test, labels_test = split_data(X_norm, labels)

    check_gradient(X, labels, [2, 100, 4], True)


def t2():
    num_samp_per_class = 200
    dim = 2
    N_class = 4

    # 生成数据
    X, labels = gen_toy_data(dim, N_class, num_samp_per_class)
    X_norm, mean, std = normalize(X)
    X_norm, mean, U, S = PCA_white(X_norm)
    X_train, labels_train, X_val, labels_val, X_test, labels_test = split_data(X_norm, labels)

    lr = 10 ** (-2.1)
    lr_decay = 1
    reg = 10 ** (-4.3)
    mu = 0.9
    max_epoch = 10000

    # 训练
    layer_param = [dim, 100, 100, N_class]
    train_net(X_train, labels_train, layer_param, lr, lr_decay, reg, mu, max_epoch, X_val, labels_val)


if __name__ == '__main__':
    t2()
