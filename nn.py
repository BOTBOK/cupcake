# copy 卷积神经网络python实现代码
import numpy as np

from create_data import gen_toy_data, split_data


def initialize_parameters(layer_param):
    weights = []
    biases = []
    vweights = []
    vbiases = []

    for i in range(len(layer_param) - 1):
        in_depth = layer_param[i]
        out_depth = layer_param[i + 1]
        std = np.sqrt(2 / in_depth) * 0.5
        weights.append(std * np.random.randn(in_depth, out_depth))
        biases.append(np.zeros((1, out_depth)))
        vweights.append(np.zeros((in_depth, out_depth)))
        vbiases.append(np.zeros((1, out_depth)))

    return weights, biases, vweights, vbiases


def forward(X, layer_param, weights, biases):
    hiddens = [X]
    for i in range(len(layer_param) - 2):
        hiddens.append(np.maximum(0, np.dot(hiddens[i], weights[i]) + biases[i]))
    scores = np.dot(hiddens[-1], weights[-1]) + biases[-1]
    return hiddens, scores


def data_loss_softmax(scores, labels):
    num_examples = scores.shape[0]
    exp_socres = np.exp(scores)
    exp_socres_sum = np.sum(exp_socres, axis=1)
    corect_probs = exp_socres[range(num_examples), labels] / exp_socres_sum
    corect_logprobs = -np.log(corect_probs)
    data_loss = np.sum(corect_logprobs) / num_examples
    return data_loss


# l2正则化损失函数
def reg_L2_loss(weights, reg):
    reg_loss = 0
    for weight in weights:
        reg_loss += 0.5 * reg * np.sum(weight * weight)
    return reg_loss


def decores_softmax(scores, labels):
    num_examples = scores.shape[0]
    exp_scores = np.exp(scores)
    decores_scores = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    decores_scores[range(num_examples), labels] -= 1
    # 暂时不理解为啥这里需要 除以num_examples
    decores_scores /= num_examples
    return decores_scores


def predict(X, labels, layer_param, weights, biases):
    hiddens = X
    for i in range(len(layer_param) - 2):
        hiddens = np.maximum(0, np.dot(hiddens, weights[i]) + biases[i])
    scores = np.dot(hiddens, weights[-1]) + biases[-1]
    predicted_class = np.argmax(scores, axis=1)
    right_class = predicted_class == labels
    return np.mean(right_class)


def gradient_backprop(dscores, hiddens, weights, biases, reg):
    dbiases = []
    dweights = []

    dhidden = dscores
    for i in range(len(hiddens)-1, -1, -1):
        dbiases.append(np.sum(dhidden, axis=0, keepdims=True))
        dweights.append(np.dot(hiddens[i].T, dhidden) + reg * weights[i])
        dhidden = np.dot(dhidden, weights[i].T)
        dhidden[dhidden <= 0] = 0
    return dweights, dbiases


def gen_random_data(dim, N_class, num_samp_per_class):
    num_example = num_samp_per_class * N_class
    X = np.random.randn(num_example, dim)
    labels = np.random.randint(N_class, size=num_example)
    return X, labels


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

        print(layer, data_loss1, data_loss2)
        error_relative = np.abs(danalytic - dnumeric) / np.maximum(danalytic, dnumeric)
        print(danalytic, dnumeric, error_relative)


if __name__ == '__main__':
    num_samp_per_class = 200
    dim = 2
    N_class = 4

    X, labels = gen_toy_data(dim, N_class, num_samp_per_class)
    X_train, labels_train, X_val, labels_val, X_test, labels_test = split_data(X, labels)

    print(X_train.shape, labels_train.shape)

    check_gradient(X, labels, [2, 100, 4], True)