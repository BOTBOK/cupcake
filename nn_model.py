import numpy as np


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
        #hiddens.append(np.dot(hiddens[i], weights[i]) + biases[i])

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
    for i in range(len(hiddens) - 1, -1, -1):
        dbiases.append(np.sum(dhidden, axis=0, keepdims=True))
        dweights.append(np.dot(hiddens[i].T, dhidden) + reg * weights[i])
        dhidden = np.dot(dhidden, weights[i].T)
        dhidden[hiddens[i] <= 0] = 0
    return dweights, dbiases


def gen_random_data(dim, N_class, num_samp_per_class):
    num_example = num_samp_per_class * N_class
    X = np.random.randn(num_example, dim)
    labels = np.random.randint(N_class, size=num_example)
    return X, labels


def nesterov_momentumSGD(vparams, params, dparams, lr, mu):
    update_ratio = []
    for i in range(len(params)):
        pre_vparam = vparams[i]
        vparams[i] = mu * vparams[i] - lr * dparams[-1 - i]
        update_param = vparams[i] + mu * (vparams[i] - pre_vparam)
        params[i] += update_param
        # params[i] += - lr * dparams[-1 - i]
        update_ratio.append(np.sum(np.abs(update_param)) / np.sum(np.abs(params[i])))

    return update_ratio
