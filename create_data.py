# copy 卷积神经网络python实现代码
import matplotlib.pyplot as plt
import numpy as np


def gen_toy_data(dim, N_class, num_samp_per_class):
    num_examples = num_samp_per_class * N_class
    X = np.zeros((num_examples, dim))
    labels = np.zeros(num_examples, dtype='uint8')
    for j in range(N_class):
        ix = range(num_samp_per_class * j, num_samp_per_class * (j + 1))
        x = np.linspace(-np.pi, np.pi, num_samp_per_class) + 5
        y = np.sin(x + j * np.pi / (0.5 * N_class))
        y += 0.2 * np.sin(10 * x + j * np.pi / (0.5 * N_class))
        y += 0.25 * x + 10
        y += np.random.randn(num_samp_per_class) * 0.1

        X[ix] = np.c_[x, y]
        labels[ix] = j
    return X, labels


def show_data(X, labels):
    plt.scatter(X[:, 0], X[:, 1], c = labels, s = 40, cmap=plt.cm.Spectral)
    plt.show()


# 中心化和归一化处理
def normalize(X):
    mean = np.mean(X, axis=0)
    X_norm = X - mean
    std = np.std(X_norm, axis=0)
    X_norm /= std + 10 ** (-5)
    return X_norm, mean, std


# PCA 白化代码
def PCA_white(X):
    mean = np.mean(X, axis=0)
    X_norm = X - mean
    cov = np.dot(X_norm.T, X_norm)/X_norm.shape[0]
    U, S, V = np.linalg.svd(cov)
    X_norm = np.dot(X_norm, U)
    X_norm /= np.sqrt(S + 10 ** (-5))
    return X_norm, mean, U, S


def split_data(X, labels):
    num_examples = X.shape[0]
    shuffe_no = list(range(num_examples))
    np.random.shuffle(shuffe_no)

    X_train = X[shuffe_no[:num_examples//2]]
    labels_train = labels[shuffe_no[:num_examples//2]]

    X_val = X[shuffe_no[num_examples//2:num_examples//2 + num_examples//4]]

    labels_val = labels[shuffe_no[num_examples//2:num_examples//2 + num_examples//4]]

    X_test = X[shuffe_no[-num_examples//4:]]
    labels_test = labels[shuffe_no[-num_examples//4:]]

    return X_train, labels_train, X_val, labels_val, X_test, labels_test


# 数据预处理
def data_prepeocess(X_train, X_val, X_test):
    (X_train_pca, mean, U, S) = PCA_white(X_train)
    X_val_pca = np.dot(X_val - mean, U)
    X_val_pca /= np.sqrt(S + 10 ** (-5))
    X_test_pca = np.dot(X_test-mean, U)
    X_test_pca /= np.sqrt(S + 10 ** (-5))

    return X_train_pca, X_val_pca, X_test_pca


if __name__ == '__main__':
    num_samp_per_class = 200
    dim = 2
    N_class = 4

    X, labels = gen_toy_data(dim, N_class, num_samp_per_class)

    X, mean, std = normalize(X)

    X, mean, U, S = PCA_white(X)

    show_data(X, labels)

