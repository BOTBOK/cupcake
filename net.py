import random
from functools import reduce

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Node(object):
    def __init__(self):
        self.__out_val = 0
        self.__err_term = 0
        self.__err_out = 0
        self.__input_link = []
        self.__output_link = []

    def add_input_link(self, input_link):
        self.__input_link.append(input_link)

    def add_output_link(self, output_link):
        self.__output_link.append(output_link)

    def set_out_val(self, out_val):
        self.__out_val = out_val

    @property
    def out_val(self):
        return self.__out_val

    def cal_out_val(self):
        self.__out_val = 0
        for link in self.__input_link:
            self.__out_val += link.input_node.out_val * link.w
        self.__out_val = sigmoid(self.__out_val)

    def cal_err_out(self):
        self.__err_out = 0
        for link in self.__output_link:
            self.__err_out += link.output_node.err_term * link.w

    def cal_err_term(self):
        self.cal_err_out()
        self.__err_term = self.__err_out * self.__out_val * (1 - self.__out_val)


    def cal_end_err_term(self, real_val):
        self.__err_term = (self.__out_val - real_val) * self.__out_val * (1 - self.__out_val)

    @property
    def err_term(self):
        return self.__err_term

    @property
    def err_out(self):
        return self.__err_out


class StaticNode(object):
    def __init__(self):
        self.__out_val = 0
        self.__err_term = 0
        self.__input_link = []
        self.__output_link = []

    def add_input_link(self, input_link):
        self.__input_link.append(input_link)

    def add_output_link(self, output_link):
        self.__output_link.append(output_link)

    def set_out_val(self, out_val):
        self.__out_val = out_val

    @property
    def out_val(self):
        return 1

    def cal_out_val(self):
        pass

    def cal_err_term(self):
        pass

    def cal_end_err_term(self, real_val):
        pass

    @property
    def err_term(self):
        return 0


class Link(object):
    def __init__(self):
        self.__w = random.uniform(-0.1, 0.1)
        self.__dw = 0
        self.__input_node = None
        self.__output_node = None

    def set_input_node(self, input_node):
        self.__input_node = input_node

    def set_output_node(self, output_node):
        self.__output_node = output_node

    @property
    def w(self):
        return self.__w

    def set_w(self, w):
        self.__w = w

    @property
    def input_node(self):
        return self.__input_node

    @property
    def output_node(self):
        return self.__output_node

    def cal_dw(self):
        self.__dw = self.__output_node.err_term * self.__input_node.out_val

    @property
    def dw(self):
        return self.__dw

    def update_w(self, step):
        self.__w -= step * self.__dw


class Net(object):
    def __init__(self, input_num, hide_num_list, output_num):
        self.__layer_list = []
        input_layer = []
        for i in range(input_num):
            input_layer.append(Node())
        input_layer.append(StaticNode())
        self.__layer_list.append(input_layer)

        for hide_num in hide_num_list:
            hide_layer = []
            for i in range(hide_num):
                hide_layer.append(Node())
            hide_layer.append(StaticNode())
            self.__layer_list.append(hide_layer)

        output_layer = []
        for i in range(output_num):
            output_layer.append(Node())
        output_layer.append(StaticNode())
        self.__layer_list.append(output_layer)

        self.__link_list = []
        for index in range(len(self.__layer_list) - 1):
            for input_node in self.__layer_list[index]:
                for output_node in self.__layer_list[index + 1][:-1]:
                    link = Link()
                    input_node.add_output_link(link)
                    output_node.add_input_link(link)
                    link.set_input_node(input_node)
                    link.set_output_node(output_node)
                    self.__link_list.append(link)

    def cal_out(self, input_list):
        for i in range(len(input_list)):
            self.__layer_list[0][i].set_out_val(input_list[i])

        for layer in self.__layer_list[1:]:
            for node in layer:
                node.cal_out_val()

        out_val = []
        for node in self.__layer_list[-1]:
            out_val.append(node.out_val)
        return out_val[:-1]

    @property
    def err_out(self):
        ret_out = []
        for node in self.__layer_list[0][:-1]:
            ret_out.append(node.err_out)
        return ret_out

    def cal_err_out(self):
        for node in self.__layer_list[0][:-1]:
            node.cal_err_out()

    def cal_err_term(self, output_list):
        for i in range(len(output_list)):
            self.__layer_list[-1][i].cal_end_err_term(output_list[i])

        for layer in reversed(self.__layer_list[1:-1]):
            for node in layer:
                node.cal_err_term()

    def cal_dw(self):
        for link in self.__link_list:
            link.cal_dw()

    def update_w(self, step):
        for link in self.__link_list:
            link.update_w(step)

    def get_all_link(self):
        return self.__link_list


def cal_err(t, y):
    ret = 0
    for i in range(len(t)):
        ret += (t[i] - y[i]) * (t[i] - y[i]) * 0.5
    return ret


def test_degit():
    t_net = Net(10, [10], 10)
    link_list = t_net.get_all_link()
    a = []
    b = []
    for i in range(10):
        a.append(i)
        b.append(0)

    out_val = t_net.cal_out(a)
    t_net.cal_err_term(b)
    t_net.cal_dw()

    for link in link_list:
        print(link.dw)
        _e = 0.00001
        tmp_w = link.w
        tmp_w += _e
        link.set_w(tmp_w)
        t_out1 = t_net.cal_out(a)
        err1 = cal_err(np.array(t_out1[:-1]), np.array(b))

        tmp_w -= 2 * _e
        link.set_w(tmp_w)
        t_out2 = t_net.cal_out(a)
        err2 = cal_err(np.array(t_out2[:-1]), np.array(b))
        print((err1 - err2) / (2 * _e))

        tmp_w += _e
        link.set_w(tmp_w)


if __name__ == '__main__':
    test_degit()
