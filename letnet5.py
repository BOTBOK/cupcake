import copy

import numpy as np

from activation import array_forward, SigmoidActivation, array_backward


class ConOp(object):
    @staticmethod
    def convolution(in1, in2, step, bias):
        if in1.ndim == 2:
            a1, b1 = in1.shape
            a2, b2 = in2.shape
            cal1 = in1.reshape((1, a1, b1))
            cal2 = in2.reshape((1, a2, b2))
        else:
            cal1 = in1
            cal2 = in2
        deep1, row1, col1 = cal1.shape
        deep2, row2, col2 = cal2.shape
        if deep1 != deep2:
            return None
        new_row = (row1 - row2) / step + 1
        ret_val = np.zeros((int(new_row), int(new_row)), dtype=float)
        for index_deep in range(deep1):
            for r in range(0, row1 - row2 + step, step):
                for c in range(0, col1 - col2 + step, step):
                    ret_val[int(r / step)][int(c / step)] += \
                        np.sum(cal1[index_deep][r: r + row2, c: c + col2] * cal2[index_deep]) + bias
        return ret_val

    @staticmethod
    def add_zero_circle(val, cir_num):
        deep, row, col = val.shape
        ret_val = np.zeros((deep, row + 2 * cir_num, col + 2 * cir_num))
        for index_deep in range(deep):
            ret_val[index_deep][cir_num: cir_num + row, cir_num: cir_num + col] = val[index_deep]
        return ret_val

    @staticmethod
    def add_row_col_by_step(val, step):
        return val

    @staticmethod
    def rot90(val, k):
        ret_val = copy.copy(val)
        deep, row, col = val.shape
        for i in range(deep):
            ret_val[i] = np.rot90(val[i], k)
        return ret_val


class FilterLayer(object):
    def __init__(self, width, input_deep, output_deep, step):
        self.__w_list = []
        self.__dw_list = []
        self.__bias = []
        self.__db = []
        for i in range(output_deep):
            self.__w_list.append(np.random.uniform(-1e-4, 1e-4, (input_deep, width, width)))
            self.__dw_list.append(np.zeros((input_deep, width, width), dtype=float))
            self.__bias.append(np.random.uniform(-1e-4, 1e-4))
            self.__db.append(0)

        self.__step = step
        self.__input_feature_map = None
        self.__output_feature_map = None

    def set_input_feature_map(self, feature_map):
        self.__input_feature_map = feature_map

    def set_output_feature_map(self, feature_map):
        self.__output_feature_map = feature_map

    @property
    def output_feature_map(self):
        return self.__output_feature_map

    def cal_dw(self):
        cal1 = self.__input_feature_map.out_val
        cal2 = ConOp.add_row_col_by_step(self.__output_feature_map.err_term, self.__step)
        input_deep, row1, col1 = cal1.shape
        output_deep, row2, col2 = cal2.shape
        for i in range(output_deep):
            for j in range(input_deep):
                self.__dw_list[i][j] = ConOp.convolution(cal1[j], cal2[i], 1, 0)
            self.__db[i] = np.sum(cal2[i])

    def update_w(self):
        for i in range(len(self.__dw_list)):
            self.__w_list[i] -= self.__dw_list[i]
            self.__bias[i] -= self.__db[i]

    @property
    def dw(self):
        return self.__dw_list

    @property
    def w_list(self):
        return self.__w_list

    def set_b(self, bias):
        self.__bias = bias

    def set_w(self, w):
        self.__w_list = w

    @property
    def bias(self):
        return self.__bias


class FeatureMap(object):
    def __init__(self, width, deep, step):
        self.__step = step
        self.__net_val = np.zeros((deep, width, width), dtype=float)
        self.__net_val_without_w_b = np.zeros((deep, width, width), dtype=float)
        self.__out_val = np.zeros((deep, width, width), dtype=float)
        self.__err_term = np.zeros((deep, width, width), dtype=float)
        self.__err_out = np.zeros((deep, width, width), dtype=float)
        self.__input_filter = None
        self.__output_filter = None
        self.__input_feature_map = None
        self.__output_feature_map = None

        self.__dw = np.ones((deep, width, width), dtype=float)
        self.__db = np.ones((deep, width, width), dtype=float)
        self.__w = np.random.uniform(-0.1, 0.1, (deep, width, width))
        self.__b = np.random.uniform(-0.1, 0.1, (deep, width, width))

    def set_input_filter(self, input_filter):
        self.__input_filter = input_filter

    def set_output_filter(self, output_filter):
        self.__output_filter = output_filter

    def set_input_feature_map(self, feature_map):
        self.__input_feature_map = feature_map

    def set_output_feature_map(self, feature_map):
        self.__output_feature_map = feature_map

    def cal_net_out(self):
        for i in range(len(self.__input_filter.w_list)):
            self.__net_val[i] = ConOp.convolution(self.__input_feature_map.out_val, self.__input_filter.w_list[i],
                                                  self.__step, self.__input_filter.bias[i])

    def cal_out_val(self):
        self.__out_val = self.__net_val

    def cal_err_out(self):
        cal1 = ConOp.add_zero_circle(self.__output_feature_map.err_term, self.__output_filter.w_list[0].shape[1] - 1)
        cal1 = ConOp.add_row_col_by_step(cal1, self.__step)
        deep, row, col = self.__output_filter.w_list[0].shape
        for index_deep in range(deep):
            cal2 = []
            for i in range(len(self.__output_filter.w_list)):
                cal2.append(self.__output_filter.w_list[i][index_deep])
            cal2 = ConOp.rot90(np.array(cal2), 2)
            self.__err_out[index_deep] = ConOp.convolution(cal1, cal2, 1, 0)

    def cal_err_term(self):
        self.__err_term = self.__err_out

    def set_err_term(self, err_term):
        self.__err_term = err_term

    def set_out_val(self, out_val):
        self.__out_val = out_val

    def get_err_out_from_pooling(self):
        next_out_val = self.__output_feature_map.out_val
        next_err_out = self.__output_feature_map.err_out
        pooling_w = self.__output_feature_map.pooling_w
        deep, row, col = next_err_out.shape
        for index_deep in range(deep):
            for i in range(row):
                for j in range(col):
                    ti = i * 2
                    tj = j * 2
                    tmp_err_val = next_err_out[index_deep][i][j] * pooling_w[index_deep][i][
                        j] * SigmoidActivation.backward(next_out_val[index_deep][i][j])
                    self.__err_out[index_deep][ti][tj] = tmp_err_val
                    self.__err_out[index_deep][ti + 1][tj] = tmp_err_val
                    self.__err_out[index_deep][ti][tj + 1] = tmp_err_val
                    self.__err_out[index_deep][ti + 1][tj + 1] = tmp_err_val

    def pooling_cal_net_val(self):
        input_out_val = self.__input_feature_map.out_val
        deep, row, col = self.__net_val.shape
        for index_deep in range(deep):
            for i in range(row):
                for j in range(col):
                    oi = 2 * i
                    oj = 2 * j
                    self.__net_val_without_w_b[index_deep][i][j] = np.sum(
                        input_out_val[index_deep][oi: oi + 2, oj: oj + 2])
                    self.__net_val[index_deep][i][j] = self.__net_val_without_w_b[index_deep][i][j] * \
                                                       self.__w[index_deep][i][j] + self.__b[index_deep][i][j]

    def pooling_cal_out_val(self):
        self.__out_val = array_forward(self.__net_val, SigmoidActivation)

    def cal_dw(self):
        tmp_err = self.__err_out * array_backward(self.__out_val, SigmoidActivation)
        self.__dw = tmp_err * self.__net_val_without_w_b
        self.__db = tmp_err

    @property
    def net_val(self):
        return self.__net_val

    @property
    def out_val(self):
        return self.__out_val

    @property
    def err_out(self):
        return self.__err_out

    @property
    def err_term(self):
        return self.__err_term

    @property
    def pooling_w(self):
        return self.__w

    @property
    def dw(self):
        return self.__dw

    def set_pooling_w(self, w):
        self.__w = w


def judge_awl(in1, in2):
    deep, row, col = in1.shape
    for index_deep in range(deep):
        for i in range(row):
            for j in range(col):
                a = in1[index_deep][i][j]
                b = in2[index_deep][i][j]
                if a <= 0 < b:
                    return True
                if a > 0 >= b:
                    return True
    return False


def connection_map(a, b, w=None):
    a.set_output_feature_map(b)
    if w is not None:
        a.set_output_filter(w)
        w.set_input_feature_map(a)

        w.set_output_feature_map(b)
        b.set_input_filter(w)

    b.set_input_feature_map(a)


def test():
    a = FeatureMap(32, 1, 1)
    w1 = FilterLayer(5, 1, 6, 1)
    b = FeatureMap(28, 6, 1)
    c = FeatureMap(14, 6, 1)
    w2 = FilterLayer(5, 6, 16, 1)
    d = FeatureMap(10, 16, 1)
    e = FeatureMap(5, 16, 1)
    w3 = FilterLayer(5, 16, 120, 1)
    f = FeatureMap(1, 120, 1)

    connection_map(a, b, w1)
    connection_map(b, c)
    connection_map(c, d, w2)
    connection_map(d, e)
    connection_map(e, f, w3)

    in_val = []
    for i in range(1 * 32 * 32):
        in_val.append(i)

    in_val = np.array(in_val).reshape((1, 32, 32))

    out_val = np.ones((120, 1, 1), dtype=float)

    a.set_out_val(in_val)
    b.cal_net_out()
    b.cal_out_val()
    c.pooling_cal_net_val()
    c.pooling_cal_out_val()
    d.cal_net_out()
    d.cal_out_val()
    e.pooling_cal_net_val()
    e.pooling_cal_out_val()
    f.cal_net_out()
    f.cal_out_val()

    f.set_err_term(out_val)
    e.cal_err_out()
    e.cal_err_term()
    d.get_err_out_from_pooling()
    d.cal_err_term()
    c.cal_err_out()
    c.cal_err_term()
    b.get_err_out_from_pooling()
    b.cal_err_term()
    a.cal_err_out()
    a.cal_err_term()

    w1.cal_dw()
    w2.cal_dw()

    err_term = lambda a: np.sum(a)

    w_list = w1.w_list
    dw = w1.dw
    for i in range(len(w_list)):
        deep, row, col = w_list[i].shape
        for index_deep in range(deep):
            for r in range(row):
                for co in range(col):
                    es = 0.00001
                    w_list[i][index_deep][r][co] += es
                    w1.set_w(w_list)

                    b.cal_net_out()
                    b.cal_out_val()
                    c.pooling_cal_net_val()
                    c.pooling_cal_out_val()
                    d.cal_net_out()
                    d.cal_out_val()
                    e.pooling_cal_net_val()
                    e.pooling_cal_out_val()
                    f.cal_net_out()
                    f.cal_out_val()
                    err1 = err_term(f.out_val)

                    w_list[i][index_deep][r][co] -= 2 * es
                    w1.set_w(w_list)

                    b.cal_net_out()
                    b.cal_out_val()
                    c.pooling_cal_net_val()
                    c.pooling_cal_out_val()
                    d.cal_net_out()
                    d.cal_out_val()
                    e.pooling_cal_net_val()
                    e.pooling_cal_out_val()
                    f.cal_net_out()
                    f.cal_out_val()
                    err2 = err_term(f.out_val)

                    w_list[i][index_deep][r][co] += es
                    w1.set_w(w_list)

                    print((err1 - err2) / (2 * es))
                    print(dw[i][index_deep][r][co])

def test2():
    a = FeatureMap(32, 1, 1)
    w1 = FilterLayer(5, 1, 6, 1)
    b = FeatureMap(28, 6, 1)
    c = FeatureMap(14, 6, 1)
    w2 = FilterLayer(5, 6, 16, 1)
    d = FeatureMap(10, 16, 1)
    e = FeatureMap(5, 16, 1)
    w3 = FilterLayer(5, 16, 120, 1)
    f = FeatureMap(1, 120, 1)

    connection_map(a, b, w1)
    connection_map(b, c)
    connection_map(c, d, w2)
    connection_map(d, e)
    connection_map(e, f, w3)

    in_val = []
    for i in range(1 * 32 * 32):
        in_val.append(i)

    in_val = np.array(in_val).reshape((1, 32, 32))

    out_val = np.ones((120, 1, 1), dtype=float)

    a.set_out_val(in_val)
    b.cal_net_out()
    b.cal_out_val()
    c.pooling_cal_net_val()
    c.pooling_cal_out_val()
    d.cal_net_out()
    d.cal_out_val()
    e.pooling_cal_net_val()
    e.pooling_cal_out_val()
    f.cal_net_out()
    f.cal_out_val()

    f.set_err_term(out_val)
    e.cal_err_out()
    e.cal_err_term()
    d.get_err_out_from_pooling()
    d.cal_err_term()
    c.cal_err_out()
    c.cal_err_term()
    b.get_err_out_from_pooling()
    b.cal_err_term()
    a.cal_err_out()
    a.cal_err_term()

    w1.cal_dw()
    w2.cal_dw()
    c.cal_dw()

    err_term = lambda a: np.sum(a)

    w_list = c.pooling_w
    dw = c.dw
    deep, row, col = w_list.shape
    for index_deep in range(deep):
        for r in range(row):
            for co in range(col):
                es = 0.00001
                w_list[index_deep][r][co] += es
                c.set_pooling_w(w_list)

                b.cal_net_out()
                b.cal_out_val()
                c.pooling_cal_net_val()
                c.pooling_cal_out_val()
                d.cal_net_out()
                d.cal_out_val()
                e.pooling_cal_net_val()
                e.pooling_cal_out_val()
                f.cal_net_out()
                f.cal_out_val()
                err1 = err_term(f.out_val)

                w_list[index_deep][r][co] -= 2 * es
                c.set_pooling_w(w_list)

                b.cal_net_out()
                b.cal_out_val()
                c.pooling_cal_net_val()
                c.pooling_cal_out_val()
                d.cal_net_out()
                d.cal_out_val()
                e.pooling_cal_net_val()
                e.pooling_cal_out_val()
                f.cal_net_out()
                f.cal_out_val()
                err2 = err_term(f.out_val)

                w_list[index_deep][r][co] += es
                c.set_pooling_w(w_list)

                print((err1 - err2) / (2 * es))
                print(dw[index_deep][r][co])


if __name__ == '__main__':
    test()