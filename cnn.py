import copy

import numpy as np


class Activition(object):
    @staticmethod
    def forward(in_val):
        return max(0, in_val)

    @staticmethod
    def array_forward(input_val):
        deep, row, col = input_val.shape
        ret = np.zeros((deep, row, col), dtype=float)
        for index_deep in range(deep):
            for i in range(row):
                for j in range(col):
                    ret[index_deep][i][j] = Activition.forward(input_val[index_deep][i][j])
        return ret

    @staticmethod
    def backward(in_val):
        return 1 if in_val > 0 else 0

    @staticmethod
    def array_backward(input_val):
        deep, row, col = input_val.shape
        ret = np.zeros((deep, row, col), dtype=float)
        for index_deep in range(deep):
            for i in range(row):
                for j in range(col):
                    ret[index_deep][i][j] = Activition.backward(input_val[index_deep][i][j])
        return ret


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
        self.__out_val = np.zeros((deep, width, width), dtype=float)
        self.__net_val = np.zeros((deep, width, width), dtype=float)
        self.__err_term = np.zeros((deep, width, width), dtype=float)
        self.__input_filter = None
        self.__output_filter = None
        self.__input_feature_map = None
        self.__output_feature_map = None

    def set_input_filter(self, input_filter):
        self.__input_filter = input_filter

    def set_output_filter(self, output_filter):
        self.__output_filter = output_filter

    def set_input_feature_map(self, feature_map):
        self.__input_feature_map = feature_map

    def set_output_feature_map(self, feature_map):
        self.__output_feature_map = feature_map

    def cal_out(self):
        for i in range(len(self.__input_filter.w_list)):
            self.__net_val[i] = ConOp.convolution(self.__input_feature_map.out_val, self.__input_filter.w_list[i],
                                                  self.__step, self.__input_filter.bias[i])
        self.__out_val = Activition.array_forward(self.__net_val)

    def cal_err_term(self):
        cal1 = ConOp.add_zero_circle(self.__output_feature_map.err_term, self.__output_filter.w_list[0].shape[1] - 1)
        cal1 = ConOp.add_row_col_by_step(cal1, self.__step)

        deep, row, col = self.__output_filter.w_list[0].shape
        for index_deep in range(deep):
            cal2 = []
            for i in range(len(self.__output_filter.w_list)):
                cal2.append(self.__output_filter.w_list[i][index_deep])
            cal2 = ConOp.rot90(np.array(cal2), 2)
            self.__err_term[index_deep] = ConOp.convolution(cal1, cal2, 1, 0)
        self.__err_term = self.__err_term * Activition.array_backward(self.__net_val)

    def set_err_term(self, err_term):
        self.__err_term = err_term

    def set_out_val(self, out_val):
        self.__out_val = out_val

    @property
    def net_val(self):
        return self.__net_val

    @property
    def out_val(self):
        return self.__out_val

    @property
    def err_term(self):
        return self.__err_term


def judge_awl(in1, in2):
    print("======================")
    print(np.sum(in1 - in2))
    print("======================")

    deep, row, col = in1.shape
    for index_deep in range(deep):
        for i in range(row):
            for j in range(col):
                a = in1[index_deep][i][j]
                b = in2[index_deep][i][j]
                if a > 0 and b <= 0:
                    return True
                if a <= 0 and b >0:
                    return True
    return False


if __name__ == '__main__':

    a = FeatureMap(5, 1, 1)
    w1 = FilterLayer(2, 1, 1, 1)
    b = FeatureMap(4, 1, 1)
    w2 = FilterLayer(2, 1, 1, 1)
    c = FeatureMap(3, 1, 1)

    a.set_output_feature_map(b)
    a.set_output_filter(w1)

    b.set_input_feature_map(a)
    b.set_output_feature_map(c)
    b.set_input_filter(w1)
    b.set_output_filter(w2)

    w1.set_input_feature_map(a)
    w1.set_output_feature_map(b)

    c.set_input_feature_map(b)
    c.set_input_filter(w2)
    w2.set_input_feature_map(b)
    w2.set_output_feature_map(c)

    in_val = []
    for i in range(1 * 5 * 5):
        in_val.append(i)

    in_val = np.array(in_val).reshape((1, 5, 5))

    out_val = np.ones((1, 3, 3), dtype=float)

    a.set_out_val(in_val)
    b.cal_out()
    c.cal_out()

    out_val = out_val * Activition.array_backward(out_val)

    c.set_err_term(out_val)
    b.cal_err_term()
    a.cal_err_term()

    w1.cal_dw()
    w2.cal_dw()

    err_term = lambda a: np.sum(a)

    w_list = w1.w_list
    dw = w1.dw
    o11 = c.net_val
    for i in range(len(w_list)):
        deep, row, col = w_list[i].shape
        for index_deep in range(deep):
            for r in range(row):
                for co in range(col):
                    e = 0.00001
                    w_list[i][index_deep][r][co] += e
                    w1.set_w(w_list)
                    b.cal_out()
                    c.cal_out()
                    o1 = c.net_val
                    err1 = err_term(c.out_val)
                    if judge_awl(o11, o1):
                        continue

                    w_list[i][index_deep][r][co] -= 2 * e
                    w1.set_w(w_list)
                    b.cal_out()
                    c.cal_out()
                    o2 = c.net_val
                    err2 = err_term(c.out_val)
                    if judge_awl(o11, o2):
                        continue

                    w_list[i][index_deep][r][co] += e
                    w1.set_w(w_list)
                    if judge_awl(o1, o2):
                        continue

                    print((err1 - err2) / (2 * e))
                    print(dw[i][index_deep][r][co])


