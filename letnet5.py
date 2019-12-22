import copy
import struct
import numpy as np

from activation import array_forward, SigmoidActivation, array_backward
from net import Net


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


def copy2arr(arr):
    deep, row, col = arr.shape
    ret = np.zeros((deep * 2, row, col), dtype=float)
    ret[0: deep] = arr
    ret[deep: deep * 2] = copy.copy(arr)
    return ret


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

    def update_w(self, step):
        for i in range(len(self.__dw_list)):
            self.__w_list[i] -= step * self.__dw_list[i]
            self.__bias[i] -= step * self.__db[i]

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

    def cal_3_6_dw(self):
        out_err_term = self.__output_feature_map.err_term[0: 6]
        out_err_term = ConOp.add_row_col_by_step(out_err_term, self.__step)
        input_val = copy2arr(self.__input_feature_map.out_val)

        deep, row, col = self.__w_list[0].shape
        for i in range(len(self.__w_list)):
            for j in range(deep):
                self.__dw_list[i][j] = ConOp.convolution(input_val[j + i], out_err_term[i], 1, 0)

    def cal_4_6_dw(self):
        out_err_term = self.__output_feature_map.err_term[6: 12]
        out_err_term = ConOp.add_row_col_by_step(out_err_term, self.__step)
        input_val = copy2arr(self.__input_feature_map.out_val)

        deep, row, col = self.__w_list[0].shape
        for i in range(len(self.__w_list)):
            for j in range(deep):
                self.__dw_list[i][j] = ConOp.convolution(input_val[j + i], out_err_term[i], 1, 0)

    def cal_2_2_dw(self):
        out_err_term = self.__output_feature_map.err_term[12: 15]
        out_err_term = ConOp.add_row_col_by_step(out_err_term, self.__step)
        input_val = copy2arr(self.__input_feature_map.out_val)

        for i in range(len(self.__w_list)):
            self.__dw_list[i][0] = ConOp.convolution(input_val[0 + i], out_err_term[i], 1, 0)
            self.__dw_list[i][1] = ConOp.convolution(input_val[1 + i], out_err_term[i], 1, 0)
            self.__dw_list[i][2] = ConOp.convolution(input_val[3 + i], out_err_term[i], 1, 0)
            self.__dw_list[i][3] = ConOp.convolution(input_val[4 + i], out_err_term[i], 1, 0)

    def cal_1_1_dw(self):
        out_err_term = self.__output_feature_map.err_term[15]
        out_err_term = ConOp.add_row_col_by_step(out_err_term, self.__step)
        input_val = self.__input_feature_map.out_val

        deep, row, col = self.__w_list[0].shape
        for i in range(deep):
            self.__dw_list[0][i] = ConOp.convolution(input_val[i], out_err_term, 1, 0)


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

    def set_err_out(self, err_out):
        self.__err_out = err_out

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

    def update_w(self, step):
        self.__w -= step * self.__dw
        self.__b -= step * self.__db

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

    @property
    def input_feature_map(self):
        return self.__input_feature_map

    @property
    def input_filter(self):
        return self.__input_filter

    def set_pooling_w(self, w):
        self.__w = w


class C3FeatureMap(FeatureMap):
    def __init__(self, width, deep, step):
        super().__init__(width, deep, step)
        self.__step = step
        self.__input_filter_3_6 = None
        self.__input_filter_4_6 = None
        self.__input_filter_2_2_3 = None
        self.__input_filter_1_1_1 = None

        self.__tmp_out_val = np.zeros((deep, width, width), dtype=float)

        self.__input_err_term = np.zeros((6, 14, 14), dtype=float)

    @property
    def input_err_term(self):
        return copy.copy(self.__input_err_term)

    def set_3_6_filter(self, filter_3_6):
        self.__input_filter_3_6 = filter_3_6

    def set_4_6_filter(self, filter_4_6):
        self.__input_filter_4_6 = filter_4_6

    def set_4_3_filter(self, filter_4_3):
        self.__input_filter_2_2_3 = filter_4_3

    def set_1_1_filter(self, filter_1_1):
        self.__input_filter_1_1_1 = filter_1_1

    def cal_3_6_out_val(self):
        input_feature_map = copy2arr(self.input_feature_map.out_val)

        w_list = self.__input_filter_3_6.w_list
        bias = self.__input_filter_3_6.bias
        for i in range(len(w_list)):
            self.__tmp_out_val[i] = ConOp.convolution(input_feature_map[i: i + 3], w_list[i], 1, bias[i])

    def cal_4_6_out_val(self):
        input_feature_map = copy2arr(self.input_feature_map.out_val)
        w_list = self.__input_filter_4_6.w_list
        bias = self.__input_filter_4_6.bias

        for i in range(len(w_list)):
            self.__tmp_out_val[i + 6] = ConOp.convolution(input_feature_map[i: i + 4], w_list[i], 1, bias[i])

    def cal_2_2_3_out_val(self):
        input_feature_map = self.input_feature_map.out_val
        deep, row, col = input_feature_map.shape
        w_list = self.__input_filter_2_2_3.w_list
        bias = self.__input_filter_2_2_3.bias

        tmp_map = np.zeros((4, row, col), dtype=float)
        tmp_map[0] = input_feature_map[0]
        tmp_map[1] = input_feature_map[1]
        tmp_map[2] = input_feature_map[3]
        tmp_map[3] = input_feature_map[4]
        self.__tmp_out_val[12] = ConOp.convolution(tmp_map, w_list[0], 1, bias[0])

        tmp_map = np.zeros((4, row, col), dtype=float)
        tmp_map[0] = input_feature_map[1]
        tmp_map[1] = input_feature_map[2]
        tmp_map[2] = input_feature_map[4]
        tmp_map[3] = input_feature_map[5]
        self.__tmp_out_val[13] = ConOp.convolution(tmp_map, w_list[1], 1, bias[1])

        tmp_map = np.zeros((4, row, col), dtype=float)
        tmp_map[0] = input_feature_map[2]
        tmp_map[1] = input_feature_map[3]
        tmp_map[2] = input_feature_map[5]
        tmp_map[3] = input_feature_map[0]
        self.__tmp_out_val[14] = ConOp.convolution(tmp_map, w_list[2], 1, bias[2])

    def cal_1_1_1_out_val(self):
        input_feature_map = self.input_feature_map.out_val
        w_list = self.__input_filter_1_1_1.w_list
        bias = self.__input_filter_1_1_1.bias

        self.__tmp_out_val[15] = ConOp.convolution(input_feature_map, w_list[0], 1, bias[0])

    def cal_out_val(self):
        self.cal_4_6_out_val()
        self.cal_3_6_out_val()
        self.cal_2_2_3_out_val()
        self.cal_1_1_1_out_val()

        self.set_out_val(self.__tmp_out_val)

    def __cal_inner_3_6_err_term(self, tmp_cal1, w_list, a, b, c):
        cal1 = np.zeros((3, 18, 18), dtype=float)
        cal2 = np.zeros((3, 5, 5), dtype=float)

        cal1[0] = tmp_cal1[a]
        cal2[0] = w_list[a][0]
        cal1[1] = tmp_cal1[b]
        cal2[1] = w_list[b][1]
        cal1[2] = tmp_cal1[c]
        cal2[2] = w_list[c][2]

        return ConOp.convolution(cal1, ConOp.rot90(cal2, 2), 1, 0)

    def cal_3_6_err_term(self, tmp_cal):
        w_list = self.__input_filter_3_6.w_list

        self.__input_err_term[0] += self.__cal_inner_3_6_err_term(tmp_cal, w_list, 0, 5, 4)
        self.__input_err_term[1] += self.__cal_inner_3_6_err_term(tmp_cal, w_list, 1, 0, 5)
        self.__input_err_term[2] += self.__cal_inner_3_6_err_term(tmp_cal, w_list, 2, 1, 0)
        self.__input_err_term[3] += self.__cal_inner_3_6_err_term(tmp_cal, w_list, 3, 2, 1)
        self.__input_err_term[4] += self.__cal_inner_3_6_err_term(tmp_cal, w_list, 4, 3, 2)
        self.__input_err_term[5] += self.__cal_inner_3_6_err_term(tmp_cal, w_list, 5, 4, 3)

    def __cal_inner_4_6_err_term(self, tmp_cal1, w_list, a, b, c, d):
        cal1 = np.zeros((4, 18, 18), dtype=float)
        cal2 = np.zeros((4, 5, 5), dtype=float)

        cal1[0] = tmp_cal1[a + 6]
        cal2[0] = w_list[a][0]
        cal1[1] = tmp_cal1[b + 6]
        cal2[1] = w_list[b][1]
        cal1[2] = tmp_cal1[c + 6]
        cal2[2] = w_list[c][2]
        cal1[3] = tmp_cal1[d + 6]
        cal2[3] = w_list[d][3]
        return ConOp.convolution(cal1, ConOp.rot90(cal2, 2), 1, 0)

    def cal_4_6_err_term(self, tmp_cal):
        w_list = self.__input_filter_4_6.w_list

        self.__input_err_term[0] += self.__cal_inner_4_6_err_term(tmp_cal, w_list, 0, 5, 4, 3)
        self.__input_err_term[1] += self.__cal_inner_4_6_err_term(tmp_cal, w_list, 1, 0, 5, 4)
        self.__input_err_term[2] += self.__cal_inner_4_6_err_term(tmp_cal, w_list, 2, 1, 0, 5)
        self.__input_err_term[3] += self.__cal_inner_4_6_err_term(tmp_cal, w_list, 3, 2, 1, 0)
        self.__input_err_term[4] += self.__cal_inner_4_6_err_term(tmp_cal, w_list, 4, 3, 2, 1)
        self.__input_err_term[5] += self.__cal_inner_4_6_err_term(tmp_cal, w_list, 5, 4, 3, 2)

    def __cal_inner_2_2_3_err_term(self, tmp_cal1, w_list, a, b, c, d):
        cal1 = np.zeros((2, 18, 18), dtype=float)
        cal2 = np.zeros((2, 5, 5), dtype=float)
        index = 0
        if a is not None:
            cal1[index] = tmp_cal1[a + 12]
            cal2[index] = w_list[a][0]
            index += 1
        if b is not None:
            cal1[index] = tmp_cal1[b + 12]
            cal2[index] = w_list[b][1]
            index += 1
        if c is not None:
            cal1[index] = tmp_cal1[c + 12]
            cal2[index] = w_list[c][2]
            index += 1
        if d is not None:
            cal1[index] = tmp_cal1[d + 12]
            cal2[index] = w_list[d][3]
            index += 1
        return ConOp.convolution(cal1, ConOp.rot90(cal2, 2), 1, 0)

    def cal_2_2_3_err_term(self, tmp_cal):
        w_list = self.__input_filter_2_2_3.w_list

        self.__input_err_term[0] += self.__cal_inner_2_2_3_err_term(tmp_cal, w_list, 0,    None, None, 2)
        self.__input_err_term[1] += self.__cal_inner_2_2_3_err_term(tmp_cal, w_list, 1,    0,    None, None)
        self.__input_err_term[2] += self.__cal_inner_2_2_3_err_term(tmp_cal, w_list, 2,    1,    None, None)
        self.__input_err_term[3] += self.__cal_inner_2_2_3_err_term(tmp_cal, w_list, None, 2,    0,    None)
        self.__input_err_term[4] += self.__cal_inner_2_2_3_err_term(tmp_cal, w_list, None, None, 1,    0)
        self.__input_err_term[5] += self.__cal_inner_2_2_3_err_term(tmp_cal, w_list, None, None, 2,    1)

    def cal_1_1_1_err_term(self, tmp_cal):

        w_list = ConOp.rot90(self.__input_filter_1_1_1.w_list[0], 2)

        for i in range(len(self.__input_err_term)):
            self.__input_err_term[i] += ConOp.convolution(tmp_cal[15], w_list[i], 1, 0)

    def cal_input_err_term(self):
        tmp_cal = ConOp.add_zero_circle(self.err_term, self.__input_filter_3_6.w_list[0].shape[1] - 1)
        tmp_cal = ConOp.add_row_col_by_step(tmp_cal, self.__step)

        self.__input_err_term = np.zeros((6, 14, 14), dtype=float)
        self.cal_3_6_err_term(tmp_cal)
        self.cal_4_6_err_term(tmp_cal)
        self.cal_2_2_3_err_term(tmp_cal)
        self.cal_1_1_1_err_term(tmp_cal)


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


def readfile():
    with open('/Users/shenjiafeng/ai/data/mnist/train-images.idx3-ubyte', 'rb') as f1:
        buf1 = f1.read()
    with open('/Users/shenjiafeng/ai/data/mnist/train-labels.idx1-ubyte', 'rb') as f2:
        buf2 = f2.read()
    return buf1, buf2


def get_image(buf1):
    image_index = 0
    image_index += struct.calcsize('>IIII')
    im = []
    for i in range(10000):
        temp = struct.unpack_from('>784B', buf1, image_index)  # '>784B'的意思就是用大端法读取784个unsigned byte
        im.append(np.reshape(temp, 784))
        image_index += struct.calcsize('>784B')  # 每次增加784B
    return im


def get_label(buf2):  # 得到标签数据
    label_index = 0
    label_index += struct.calcsize('>II')
    return struct.unpack_from('>10000B', buf2, label_index)


def cal_err(t, y):
    ret = 0
    for i in range(len(t)):
        ret += (t[i] - y[i]) * (t[i] - y[i]) * 0.5
    return ret


def test_train1():
    image_data, label_data = readfile()
    im = get_image(image_data)
    label = get_label(label_data)

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

    net = Net(120, [84], 10)

    for i in range(1000):
        a.set_out_val(ConOp.add_zero_circle(im[i].reshape((1, 28, 28)), 2))
        label_v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        label_v[label[i]] = 1

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

        con_out = f.out_val
        net_input = con_out.reshape(120)
        net.cal_out(net_input)

        net.cal_err_term(label_v)
        net.cal_err_out()

        err_out_val = np.array(net.err_out).reshape((120, 1, 1))
        f.set_err_term(err_out_val)
        e.cal_err_out()
        e.cal_err_term()
        d.get_err_out_from_pooling()
        d.cal_err_term()
        c.cal_err_out()
        c.cal_err_term()
        b.get_err_out_from_pooling()
        b.cal_err_term()

        w1.cal_dw()

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

                        con_out = f.out_val
                        net_input = con_out.reshape(120)
                        out1 = net.cal_out(net_input)
                        err1 = cal_err(out1, label_v)

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

                        con_out = f.out_val
                        net_input = con_out.reshape(120)
                        out2 = net.cal_out(net_input)
                        err2 = cal_err(out2, label_v)

                        w_list[i][index_deep][r][co] += es
                        w1.set_w(w_list)

                        print((err1 - err2) / (2 * es))
                        print(dw[i][index_deep][r][co])

    for i in range(10):
        label_v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        label_v[label[i]] = 1
        print(label[i])
        print(net.cal_out(im[i]))


def test_train():
    image_data, label_data = readfile()
    im = get_image(image_data)
    label = get_label(label_data)

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

    net = Net(120, [84], 10)

    for i in range(10):
        print(i)

        a.set_out_val(ConOp.add_zero_circle(im[i].reshape((1, 28, 28)), 2))
        label_v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        label_v[label[i]] = 1

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

        con_out = f.out_val
        net_input = con_out.reshape(120)
        net.cal_out(net_input)

        net.cal_err_term(label_v)
        net.cal_err_out()

        err_out_val = np.array(net.err_out).reshape((120, 1, 1))
        f.set_err_term(err_out_val)
        e.cal_err_out()
        e.cal_err_term()
        d.get_err_out_from_pooling()
        d.cal_err_term()
        c.cal_err_out()
        c.cal_err_term()
        b.get_err_out_from_pooling()
        b.cal_err_term()

        w1.cal_dw()
        w2.cal_dw()
        e.cal_dw()
        c.cal_dw()
        net.cal_dw()

        w1.update_w(0.01)
        w2.update_w(0.01)
        e.update_w(0.01)
        c.update_w(0.01)
        net.update_w(0.01)

    for i in range(10):
        a.set_out_val(ConOp.add_zero_circle(im[i].reshape((1, 28, 28)), 2))
        label_v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        label_v[label[i]] = 1

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

        con_out = f.out_val
        net_input = con_out.reshape(120)
        print(net.cal_out(net_input))


if __name__ == '__main__':
    test2()
