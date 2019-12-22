import numpy as np

import letnet5 as n5
import data_mnist as mnist_data


class LetNet5(object):
    def __init__(self):
        self.c1_map = n5.FeatureMap(32, 1, 1)
        self.c1_filter = n5.FilterLayer(5, 1, 6, 1)
        self.c1_out_s2_in_map = n5.FeatureMap(28, 6, 1)
        self.s2_out_c3_in_map = n5.FeatureMap(14, 6, 1)
        self.c3_3_6_filter = n5.FilterLayer(5, 3, 6, 1)
        self.c3_4_6_filter = n5.FilterLayer(5, 4, 6, 1)
        self.c3_4_3_filter = n5.FilterLayer(5, 4, 3, 1)
        self.c3_1_1_filter = n5.FilterLayer(5, 6, 1, 1)
        self.c3_out_s4_in_map = n5.C3FeatureMap(10, 16, 1)
        self.s4_out_c5_in_map = n5.FeatureMap(5, 16, 1)
        self.c5_filter = n5.FilterLayer(5, 16, 120, 1)
        self.c5_out_map = n5.FeatureMap(1, 120, 1)

        LetNet5.connection_map(self.c1_map, self.c1_out_s2_in_map, self.c1_filter)
        LetNet5.connection_map(self.c1_out_s2_in_map, self.s2_out_c3_in_map)
        LetNet5.connection_c3_map(self.s2_out_c3_in_map, self.c3_out_s4_in_map, self.c3_3_6_filter, self.c3_4_6_filter,
                                  self.c3_4_3_filter, self.c3_1_1_filter)
        LetNet5.connection_map(self.c3_out_s4_in_map, self.s4_out_c5_in_map)
        LetNet5.connection_map(self.s4_out_c5_in_map, self.c5_out_map, self.c5_filter)

    @staticmethod
    def connection_map(a, b, w=None):
        a.set_output_feature_map(b)
        if w is not None:
            a.set_output_filter(w)
            w.set_input_feature_map(a)

            w.set_output_feature_map(b)
            b.set_input_filter(w)
        b.set_input_feature_map(a)

    @staticmethod
    def connection_c3_map(c3_in_map, c3_out_map, c3_3_6_filter, c3_4_6_filter, c3_4_3_filter, c3_1_1_filter):
        LetNet5.connection_map(c3_in_map, c3_out_map)

        c3_3_6_filter.set_input_feature_map(c3_in_map)
        c3_3_6_filter.set_output_feature_map(c3_out_map)

        c3_4_6_filter.set_input_feature_map(c3_in_map)
        c3_4_6_filter.set_output_feature_map(c3_out_map)

        c3_4_3_filter.set_input_feature_map(c3_in_map)
        c3_4_3_filter.set_output_feature_map(c3_out_map)

        c3_1_1_filter.set_input_feature_map(c3_in_map)
        c3_1_1_filter.set_output_feature_map(c3_out_map)

        c3_out_map.set_3_6_filter(c3_3_6_filter)
        c3_out_map.set_4_6_filter(c3_4_6_filter)
        c3_out_map.set_4_3_filter(c3_4_3_filter)
        c3_out_map.set_1_1_filter(c3_1_1_filter)

    def cal_out(self, input_val):
        self.c1_map.set_out_val(input_val)
        self.c1_out_s2_in_map.cal_net_out()
        self.c1_out_s2_in_map.cal_out_val()
        self.s2_out_c3_in_map.pooling_cal_net_val()
        self.s2_out_c3_in_map.pooling_cal_out_val()
        self.c3_out_s4_in_map.cal_out_val()
        self.s4_out_c5_in_map.pooling_cal_net_val()
        self.s4_out_c5_in_map.pooling_cal_out_val()
        self.c5_out_map.cal_net_out()
        self.c5_out_map.cal_out_val()

    @property
    def out_val(self):
        return self.c5_out_map.out_val

    def cal_err_term(self, end_err_term):
        self.c5_out_map.set_err_term(end_err_term)
        self.s4_out_c5_in_map.cal_err_out()
        self.s4_out_c5_in_map.cal_err_term()
        self.c3_out_s4_in_map.get_err_out_from_pooling()
        self.c3_out_s4_in_map.cal_err_term()

        # s2 层的误差项需要特殊计算
        self.c3_out_s4_in_map.cal_input_err_term()
        self.s2_out_c3_in_map.set_err_out(self.c3_out_s4_in_map.input_err_term)
        self.s2_out_c3_in_map.set_err_term(self.c3_out_s4_in_map.input_err_term)
        self.c1_out_s2_in_map.get_err_out_from_pooling()
        self.c1_out_s2_in_map.cal_err_term()

    def cale_dw(self):
        self.c1_filter.cal_dw()
        self.c3_3_6_filter.cal_3_6_dw()
        self.c3_4_6_filter.cal_4_6_dw()
        self.c3_4_3_filter.cal_2_2_dw()
        self.c3_1_1_filter.cal_1_1_dw()
        self.c5_filter.cal_dw()

        self.s2_out_c3_in_map.cal_dw()
        self.s4_out_c5_in_map.cal_dw()

    def update_w(self, learn_rate):
        self.c1_filter.update_w(learn_rate)
        self.c3_3_6_filter.update_w(learn_rate)
        self.c3_4_6_filter.update_w(learn_rate)
        self.c3_4_3_filter.update_w(learn_rate)
        self.c3_1_1_filter.update_w(learn_rate)
        self.c5_filter.update_w(learn_rate)

        self.s2_out_c3_in_map.update_w(learn_rate)
        self.s4_out_c5_in_map.update_w(learn_rate)

def test():
    image_data, label_data = mnist_data.readfile()
    im = mnist_data.get_image(image_data)
    label = mnist_data.get_label(label_data)

    let_net_5 = LetNet5()

    for i in range(10):
        print(i)

        input_val = n5.ConOp.add_zero_circle(im[i].reshape((1, 28, 28)), 2)

        label_v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        out_err = np.ones((120, 1, 1), dtype=float)

        label_v[label[i]] = 1
        let_net_5.cal_out(input_val)
        let_net_5.cal_err_term(out_err)
        let_net_5.cale_dw()

        err_term = lambda a: np.sum(a)

        w_list = let_net_5.c1_filter.w_list
        dw = let_net_5.c1_filter.dw
        for i in range(len(w_list)):
            deep, row, col = w_list[i].shape
            for index_deep in range(deep):
                for r in range(row):
                    for co in range(col):
                        es = 0.00001
                        w_list[i][index_deep][r][co] += es
                        let_net_5.c1_filter.set_w(w_list)

                        let_net_5.cal_out(input_val)
                        err1 = err_term(let_net_5.out_val)

                        w_list[i][index_deep][r][co] -= 2 * es
                        let_net_5.c1_filter.set_w(w_list)

                        let_net_5.cal_out(input_val)
                        err2 = err_term(let_net_5.out_val)

                        w_list[i][index_deep][r][co] += es
                        let_net_5.c1_filter.set_w(w_list)

                        print((err1 - err2) / (2 * es))
                        print(dw[i][index_deep][r][co])


if __name__ == '__main__':
    image_data, label_data = mnist_data.readfile()
    im = mnist_data.get_image(image_data)
    label = mnist_data.get_label(label_data)

    let_net_5 = LetNet5()

    for i in range(10):
        print(i)

        input_val = n5.ConOp.add_zero_circle(im[i].reshape((1, 28, 28)), 2)

        label_v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        out_err = np.ones((120, 1, 1), dtype=float)

        label_v[label[i]] = 1
        let_net_5.cal_out(input_val)
        let_net_5.cal_err_term(out_err)
        let_net_5.cale_dw()

        err_term = lambda a: np.sum(a)

        w_list = let_net_5.c3_out_s4_in_map.pooling_w
        dw = let_net_5.c3_out_s4_in_map.dw

        deep, row, col = w_list.shape
        for index_deep in range(deep):
            for r in range(row):
                for co in range(col):
                    es = 0.00001
                    w_list[index_deep][r][co] += es
                    let_net_5.c3_out_s4_in_map.set_pooling_w(w_list)

                    let_net_5.cal_out(input_val)
                    err1 = err_term(let_net_5.out_val)

                    w_list[index_deep][r][co] -= 2 * es
                    let_net_5.c3_out_s4_in_map.set_pooling_w(w_list)

                    let_net_5.cal_out(input_val)
                    err2 = err_term(let_net_5.out_val)

                    w_list[index_deep][r][co] += es
                    let_net_5.c3_out_s4_in_map.set_pooling_w(w_list)

                    print((err1 - err2) / (2 * es))
                    print(dw[index_deep][r][co])
