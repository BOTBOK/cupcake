import numpy as np

from cnn2.cnn_main import CnnBlock


class CnnListStruct(object):
    def __init__(self, layper_param, num_class, reg, regulation, activation):
        self.__layer_param_list = layper_param
        self.__layer_param_list.append(('', 'last_FC'))
        self.__layer_param_list.append(('', 'softmax'))
        self.__map_shape_list = []
        self.__weights = []
        self.__biases = []

        self.__prob = []

        self.__matric_data_list = []
        self.__filter_data_list = []
        self.__max_matric_data_pox_list = []

        self.__dweights = []
        self.__dbiases = []

        self.__num_class = num_class
        self.__cnn_block = CnnBlock()
        self.__activation = activation
        self.__regulation = regulation
        self.__reg = reg

        print(self.__layer_param_list)

    @property
    def map_shape(self):
        return self.__map_shape_list

    @property
    def weights(self):
        return self.__weights

    @property
    def biases(self):
        return self.__biases

    @property
    def dweights(self):
        return self.__dweights

    @property
    def dbiases(self):
        return self.__dbiases

    def feature_maps(self, in_data):
        batch, in_height, in_width, in_depth = in_data.shape
        self.__map_shape_list.append((in_height, in_width, in_depth))
        for layer_param in self.__layer_param_list:
            if layer_param[-1] == 'conv':
                out_height = (in_height - layer_param[1] + 2 * layer_param[3]) // layer_param[2] + 1
                out_width = (in_width - layer_param[1] + 2 * layer_param[3]) // layer_param[2] + 1
                out_depth = layer_param[0]
                self.__map_shape_list.append((out_height, out_width, out_depth))
            elif layer_param[-1] == 'pool':
                out_height = (in_height - 2) // 2 + 1
                out_width = (in_width - 2) // 2 + 1
                out_depth = layer_param[0]
                self.__map_shape_list.append((out_height, out_width, out_depth))
            elif layer_param[-1] == 'FC':
                out_height = 1
                out_width = 1
                out_depth = layer_param[0]
                self.__map_shape_list.append((out_height, out_width, out_depth))
            elif layer_param[-1] == 'last_FC':
                out_height = 1
                out_width = 1
                out_depth = self.__num_class
                self.__map_shape_list.append((out_height, out_width, out_depth))
            elif layer_param[-1] == 'softmax':
                out_height = 1
                out_width = 1
                pass
            else:
                print(layer_param[-1])
                raise KeyError('''struct中 层标识符错误 ''')
            in_height = out_height
            in_width = out_width
        print(self.__map_shape_list)

    def init_param(self):
        for layer_param, map_shape in zip(self.__layer_param_list, self.__map_shape_list):
            if layer_param[-1] == 'conv':
                weights, biases = self.__cnn_block.param_init(layer_param[0], map_shape[2],
                                                              layer_param[1] * layer_param[1])
                self.__weights.append(weights)
                self.__biases.append(biases)
            elif layer_param[-1] == 'pool':
                weights = np.array([])
                biases = np.array([])
                self.__weights.append(weights)
                self.__biases.append(biases)
            elif layer_param[-1] == 'FC':
                weights, biases = self.__cnn_block.param_init(layer_param[0], map_shape[2],
                                                              map_shape[0] * map_shape[1])
                self.__weights.append(weights)
                self.__biases.append(biases)
            elif layer_param[-1] == 'last_FC':
                weights, biases = self.__cnn_block.param_init(self.__num_class, map_shape[2],
                                                              map_shape[0] * map_shape[1])
                self.__weights.append(weights)
                self.__biases.append(biases)
            elif layer_param[-1] == 'softmax':
                weights = np.array([])
                biases = np.array([])
                self.__weights.append(weights)
                self.__biases.append(biases)
            else:
                raise KeyError('''struct中 层标识符错误 ''')

    def forward(self, in_data):
        self.__matric_data_list = []
        self.__filter_data_list = []
        self.__max_matric_data_pox_list = []

        data = in_data
        for i in range(len(self.__layer_param_list)):
            matric_data = []
            filter_data = []
            max_matric_data_pox = []
            if self.__layer_param_list[i][-1] == 'conv':
                out_data, matric_data, filter_data = self.__cnn_block.conv_layer(data, self.__weights[i],
                                                                                 self.__biases[i],
                                                                                 self.__layer_param_list[i][0:-1],
                                                                                 self.__activation)
            elif self.__layer_param_list[i][-1] == 'pool':
                out_data, max_matric_data_pox = self.__cnn_block.pool_layer(data)
            elif self.__layer_param_list[i][-1] == 'FC':
                out_data, matric_data, filter_data = self.__cnn_block.FC_layer(data, self.__weights[i],
                                                                               self.__biases[i],
                                                                               self.__layer_param_list[i][0], False,
                                                                               self.__activation)
            elif self.__layer_param_list[i][-1] == 'last_FC':
                out_data, matric_data, filter_data = self.__cnn_block.FC_layer(data, self.__weights[i],
                                                                               self.__biases[i],
                                                                               self.__num_class, True,
                                                                               self.__activation)
            elif self.__layer_param_list[i][-1] == 'softmax':
                out_data = self.__cnn_block.softmax_layer(data)
                self.__prob = out_data
            else:
                raise KeyError('''struct中 层标识符错误 ''')
            self.__matric_data_list.append(matric_data)
            self.__filter_data_list.append(filter_data)
            self.__max_matric_data_pox_list.append(max_matric_data_pox)
            data = out_data

    def data_loss(self, labels):
        return self.__cnn_block.data_loss(self.__prob, labels)

    def reg_loss(self):
        reg_loss = 0
        for weight in self.__weights:
            if weight.size != 0:
                reg_loss += np.sum(self.__regulation.dreg(weight, self.__reg))
        return reg_loss

    def dweight_reg(self):
        for i in range(len(self.__weights)):
            if self.__weights[i].size >= 0:
                self.__dweights[-1 - i] += self.__regulation.dreg(self.__weights[i], self.__reg)

    def backward(self, labels):
        self.__dweights = []
        self.__dbiases = []

        dout_data = []
        for layer_param, map_shape, matric_data, filter_data, weight, matric_max_data_pox in zip(
                reversed(self.__layer_param_list),
                reversed(self.__map_shape_list),
                reversed(self.__matric_data_list),
                reversed(self.__filter_data_list),
                reversed(self.__weights),
                reversed(self.__max_matric_data_pox_list)):
            dbiase = []
            dweight = []
            if layer_param[-1] == 'softmax':
                din_data = self.__cnn_block.evaluate_dscores(self.__prob, labels)
            elif layer_param[-1] == 'last_FC':
                din_data, dweight, dbiase = self.__cnn_block.dFC_layer(dout_data, matric_data, filter_data, weight,
                                                                       map_shape, True, self.__activation)
            elif layer_param[-1] == 'FC':
                din_data, dweight, dbiase = self.__cnn_block.dFC_layer(dout_data, matric_data, filter_data, weight,
                                                                       map_shape, False, self.__activation)
            elif layer_param[-1] == 'pool':
                din_data = self.__cnn_block.dpooling_layer(dout_data, map_shape, matric_max_data_pox)
            elif layer_param[-1] == 'conv':
                din_data, dweight, dbiase = self.__cnn_block.dconv_layer(dout_data, matric_data, weight, filter_data,
                                                                         layer_param[0:-1], map_shape,
                                                                         self.__activation)
            else:
                raise KeyError('''struct中 层标识符错误 ''')
            self.__dweights.append(dweight)
            self.__dbiases.append(dbiase)
            dout_data = din_data
        self.dweight_reg()
