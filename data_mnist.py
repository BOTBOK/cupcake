import struct

import numpy as np


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