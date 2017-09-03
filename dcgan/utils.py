import tensorflow as tf
import numpy as np
import math

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size)/ float(stride)))