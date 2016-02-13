import numpy as np
import math


class Node(object):
    BIAS_X = 1
    inputs = []
    output = 0
    weights = []
    threshold = 0
    error = 0

    def __init__(self):
        pass

    def set_inputs(self, inputs: list):
        self.inputs = [ self.BIAS_X ]
        self.inputs.extend(inputs)

    def set_weights(self):
        self.weights = np.random.ranf(self.inputs.__len__()) - 0.5

    def activation_function(self, total):
        self.output = 1 / (1 + math.e**(-1*total))

