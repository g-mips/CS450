import numpy as np


class Node(object):
    BIAS_X = 1
    inputs = []
    output = 0
    weights = []
    threshold = 0

    def __init__(self):
        pass

    def set_inputs(self, inputs: list):
        self.inputs = [ self.BIAS_X ]
        self.inputs.extend(inputs)

    def set_weights(self):
        self.weights = np.random.ranf(self.inputs.__len__()) - 0.5

    def activation_function(self, total):
        if total > self.threshold:
            output = 1
        else:
            output = 0
