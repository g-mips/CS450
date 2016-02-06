from classifier import Classifier
from node import Node


class NeuralNetwork(Classifier):
    nodes = []

    def __init__(self):
        super().__init__()

    def train(self, dataset):
        self.training_set = dataset

        for i in range(self.training_set.data.__len__()):
            self.__set_nodes_on_layer(0, self.training_set.data[i])
            for j in range(self.nodes.__len__()-1):
                self.__calculate_nodes_on_layer(j)
                self.__set_nodes_on_layer(j+1, self.__get_outputs_on_layer(j))
            self.__calculate_nodes_on_layer(self.nodes.__len__()-1)
            for k in range(self.nodes[self.nodes.__len__()-1].__len__()):
                print("OUTPUTS: ", end='')
                print(self.nodes[self.nodes.__len__()-1][k].output)
                print("FEATURE: ", end='')
                print(self.training_set.target[i])


    def predict(self, dataset):
        return dataset.target

    def create_layered_network(self, num_nodes_on_layers: list):
        for i in range(num_nodes_on_layers.__len__()):
            self.__create_layer(num_nodes_on_layers[i])
        print("NODES: ", end='')
        print(self.nodes)

    def __get_outputs_on_layer(self, layer):
        outputs = []
        for i in range(self.nodes[layer].__len__()):
            outputs.append(self.nodes[layer][i].output)
        return outputs

    def __set_nodes_on_layer(self, layer, inputs):
        for i in range(self.nodes[layer].__len__()):
            self.nodes[layer][i].set_inputs(inputs)
            self.nodes[layer][i].set_weights()
            print("NODE: ", end='')
            print(self.nodes[layer][i].inputs, end='')
            print(" ", end='')
            print(self.nodes[layer][i].weights)

    def __create_layer(self, num_nodes):
        layer = []
        for i in range(num_nodes):
            layer.append(Node())
        self.nodes.append(layer)

    def __calculate_nodes_on_layer(self, layer):
        for i in range(self.nodes[layer].__len__()):
            total = 0
            for j in range(self.nodes[layer][i].inputs.__len__()):
                total += self.nodes[layer][i].inputs[j]*self.nodes[layer][i].weights[j]
            self.nodes[layer][i].activation_function(total)
