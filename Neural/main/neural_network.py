from classifier import Classifier
from node import Node


class NeuralNetwork(Classifier):
    nodes = []

    def __init__(self):
        super().__init__()

    def train(self, dataset):
        if self.training_set is None:
            self.create_layered_network([dataset.target_names.__len__()], dataset)

        for i in range(self.training_set.data.__len__()):
            #print("layer num= 0")
            self.__set_nodes_on_layer(0, self.training_set.data[i])
            for j in range(self.nodes.__len__()-1):
                #print("layer num= ", end='')
                #print(j+1)
                self.__calculate_nodes_on_layer(j)
                self.__set_nodes_on_layer(j+1, self.__get_outputs_on_layer(j))
            self.__calculate_nodes_on_layer(self.nodes.__len__()-1)
            self.__evaluate_outputs()

    def predict(self, dataset):
        predictions = []

        for i in range(dataset.data.__len__()):
            self.__set_nodes_on_layer(0, dataset.data[i])
            for j in range(self.nodes.__len__()-1):
                self.__calculate_nodes_on_layer(j)
                self.__set_nodes_on_layer(j+1, self.__get_outputs_on_layer(j))
            self.__calculate_nodes_on_layer(self.nodes.__len__()-1)
            predictions.append(self.__evaluate_outputs())

        print(predictions)
        return predictions

    def create_layered_network(self, num_nodes_on_layers: list, dataset):
        self.training_set = dataset

        for i in range(num_nodes_on_layers.__len__()):
            #print("layer num= ", end='')
            #print(i)
            self.__create_layer(num_nodes_on_layers[i])

    def __get_outputs_on_layer(self, layer):
        outputs = []
        for i in range(self.nodes[layer].__len__()):
            outputs.append(self.nodes[layer][i].output)
        return outputs

    def __set_nodes_on_layer(self, layer, inputs):
        for i in range(self.nodes[layer].__len__()):
            #print("\tnode_num= ", end='')
            #print(i)
            self.nodes[layer][i].set_inputs(inputs)
            self.nodes[layer][i].set_weights()

    def __create_layer(self, num_nodes):
        layer = []
        for i in range(num_nodes):
            #print("\tnode_num= ", end='')
            #print(i)
            layer.append(Node())
        self.nodes.append(layer)

    def __calculate_nodes_on_layer(self, layer):
        for i in range(self.nodes[layer].__len__()):
            total = 0
            for j in range(self.nodes[layer][i].inputs.__len__()):
                total += self.nodes[layer][i].inputs[j]*self.nodes[layer][i].weights[j]
            self.nodes[layer][i].activation_function(total)

    def __evaluate_outputs(self):
        prediction = 0
        current_best = -999999999
        last_layer = self.nodes.__len__() - 1

        for i in range(self.nodes[last_layer].__len__()):
            if self.nodes[last_layer][i].output > current_best:
                prediction = i
                current_best = self.nodes[last_layer][i].output

        return prediction
