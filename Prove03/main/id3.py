from classifier import Classifier
from dataset import Dataset

import numpy as np


class ID3(Classifier):
    tree = {}

    def __init__(self):
        super().__init__()

    def train(self, dataset):
        self.training_set = dataset
        self.tree = self.make_tree(self.training_set)

    def predict(self, dataset):
        predictions = []

        feature = ""

        print("TREE: " + str(self.tree))

        for feature_name in dataset.feature_names:
            if feature_name in self.tree:
                feature = feature_name

        for datapoint in dataset.data:
            predictions.append(self.compare_to_feature(datapoint, self.tree[feature], dataset.feature_names, feature))

        return predictions

    def compare_to_feature(self, datapoint, feature, feature_names, feature_name):
        prediction = 0
        for key, value in feature.items():
            if key == datapoint[feature_names.index(feature_name)]:
                if isinstance(value, dict):
                    new_feature_name = ""

                    for temp_feature_name in feature_names:
                        if temp_feature_name in value:
                            new_feature_name = temp_feature_name

                    prediction = self.compare_to_feature(datapoint, value[new_feature_name], feature_names, new_feature_name)

                else:
                    return value

        return prediction

    def make_tree(self, dataset):
        nData = len(dataset.data)
        nFeatures = len(dataset.feature_names)

        default = dataset.target[np.argmax(dataset.target)]

        # There is no more features nor data to work with
        if nData == 0 or nFeatures == 0:
            # Return the target values that are the most common
            return default
        # There is only one target/class left
        elif dataset.target.count(dataset.target[0]) == nData:
            # Return that target/class
            return dataset.target[0]
        # Time to create a sub part of the tree and not a leaf!
        else:
            # Using information gain, figure out which feature/attribute is best
            gain = np.zeros(nFeatures)

            # Go through the available features to figure out the best feature!
            for feature in range(nFeatures):
                # Calculate the information gain of each feature
                gain[feature] = self.calc_info_gain(dataset, feature)

            # Determine the best feature by highest
            bestFeature = np.argmax(gain)

            # Create a tree by adding the feature name of the best feature to it
            tree = { dataset.feature_names[bestFeature]: {} }

            # Retrieve all the available values of the best feature from the data.
            values = []
            for datapoint in dataset.data:
                if datapoint[bestFeature] not in values and datapoint[bestFeature] != -1:
                    values.append(datapoint[bestFeature])

            # Loop through those available values
            for value in values:
                index = 0
                newNames = []
                newClasses = []
                newData = []

                # Loop through each datapoint to remove the feature from our list.
                for datapoint in dataset.data:
                    if datapoint[bestFeature] == value:
                        new_datapoint = []
                        if bestFeature == 0:
                            new_datapoint = datapoint[1:]
                            newNames = dataset.feature_names[1:]
                        elif bestFeature == nFeatures:
                            new_datapoint = datapoint[:-1]
                            newNames = dataset.feature_names[:-1]
                        else:
                            new_datapoint = datapoint[:bestFeature]
                            new_datapoint.extend(datapoint[bestFeature+1:])
                            newNames = dataset.feature_names[:bestFeature]
                            newNames.extend(dataset.feature_names[bestFeature+1:])
                        newData.append(new_datapoint)
                        newClasses.append(dataset.target[index])
                    index += 1

                # Let's go down this part of the tree and create a sub tree.
                subtree = self.make_tree(Dataset(newData, newClasses, dataset.target_names, newNames))

                # Add this subtree to our list!
                tree[dataset.feature_names[bestFeature]][value] = subtree
            # Whew, we've made it!
            return tree

    def calc_info_gain(self, dataset, feature):
        # Initialize gain
        gain = 0

        # Get the number of data in our set
        nData = len(dataset.data)

        # Get all the values the feature can be.
        values = []
        for datapoint in dataset.data:
            if datapoint[feature] not in values and datapoint[feature] != -1:
                values.append(datapoint[feature])

        # Get an array of the size of the values
        featureCounts = np.zeros(len(values))

        # Get an array of the size of the values to represent the entropy.
        entropy = np.zeros(len(values))
        valueIndex = 0

        # Loop through each value that a feature can be.
        for value in values:
            dataIndex = 0

            # Keep track of how many times a value is found in the dataset along with its corresponding class.
            newClasses = []
            for datapoint in dataset.data:
                if datapoint[feature] == value:
                    featureCounts[valueIndex] += 1
                    newClasses.append(dataset.target[dataIndex])
                dataIndex += 1

            # Get unique values essentially of the classes.
            classValues = []
            for class_item in newClasses:
                if classValues.count(class_item) == 0:
                    classValues.append(class_item)

            # Create an empty array that is the length of how many classes there are.
            classCounts = np.zeros(len(classValues))
            classIndex = 0

            # Count how many of each class there are in the newClasses array.
            for classValue in classValues:
                for class_item in newClasses:
                    if class_item == classValue:
                        classCounts[classIndex] += 1
                classIndex += 1

            # Calculate the entropy of the value summing up the entropys of the frequencies of the classes.
            # This is calculated by passing in the number of times a class was there over the total number of classes.
            for classIndex in range(len(classValues)):
                entropy[valueIndex] += self.calc_entropy(float(classCounts[classIndex])/sum(classCounts))

            # Finally to calculate the gain, we need to divide the number of times the value was found in the dataset
            # by the number of data over all. Then multiply that by the value's entropy.
            gain += float(featureCounts[valueIndex])/nData * entropy[valueIndex]
            valueIndex += 1

        return gain

    def calc_entropy(self, probability):
        if probability != 0:
            return -probability * np.log2(probability)
        else:
            return 0

    def display_tree(self, level, tree):
        for key, value in tree.items():
            tab_str = ""
            for index in range(level):
                tab_str += "\t"
            print(tab_str + str(key),end='')

            if isinstance(value, dict):
                print()
                self.display_tree(level+1, value)
            else:
                print(": " + str(self.training_set.target_names[value]))