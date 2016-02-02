from sklearn import datasets
from sklearn import tree

from dataset import Dataset
from id3 import ID3
from sklearn.preprocessing import normalize

import math


def split_dataset(data_set, split_percentage):
    """
    Creates a training set and a testing set based on the data set given. Splits the sets by 70/30 respectively
    :param data_set: The dataset that will be split
    :return: Returns a tuple holding the training and testing sets
    """
    length = data_set.data.__len__()
    top_training_index = math.floor(length * split_percentage)
    training_set_data = []
    training_set_target = []
    testing_set_data = []
    testing_set_target  = []

    for index in range(0, top_training_index):
        training_set_data.append([item for item in data_set.data[index]])
        training_set_target.append(data_set.target[index])

    for index in range(top_training_index, length):
        testing_set_data.append([item for item in data_set.data[index]])
        testing_set_target.append(data_set.target[index])

    training_set_dataset = Dataset(training_set_data, training_set_target, data_set.target_names, data_set.feature_names)
    testing_set_dataset = Dataset(testing_set_data, testing_set_target, data_set.target_names, data_set.feature_names)

    return { 'train': training_set_dataset, 'test': testing_set_dataset }


def get_accuracy(predictions, actuals):
    """
    Trains the classifier with the given training set and predicts using the testing set.
    :param data_sets: A tuple that holds the training set in spot 0 and the testing set in spot 1
    :return: return the accuracy as a percentage
    """
    count = 0

    for index in range(actuals.__len__()):
        if int(predictions[index]) is int(actuals[index]):
            count += 1

    return round((count / actuals.__len__()) * 100, 2)


def write_to_results_file(file, data):
    with open(file, mode='a') as results:
        results.write(data)


def main():
    while True:
        data_set_name = input("Please provide the name of the data set you want to work with: ")

        # Load, Randomize, Normalize, Discretize Dataset
        data_set = Dataset()
        data_set.read_file_into_dataset("C:\\Users\\Grant\\Documents\\School\\Winter 2016\\CS 450\\Prove03\\" + data_set_name)
        data_set.randomize()
        data_set.data = normalize(data_set.data)
        data_set.discretize()

        data_set.set_missing_data()
        # Split Dataset
        split_percentage = 0.7
        data_sets    = split_dataset(data_set, split_percentage)
        training_set = data_sets['train']
        testing_set  = data_sets['test']

        # Create Custom Classifier, Train Dataset, Predict Target From Testing Set
        id3Classifier = ID3()
        id3Classifier.train(training_set)
        predictions = id3Classifier.predict(testing_set)

        id3Classifier.display_tree(0, id3Classifier.tree)
        # Check Results
        my_accuracy = get_accuracy(predictions, testing_set.target)
        print("Accuracy: " + str(my_accuracy) + "%")

        # Compare To Existing Implementations
        dtc = tree.DecisionTreeClassifier()
        dtc.fit(training_set.data, training_set.target)
        predictions = dtc.predict(testing_set.data)

        dtc_accuracy = get_accuracy(predictions, testing_set.target)
        print("DTC Accuracy: " + str(dtc_accuracy) + "%")

        # Do another or not
        toContinue = False

        while True:
            another = input("Do you want to examine another dataset? (y / n) ")

            if another != 'y' and another != 'n':
                print("Please provide you answer in a 'y' or 'n' format.")
            elif another == 'y':
                toContinue = True
                break
            else:
                toContinue = False
                break

        if not toContinue:
            break

# Produce a textual view of your resulting tree ASK

if __name__ == '__main__':
    main()
