from sklearn import datasets
import random
import math
from dataset import Dataset
from classifier import Classifier

def randomize_dataset(iris):
    '''
    Shuffles the data and it's targets and puts them in a new dataset
    :param iris: The dataset that will be randomized
    :return: A new dataset that contains the newly ordered data and target lists
    '''
    data_list = []
    target_list = []
    index_list = list(range(0, 150))
    random.shuffle(index_list)

    for index in range(index_list.__len__()):
        data_list.append(iris.data[index_list[index]])
        target_list.append(iris.target[index_list[index]])

    return Dataset(data_list, target_list, iris.target_names)

def get_sets(data_set):
    '''
    Creates a training set and a testing set based on the data set given. Splits the sets by 70/30 respectively
    :param data_set: The dataset that will be split
    :return: Returns a tuple holding the training and testing sets
    '''
    length = data_set.data.__len__()
    top_training_index = math.floor(length * 0.7)
    training_set_data = []
    training_set_target = []
    testing_set_data = []
    testing_set_target  = []

    for index in range(0, top_training_index):
        training_set_data.append(data_set.data[index])
        training_set_target.append(data_set.target[index])

    for index in range(top_training_index, length):
        testing_set_data.append(data_set.data[index])
        testing_set_target.append(data_set.target[index])

    training_set_dataset = Dataset(training_set_data, training_set_target, data_set.target_names)
    testing_set_dataset = Dataset(testing_set_data, testing_set_target, data_set.target_names)

    return (training_set_dataset, testing_set_dataset)

def set_up_datasets():
    '''
    Gets the iris dataset, randomizes it, and produces a training and testing set from it
    :return: a tuple that holds the training and testing sets
    '''
    iris = datasets.load_iris()
    data_set = randomize_dataset(iris)
    data_sets = get_sets(data_set)

    return data_sets

def get_accuracy(data_sets: tuple):
    '''
    Trains the classifier with the given training set and predicts using the testing set.
    :param data_sets: A tuple that holds the training set in spot 0 and the testing set in spot 1
    :return: return the accuracy as a percentage
    '''
    new_classifier = Classifier()
    new_classifier.train(data_sets[0])

    count = 0

    for index in range(data_sets[1].data.__len__()):
        prediction = new_classifier.predict(data_sets[1].data[index])

        if prediction == data_sets[1].target[index]:
            count += 1

    return round((count / data_sets[1].data.__len__()) * 100, 2)

data_sets = set_up_datasets()
accuracy  = get_accuracy(data_sets)

print("Accuracy of " + str(accuracy) + "%")
