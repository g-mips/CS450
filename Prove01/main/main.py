from sklearn import datasets
from dataset import Dataset
from knn_classifier import KnnClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

import random
import math
import re
import os


def normalize_data():
    # Normalize numeric data
    pass


def read_file_into_dataset(data_set_name):
    target_info = []
    target_names = []
    target_names_dict = {}
    target = []
    data = []

    data_file = data_set_name + ".data"
    name_file = data_set_name + ".names"

    # Be able to load datasets from text files
    line_number = 1
    with open(name_file) as name_info:
        for line in name_info:
            pattern = re.compile(r'\s+')
            line = re.sub(pattern, '', line)

            if line is not '' and line[0] is not '|':
                if line_number is 1:
                    target_names = line.split(',')
                    for index in range(target_names.__len__()):
                        target_names_dict[target_names[index]] = index
                    line_number += 1
                else:
                    line = line.split(':')[1]
                    line = line.replace('.', '').split(',')

                    line_dict = {}
                    integer = 0

                    for index in range(line.__len__()):
                        if line[index].isnumeric():
                            line_dict[line[index]] = int(line[index])
                            integer = int(line[index]) + 1
                        else:
                            line_dict[line[index]] = integer
                            integer += 1

                    target_info.append(line_dict)

    with open(data_file) as data_info:
        for line in data_info:
            pattern = re.compile(r'\s+')
            line = re.sub(pattern, '', line)

            single_data = line.split(',')

            for index in range(single_data.__len__() - 1):
                single_data[index] = target_info[index][single_data[index]]

            target.append(target_names_dict[single_data.pop()])
            data.append(single_data)

    return Dataset(data, target, target_names)


def randomize_dataset(data):
    """
    Shuffles the data and it's targets and puts them in a new dataset
    :param iris: The dataset that will be randomized
    :return: A new dataset that contains the newly ordered data and target lists
    """
    data_list = []
    target_list = []
    index_list = list(range(0, data.data.__len__()))
    random.shuffle(index_list)

    for index in range(index_list.__len__()):
        data_list.append(data.data[index_list[index]])
        target_list.append(data.target[index_list[index]])

    return Dataset(data_list, target_list, data.target_names)


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

    training_set_dataset = Dataset(training_set_data, training_set_target, data_set.target_names)
    testing_set_dataset = Dataset(testing_set_data, testing_set_target, data_set.target_names)

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


def write_to_results_file(file, data, k):
    with open(file, mode='a') as results:
        results.write(data)


def main(k, data_set_name=None):
    split_percentage = 0.7

    # Load dataset
    if data_set_name is not None:
        data_set = read_file_into_dataset("C:\\Users\\Grant\\Documents\\School\\Winter 2016\\CS 450\\Prove01\\" + data_set_name)
        data_set = randomize_dataset(data_set)
    else:
        data_set_name = "iris"
        iris = datasets.load_iris()
        data_set = randomize_dataset(iris)

    data_set.data = normalize(data_set.data)
    data_sets    = split_dataset(data_set, split_percentage)
    training_set = data_sets['train']
    testing_set  = data_sets['test']

    # My Classifier
    knnClassifier = KnnClassifier()
    knnClassifier.k = k
    knnClassifier.train(training_set.data, training_set.target, training_set.target_names)
    predictions = knnClassifier.predict(testing_set.data)

    my_accuracy = get_accuracy(predictions, testing_set.target)

    # Better Classifier
    better_classifier = KNeighborsClassifier(n_neighbors=k)
    better_classifier.fit(training_set.data, training_set.target)
    predictions = better_classifier.predict(testing_set.data)

    better_accuary = get_accuracy(predictions, testing_set.target)

    print("My results: " + str(my_accuracy) + "%")
    print("Better results: " + str(better_accuary) + "%")

    results = "k = " + str(k) + "\nMy results: " + str(my_accuracy) + "%\n" + "Better results: " + str(better_accuary) + "%\n"

    write_to_results_file(os.getcwd() + os.sep + ".." + os.sep + str(k) + "-" + data_set_name + "_results.txt", results, k)


for k in range(1, 106):
    main(k, "car")
    main(k)
