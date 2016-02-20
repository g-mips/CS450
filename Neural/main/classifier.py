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


class Classifier(object):
    """
    A classifier that can be trained to understand a certain dataset and can predict new data given to it.
    """

    def __init__(self):
        self.training_set   = None
        self.testing_set    = None
        self.validation_set = None

    def train(self, dataset):
        """
        Trains the classifier to be knowledgeable of this type of data.
        :param data: data set to train on
        :param target: target class
        :param target_names: target names
        """
        if self.training_set is None:
            self.training_set = dataset
        else:
            self.training_set.add_to_data(dataset.data, dataset.target, dataset.target_names)

    def predict(self, dataset):
        return 0
