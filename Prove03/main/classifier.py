from dataset import Dataset

class Classifier(object):
    """
    A classifier that can be trained to understand a certain dataset and can predict new data given to it.
    """
    training_set = None

    def __init__(self):
        pass

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
