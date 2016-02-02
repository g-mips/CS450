import numpy as np

from dataset import Dataset
from classifier import Classifier


class KnnClassifier(Classifier):
    k = 3

    def __init__(self):
        super().__init__()

    def predict(self, instances):
        """
        Predicts the classes of the instances given by using the knn algorithm.
        :param instances: data that is being tested
        :return: predictions of the instances
        """
        # Gives me the shape of the instances
        data_shape = np.shape(instances)[0]

        # Creates an array of zeros with data_shape number
        closest = np.zeros(data_shape)

        # Determine a useful k for these instances
        k = self.determine_k(instances)

        # Loop through the testing data
        for n in range(data_shape):
            # Get the distances between the testing data and training data.
            distances = np.sum(list((np.array(self.training_set.data) - np.array(instances[n]))**2), axis=1)
            indices = np.argsort(distances, axis=0)
            closest_training = []

            # Create a list that contains only up to k of the closest targets in the training set
            for i in range(k):
                closest_training.append(self.training_set.target[indices[i]])

            classes = np.unique(closest_training)

            # Determine the class for the nth item.
            if len(classes) is 1:
                closest[n] = np.unique(classes)
            else:
                # Having a dictionary gets rid of out of bounds errors.
                counts = dict.fromkeys(classes, 0)
                for i in range(k):
                    counts[self.training_set.target[indices[i]]] += 1

                list_values = list(counts.values())

                # Tie-breaker is simply the first one. I know it's bad but it's easy.
                closest[n] = list_values.index(max(list_values))

        return closest

    def determine_k(self, instances):
        """
        Determines a useful k based on these instances. For now return a number.
        :param instances:
        :return: k
        """
        return self.k
