import numpy as np
import os

from sklearn.preprocessing import normalize

first = [
    [1.7, 4.9, 2.1],
    [1.3, 5.0, 3.3],
    [4.4, 5.4, 1.1],
    [9.0, 2.0, 5.0],
    [4.1, 3.2, 0.4]
]

second = [
    [1.2, 3.6, 2.8],
    [2.6, 4.6, 1.1],
    [3.1, 2.5, 3.4],
    [2.2, 2.6, 2.1],
    [9.6, 3.8, 5.2],
    [2.1, 1.6, 8.1],
    [2.5, 2.2, 3.6],
    [3.6, 2.8, 2.1]
]

third = [
    1, 0, 1, 1, 0, 1, 1, 0
]

print(first)
print(normalize(first))
first_shape = np.shape(first)[0]
closest = np.zeros(first_shape)
k = 3

for n in range(first_shape):
    distances = np.sum(list((np.array(second) - np.array(first[n]))**2), axis=1)
    indices = np.argsort(distances, axis=0)
    fourth = []

    for i in range(k):
        fourth.append(third[indices[i]])

    classes = np.unique(fourth)

    if len(classes) is 1:
        closest[n] = np.unique(classes)
    else:
        counts = dict.fromkeys(classes, 0)
        for i in range(k):
            counts[third[indices[i]]] += 1

        list_values = list(counts.values())
        closest[n] = list_values.index(max(list_values))
