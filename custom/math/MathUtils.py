import numpy as np


def gini(p):
    q = 1 - p
    return p * q + q * (1 - q)


def entropy(distribution):

    _entropy = 0
    for p in distribution:
        _entropy += p * np.log2(p)

    return - _entropy


def error(p):
    q = 1 - p
    return 1 - np.max([p, q])


def get_distribution(data):
    np_data = np.array(data)
    unique, counts = np.unique(np_data, return_counts=True)
    probabilities = counts / np.sum(counts)

    return unique, probabilities
