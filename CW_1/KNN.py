import numpy as np


# KNN learning class
class KNN:

    def __init__(self):
        self.target = None
        self.data = None

    # KNN using lazy learning algorithm, hence only copy the data points
    def train(self, data, target):
        self.data = data
        self.target = target

    # using the majority of nearest k points to predict the label of new incoming data
    def predict(self, x, k=3):
        labels = []
        distances = self.euclidean_dist(x)
        for i in distances:
            # sort by distance to input data point
            neighbors_index = np.argsort(i)[:k]
            neighbor_labels = self.target[neighbors_index]
            # take k nearest neighbor points
            label = np.argmax(np.bincount(neighbor_labels.astype(int)))
            labels.append(label)

        return np.array(labels)

    # calculate the distances from each data point to all training points
    def euclidean_dist(self, x):
        out = []
        for i in x:
            distances = np.sqrt(np.sum((self.data - i) ** 2, axis=1))
            out.append(distances)

        return np.array(out)

    # uniformly generate [1,0] and split into training data and test data
    def generate_uniform(self, number):
        data = np.random.uniform(0, 1, size=(number + 1000, 2))

        target = np.random.randint(0, 2, size=number + 1000)

        return np.array(data[number:]), np.array(target[number:]), np.array(data[:number]), np.array(target[:number])
