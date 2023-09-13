import itertools

import numpy as np

# implement poly kernel
def ploy_kernel(M, N, n):
    return (M.dot(N.T)) ** n

# implement Gussion kernel
def Gaussion_Kernel(M, N, c):
    # ||u - v||^2 = ||u||^2 + ||v||^2 - 2 <u,v>
    M1 = np.expand_dims(np.sum(M ** 2, axis=-1), axis=1)
    N1 = np.expand_dims(np.sum(N ** 2, axis=-1), axis=0)

    kernel_matrix = np.exp(-c * (M1 + N1 - 2 * (np.dot(M, N.T))))

    return kernel_matrix

# perceptron class
class Perceptron:

    def __init__(self, class_number, kernelize=ploy_kernel):
        # kernel dimension
        self.kernel_d = None
        # training data
        self.train_X = None

        self.weight = None

        self.class_number = class_number
        # predifened kernel function
        self.kernelize = kernelize

    # train the model with dataset
    def fit(self, data_X, data_y, epoch, d):

        # set the size of dataset
        data_length, data_index = data_X.shape
        self.kernel_d = d
        self.weight = np.zeros((self.class_number, data_length))
        # kernelize the x data with dimension d
        kernel_matrix = self.kernelize(data_X, data_X, d)
        self.train_X = data_X
        # repeat for fixed epochs
        test_converage = 0
        for i in range(epoch):
            # maintaining the error of each epoch
            errors = 0
            for t in range(data_length):

                # pick the target y of the greatest magnitude
                predict_y = self.predict_one(kernel_matrix[t, :])
                # check is it close to the target output
                if (predict_y - data_y[t]) > 0.01:
                    errors += 1
                # update the weight by the difference of the predicted value
                # to the real value, apply half of their difference as reward
                # /punishment to reduce the search space
                self.weight[predict_y, t] -= abs(predict_y - data_y[t]) / 2
                self.weight[int(data_y[t]), t] += abs(predict_y - data_y[t]) / 2

            # if no more training needed exit
            if test_converage == errors:
                break
            test_converage = errors
        return self.weight

    def fit_test(self, data_X, data_y, epoch, d):
        # set the size of dataset
        data_length, data_index = data_X.shape
        self.kernel_d = d
        self.weight = np.zeros((self.class_number, data_length))
        # kernelize the x data with dimension d
        kernel_matrix = self.kernelize(data_X, data_X, d)
        epoch_error = 0
        self.train_X = data_X
        # repeat for fixed epochs
        test_converage = 0
        for i in range(epoch):
            # maintaining the error of each epoch
            errors = 0
            for t in range(data_length):

                predict_y = self.predict_one(kernel_matrix[t, :])
                # check is it close to the target output
                if abs(predict_y - data_y[t]) > 0.01:
                    errors += 1
                # update the weight by the difference of the predicted value
                # to the real value, apply half of their difference as reward
                # /punishment to reduce the search space
                self.weight[predict_y, t] -= abs(predict_y - data_y[t]) / 2
                self.weight[int(data_y[t]), t] += abs(predict_y - data_y[t]) / 2

            epoch_error = errors / data_length
            # if no more training needed exit
            if test_converage == errors:
                break

            test_converage = errors

        return self.weight, epoch_error

    # predict one data sample
    def predict_one(self, M):
        # magnitude of blief of the prediction y that is true
        # pick the target y of the greatest magnitude
        confidence = np.dot(self.weight, M)
        return int(np.argmax(confidence))

    # predict set of data
    def predict(self, M):
        # magnitude of blief of the prediction y that is true
        # pick the target y of the greatest magnitude
        y = np.zeros(M.shape[0])
        M = self.kernelize(self.train_X, M, self.kernel_d)
        predict = np.dot(self.weight, M)
        for _ in range(y.shape[0]):
            y[_] = np.argmax(predict[:, _])
        return y

    # predict function with error
    def predict_test(self, test_data, test_target):
        error = 0
        predicts = self.predict(test_data)
        for i in range(len(test_target)):
            if abs(predicts[i] - test_target[i]) > 0.01:
                error += 1
        error = error / len(test_target)
        return error

    #predict function with confusion matrix
    def predict_confusion(self, test_data, test_target):
        error = 0
        predicts = self.predict(test_data)

        confusion_matrix = np.zeros((10, 10))
        count_matrix = np.ones((10, 10))

        for i in range(len(test_target)):
            count_matrix[int(test_target[i]), :] += 1
            if abs(predicts[i] - test_target[i]) > 0.01:
                error += 1
                confusion_matrix[int(test_target[i]), int(predicts[i])] += 1

        error = error / len(test_target)

        return error, confusion_matrix / count_matrix

    # predict with the dataset index
    def predict_with_index(self, test_data, test_target):
        errors = np.zeros(len(test_target))

        predicts = self.predict(test_data)
        for i in range(len(test_target)):
            if predicts[i] != test_target[i]:
                errors[i] += 1

        return errors


class OvOPerceptron:

    def __init__(self, class_number, kernelize=ploy_kernel):
        # the algorithm maintains set of classifiers
        self.classifiers = None
        # map the index into classifier index
        self.value_table = None
        # number of classifiers
        self.classifier_number = None
        # kernel dimesion
        self.kernel_d = None
        # training data
        self.train_X = None
        # number of classes
        self.class_number = class_number
        # predefined kernel function
        self.kernelize = kernelize
        # map the index into classifier index
        self.njis = self.generate_classifier()

    # generate data map
    def generate_classifier(self):
        ijs = []
        for i in range(self.class_number):
            for j in range(i + 1, self.class_number):
                ijs.append((i, j))
        return ijs

    def fit_test(self, data_X, data_y, epoch, d):
        # set the size of dataset
        data_length, data_index = data_X.shape
        self.classifier_number = int(self.class_number * (self.class_number - 1) / 2)
        self.kernel_d = d

        self.weights = np.zeros((self.classifier_number, data_length))

        # kernelize the x data with dimension d
        self.kernel_matrix = self.kernelize(data_X, data_X, d)
        epoch_error = 0
        self.train_X = data_X
        # repeat for fixed epochs
        test_converage = 0
        for i in range(epoch):
            # maintaining the error of each epoch
            errors = 0

            for t in range(data_length):
                # calculate the magnitude of choosing a set of classes
                confidence = np.dot(self.weights, self.kernel_matrix[:, t])
                # pick the one with highest vote as predict value
                predict_y = self.predict_one(confidence)
                # count errors
                if data_y[t] != predict_y:
                    errors += 1
                confidence = np.sign(confidence)
                # update the classifier when the classifier engaged the correct prediction
                for k in range(len(self.njis)):
                    a = self.njis[k][0]
                    b = self.njis[k][1]

                    if data_y[t] == a and confidence[k] > -1:
                        # if a is predicted correctly increase the confident of predict a for the classifier
                        self.weights[k, t] = self.weights[k, t] - 1
                    elif data_y[t] == b and confidence[k] < 1:
                        # if a is predicted correctly increase the confident of predict a for the classifier
                        self.weights[k, t] = self.weights[k, t] + 1

            epoch_error += errors
            # if no more training needed exit
            if test_converage == errors:
                break

            test_converage = errors

        return epoch_error / data_length / epoch

    def predict_one(self, confidence):
        # fit the data with weight
        vote_count = np.zeros(self.class_number)

        # use confidence to vote between two classes
        confidence = np.sign(confidence) + 1
        confidence[confidence > 1] = 1

        # take the class with most votes
        for a, b in enumerate(confidence.tolist()):
            vote_value = self.njis[a][int(b)]
            vote_count[vote_value] += 1

        return np.argmax(vote_count)

    # predict function with error
    def predict_test(self, test_data, test_target):
        error = 0
        kernel_matrix = self.kernelize(self.train_X, test_data, self.kernel_d)
        for i in range(len(test_target)):
            confidence = np.dot(self.weights , kernel_matrix[:,i])
            predicts = self.predict_one(confidence)
            if predicts != test_target[i]:
                error += 1
        error = error / len(test_target)
        return error
