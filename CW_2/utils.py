import numpy as np
import random

# split train test data set in given ratio
def train_test_split(data, ratio=0.8):

    dataset = data

    np.random.shuffle(dataset)

    datas = dataset[:, 1:]
    targets = dataset[:, 0]

    i = round(ratio * len(datas))

    training_data = datas[:i, :]
    training_target = targets[:i]

    testing_data = datas[i:, :]
    testing_target = targets[i:]

    return training_data, training_target, testing_data, testing_target

# split data to perform cross_validation
def cross_validation(dataset, fold):

    datas = dataset[:, 1:].tolist()
    targets = dataset[:, 0].tolist()

    length = len(dataset)

    training_datas = []
    testing_datas = []
    take_index = int((length / fold) * (fold - 1))

    for i in range(fold):
        train_data = np.asarray(datas[take_index:])
        train_target = np.asarray(targets[take_index:])

        test_data = np.asarray(datas[0:take_index])
        test_target = np.asarray(targets[0:take_index])

        training_datas.append([[train_data, train_target]])
        testing_datas.append([test_data, test_target])

        datas.insert(0, datas.pop())
        targets.insert(0, targets.pop())

    return training_datas, testing_datas


def cross_validation1(dataset, targets, fold):

    datas = dataset.tolist()
    targets = targets.tolist()

    length = len(dataset)

    training_datas = []
    testing_datas = []
    take_index = int((length / fold) * (fold - 1))

    for i in range(fold):
        train_data = np.asarray(datas[take_index:])
        train_target = np.asarray(targets[take_index:])

        test_data = np.asarray(datas[0:take_index])
        test_target = np.asarray(targets[0:take_index])

        training_datas.append((train_data, train_target))
        testing_datas.append((test_data, test_target))

        datas.insert(0, datas.pop())
        targets.insert(0, targets.pop())

    return training_datas, testing_datas
