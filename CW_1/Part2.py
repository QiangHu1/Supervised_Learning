import random

from matplotlib.colors import ListedColormap

from KNN import KNN
import numpy as np
import matplotlib.pyplot as plt

KNN0 = KNN()


# 2.1.1
# question 6
def plot_KNN_boundary():
    # generate trains data set and test data set
    data_test, target_test, data, target = KNN0.generate_uniform(100)
    # get data points of 1s and 0s
    data1 = data[np.where(target == 1)]
    data0 = data[np.where(target == 0)]
    # train the KNN model
    KNN0.train(data, target)
    # print in Contour lines to represent decision boundary
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    x1, y1 = np.meshgrid(x, y)
    xy = np.c_[x1.ravel(), y1.ravel()]
    predict = KNN0.predict(xy, 3).reshape(x1.shape)
    custom_cmap = ListedColormap(['#7FFFAA', '#FFC0CB'])
    plt.title("figure 1")
    plt.contourf(x1, y1, predict, alpha=0.3, cmap=custom_cmap)
    plt.scatter(data0[:, 0], data0[:, 1], c='green', label="0")
    plt.scatter(data1[:, 0], data1[:, 1], c='red', label="1")
    plt.legend()
    plt.show()


plot_KNN_boundary()


# 2.1.2
# question 7
# generate dataset with bias
def generate_bias(n):

    data = np.random.uniform(0, 1, size=(n + 1000, 2))
    target = KNN0.predict(data[:n])

    test_data = data[n:]
    test_target = KNN0.predict(test_data)

    # generate bias with flipping coin method to training target
    for i in range(len(target)):
        if random.random() < 0.8:
            pass
        else:
            target[i] = random.randint(0, 1)
    # generate bias with flipping coin method to testing target
    for i in range(len(test_target)):
        if random.random() < 0.8:
            pass
        else:
            test_target[i] = random.randint(0, 1)

    return test_data, test_target, data[:n], target


# estimate generalization error with different ks
def estimate_generalization_error():
    ks = []
    errors = []
    # repeat from k = 1 to k = 50
    for k in range(1, 51):
        error = 0
        # repeat 100 times and take average
        for i in range(100):
            # train with 4000 data points
            test_data, test_target, data, target = generate_bias(4000)
            KNN1 = KNN()
            KNN1.train(data, target)
            # test with 1000 data points
            result = KNN1.predict(test_data, k)
            error += np.sum(np.abs(result - test_target)) / 1000
        error = error / 100
        ks.append(k)
        errors.append(error)
    plt.xlabel("k")
    plt.ylabel("generalization error")
    plt.plot(ks, errors)
    plt.show()


estimate_generalization_error()


# 2.1.3
# question 8
# plot the graph of m against k
def find_optimal_k():
    ms = list(range(0, 4001, 500))
    ms[0] += 100
    xs = []
    ys = []
    # run for different value of m
    for m in ms:
        total_k = 0
        # repeat 100 times and take the average
        for i in range(100):
            k = 0
            error = 1
            # repeat from k = 1 to k = 50
            for k1 in range(1, 51, 1):
                # use data with bias to train KNN
                data_test, target_test, data, target = generate_bias(m)
                KNN1 = KNN()
                KNN1.train(data, target)
                # test model with 1000 data points with bias
                result = KNN1.predict(data_test, k1)
                e = np.sum(np.abs(result - target_test)) / 1000
                # record the minimum error
                if e < error:
                    error = e
                    k = k1
            total_k += k
        optimal_k = total_k / 100
        xs.append(m)
        ys.append(optimal_k)
    plt.xlabel("m")
    plt.ylabel("k")
    plt.plot(xs, ys)
    plt.show()


find_optimal_k()
