import csv
import part1 as P1
from Perceptron import OvOPerceptron
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from Perceptron import Perceptron, Gaussion_Kernel, OvOPerceptron
import utils
import heapq


def readData(path):
    data = np.asarray(pd.read_csv(path, header=None, delimiter=r"\s+"))

    return data[:, 0], data[:, 1:]


def readData1(path):
    data = np.asarray(pd.read_csv(path, header=None, delimiter=r"\s+"))

    return data


dataset = readData1("zipcombo.dat")

# print the image from dataset
def print_figure3(figure_data, figure_label, row, col):
    plt.figure()
    for i in range(len(figure_data)):
        plt.subplot(row, col, i + 1)
        plt.imshow(figure_data[i].reshape(16, 16), cmap=plt.cm.gray)
        plt.title(figure_label[i])
        plt.axis('off')

    plt.show()


def Part1_question1():
    train_lists = []
    test_lists = []
    train_stds = []
    test_stds = []
    # d from range 1 - 7
    for d in range(1, 8):
        train_list = []
        test_list = []
        # run 20 ranges
        for i in range(20):
            perceptron = Perceptron(10)
            # split the train test data in 80% scale
            training_data, training_target, testing_data, testing_target = \
                utils.train_test_split(dataset)

            weight, train_error = perceptron.fit_test(training_data, training_target, 10, d)
            train_list.append(train_error)

            test_error = perceptron.predict_test(testing_data, testing_target)
            test_list.append(test_error)
            del perceptron

        train_lists.append(np.mean(train_list))
        test_lists.append(np.mean(test_list))

        train_stds.append(np.std(train_list))
        test_stds.append(np.std(test_list))

    printlist = []
    # plot table for question1
    for i in range(7):
        train_error, train_std = train_lists[i], train_stds[i]
        test_error, test_std = test_lists[i], test_stds[i]
        printlist.append(["{:.3f}%±{:.3f}%".format(train_error * 100, train_std * 100),
                          "{:.3f}%±{:.3f}%".format(test_error * 100, test_std * 100)])

    col = ["train ± std", "test ± std"]
    row = [f"d={x}" for x in range(1, 8)]
    plt.axis("off")
    plt.table(
        cellText=printlist,
        colLabels=col,
        rowLabels=row,
        rowLoc="center",
        loc="center"
    )
    plt.figure(dpi=80)
    plt.show()


def Part1_question2(confusioin=False):
    model_errors = []
    ds = []
    confusion_matrixs = []
    # run 20 rounds
    for i in range(20):
        training_data, training_target, testing_data, testing_target = \
            utils.train_test_split(dataset)

        V_train_data, V_test_data = utils.cross_validation1(training_data, training_target, 5)
        error_list = []
        lowest_error = (0, 1, [])
        # for d in range 1 - 7
        for d in range(1, 8):
            # run cross validation
            for j in range(len(V_train_data)):
                train_d = V_train_data[j]
                test_d = V_test_data[j]

                p2 = Perceptron(10)

                p2.fit_test(train_d[0], train_d[1], 5, d)
                error = p2.predict_test(test_d[0], test_d[1])

                if error < lowest_error[1]:
                    lowest_error = (d, error, train_d)

        p2 = Perceptron(10)
        optimal_test, d_s = lowest_error[2], lowest_error[0]
        p2.fit(optimal_test[0], optimal_test[1], 5, d_s)
        if confusioin:
            model_error, confusion_matrix = p2.predict_confusion(testing_data, testing_target)
            confusion_matrixs.append(confusion_matrix)
            model_errors.append(model_error)
        else:
            model_error = p2.predict_test(testing_data, testing_target)
            model_errors.append(model_error)
        ds.append(d_s)

    error_d = np.mean(model_errors)
    std_error = np.std(model_errors)
    mean_d = np.mean(ds)
    std_d = np.std(ds)
    # plot table for quesiton 2
    col = ["optimal d±std", "test±std"]
    row = [""]
    plt.axis("off")
    plt.table(
        cellText=[["{:E}±{:E}".format(mean_d, std_d), "{:.3f}%±{:.3f}%".format(error_d * 100, std_error * 100)]],
        colLabels=col,
        rowLabels=row,
        rowLoc="center",
        loc="center"
    )
    plt.figure(dpi=80)
    plt.show()
    if confusioin:
        return np.mean(confusion_matrixs, axis=0), np.std(confusion_matrixs, axis=0)


def Part1_question3(confusion_mean, confusion_std):
    print_matrix = [[0 for col in range(10)] for row in range(10)]
    # construct confusion matrix
    for i in range(10):
        for j in range(10):
            print_matrix[i][j] = "{:.2f}%±{:.1f}%".format(confusion_mean[i, j] * 100, confusion_std[i, j] * 100)
    # pot table for question 3
    col = [x for x in range(10)]
    row = [x for x in range(10)]
    plt.figure(dpi=500)
    plt.axis("off")
    table = plt.table(
        cellText=print_matrix,
        colLabels=col,
        rowLabels=row,
        rowLoc="center",
        loc="center",
    )
    table.set_fontsize(12)
    table.scale(1.5,1.5)
    plt.show()


def Part1_question4():
    dataset1 = readData1("zipcombo.dat")
    test_data, test_target = dataset1[:, 1:], dataset1[:, 0]
    errors = np.zeros(len(test_data))
    # run 40 iterations to find the top five hard-to-determine data
    for i in range(40):
        train_data, train_target, _, _ = utils.train_test_split(dataset, 0.5)
        p4 = Perceptron(10)
        p4.fit_test(train_data, train_target, 10, 2)
        error = p4.predict_with_index(test_data, test_target)
        errors = errors + error

    top_five = heapq.nlargest(5, range(len(errors)), errors.take)
    # virtualize top five images
    top_five_data = test_data[top_five]
    top_five_labels = test_target[top_five]
    print_figure3(top_five_data, top_five_labels, 1, 5)

def find_S():
    # function to determine the suitable set of S
    train_lists = []
    test_lists = []
    train_stds = []
    test_stds = []
    c_list = []
    # for range in 1 - 7
    for i in range(1, 8):
        train_list = []
        test_list = []
        c = 10 ** -i
        c_list.append(c)
        # run ten iterations
        for j in range(10):
            perceptron = Perceptron(10, Gaussion_Kernel)
            training_data, training_target, testing_data, testing_target = \
                utils.train_test_split(dataset)

            weight, train_error = perceptron.fit_test(training_data, training_target, 10, c)
            train_list.append(train_error)
            test_error = perceptron.predict_test(testing_data, testing_target)
            test_list.append(test_error)
            del perceptron

        train_lists.append(np.mean(train_list))
        test_lists.append(np.mean(test_list))

        train_stds.append(np.std(train_list))
        test_stds.append(np.std(test_list))

    # plot graph to determine s
    plt.errorbar(c_list, train_lists, yerr=train_stds,label="train error")
    plt.errorbar(c_list, test_lists, yerr=test_stds,label="test error")
    plt.legend()
    plt.show()


def Part1_question5():

    train_lists = []
    test_lists = []
    train_stds = []
    test_stds = []
    cs = []
    # for  d in S 1-7
    for i in range(1, 8):
        train_list = []
        test_list = []
        # calculate c with d
        c = 1.97 ** (-i)
        cs.append(c)
        # run 20 iterations
        for j in range(20):

            perceptron = Perceptron(10, Gaussion_Kernel)
            training_data, training_target, testing_data, testing_target = \
                utils.train_test_split(dataset)

            weight, train_error = perceptron.fit_test(training_data, training_target, 10, c)
            train_list.append(train_error)
            test_error = perceptron.predict_test(testing_data, testing_target)
            test_list.append(test_error)
            del perceptron

        train_lists.append(np.mean(train_list))
        test_lists.append(np.mean(test_list))

        train_stds.append(np.std(train_list))
        test_stds.append(np.std(test_list))
    # plot the table for question 1
    printlist = []
    for i in range(7):
        train_error, train_std = train_lists[i], train_stds[i]
        test_error, test_std = test_lists[i], test_stds[i]
        printlist.append(["{:.2f}%±{:.2f}%".format(train_error * 100, train_std * 100),
                          "{:.2f}%±{:.2f}%".format(test_error * 100, test_std * 100)])

    col = ["train ± std(%)", "test ± std(%)"]
    row = [f"c={cs[x]}" for x in range(0, 7)]
    plt.axis("off")
    plt.table(
        cellText=printlist,
        colLabels=col,
        rowLabels=row,
        rowLoc="center",
        loc="center"
    )
    plt.figure(dpi=80)
    plt.show()

def Part1_question5_2():
    model_errors = []
    cs = []
    dataset = readData1("zipcombo.dat")
    for i in range(20):
        training_data, training_target, testing_data, testing_target = \
            utils.train_test_split(dataset)

        V_train_data, V_test_data = utils.cross_validation1(training_data, training_target, 5)
        lowest_error = (0, 1, [])
        for d in range(1, 8):
            c = 1.97 ** (-d)
            for j in range(len(V_train_data)):
                train_d = V_train_data[j]
                test_d = V_test_data[j]

                p2 = Perceptron(10, Gaussion_Kernel)

                p2.fit_test(train_d[0], train_d[1], 5, c)

                error = p2.predict_test(test_d[0], test_d[1])

                if error < lowest_error[1]:
                    lowest_error = (c, error, train_d)

        p2 = Perceptron(10, Gaussion_Kernel)
        optimal_test, c_s = lowest_error[2], lowest_error[0]
        p2.fit(optimal_test[0], optimal_test[1], 10, 0.014)
        model_error = p2.predict_test(testing_data, testing_target)
        model_errors.append(model_error)
        cs.append(c_s)
    # plot the table for question 5_1
    error_d = np.mean(model_errors)
    std_error = np.std(model_errors)
    mean_d = np.mean(cs)
    std_d = np.std(cs)

    col = ["optimal c±std", "test±std"]
    row = [""]
    plt.axis("off")
    plt.table(
        cellText=[["{:E}±{:E}".format(mean_d, std_d), "{:.3f}%±{:.3f}%".format(error_d * 100, std_error * 100)]],
        colLabels=col,
        rowLabels=row,
        rowLoc="center",
        loc="center"
    )
    plt.figure(dpi=80)
    plt.show()


def Part1_question6():
    train_lists = []
    test_lists = []
    train_stds = []
    test_stds = []
    # for s in 1 - 7
    for d in range(1, 8):
        train_list = []
        test_list = []
        # run 20 rounds and take average
        for i in range(20):
            perceptron = OvOPerceptron(10)
            training_data, training_target, testing_data, testing_target = \
                utils.train_test_split(dataset)

            train_error = perceptron.fit_test(training_data, training_target, 10, d)
            train_list.append(train_error)
            test_error = perceptron.predict_test(testing_data, testing_target)
            test_list.append(test_error)
            del perceptron

        train_lists.append(np.mean(train_list))
        test_lists.append(np.mean(test_list))

        train_stds.append(np.std(train_list))
        test_stds.append(np.std(test_list))
    # plot table 6 _1
    printlist = []
    for i in range(7):
        train_error, train_std = train_lists[i], train_stds[i]
        test_error, test_std = test_lists[i], test_stds[i]
        printlist.append(["{:.3f}%±{:.3f}%".format(train_error * 100, train_std * 100),
                          "{:.3f}%±{:.3f}%".format(test_error * 100, test_std * 100)])

    col = ["train ± std", "test ± std"]
    row = [f"d={x}" for x in range(1, 8)]
    plt.axis("off")
    plt.table(
        cellText=printlist,
        colLabels=col,
        rowLabels=row,
        rowLoc="center",
        loc="center"
    )
    plt.figure(dpi=80)
    plt.show()

def Part1_question6_2(confusion = False):
    model_errors = []
    ds = []
    confusion_matrixs = []
    # run 20 rounds and take average
    for i in range(20):
        training_data, training_target, testing_data, testing_target = \
            utils.train_test_split(dataset)
        # run cross validation
        V_train_data, V_test_data = utils.cross_validation1(training_data, training_target, 5)
        error_list = []
        lowest_error = (0, 1, [])
        for d in range(1, 8):
            for j in range(len(V_train_data)):
                train_d = V_train_data[j]
                test_d = V_test_data[j]

                p2 = OvOPerceptron(10)

                p2.fit_test(train_d[0], train_d[1], 5, d)
                error = p2.predict_test(test_d[0], test_d[1])

                if error < lowest_error[1]:
                    lowest_error = (d, error, train_d)

        p2 = Perceptron(10)
        optimal_test, d_s = lowest_error[2], lowest_error[0]
        p2.fit(optimal_test[0], optimal_test[1], 5, d_s)
        if confusion:
            model_error, confusion_matrix = p2.predict_confusion(testing_data, testing_target)
            confusion_matrixs.append(confusion_matrix)
            model_errors.append(model_error)
        else:
            model_error = p2.predict_test(testing_data, testing_target)
            model_errors.append(model_error)
        ds.append(d_s)
    # plot table 6 2
    error_d = np.mean(model_errors)
    std_error = np.std(model_errors)
    mean_d = np.mean(ds)
    std_d = np.std(ds)

    col = ["optimal d±std", "test±std"]
    row = [""]
    plt.axis("off")
    plt.table(
        cellText=[["{:E}±{:E}".format(mean_d, std_d), "{:.3f}%±{:.3f}%".format(error_d * 100, std_error * 100)]],
        colLabels=col,
        rowLabels=row,
        rowLoc="center",
        loc="center"
    )
    plt.figure(dpi=80)
    plt.show()
    if confusion:
        return np.mean(confusion_matrixs, axis=0), np.std(confusion_matrixs, axis=0)

P1.Part1_question1()
confusion_mean, confusion_std = P1.Part1_question2(True)
P1.Part1_question3(confusion_mean, confusion_std)
P1.Part1_question4()
P1.find_S()
P1.Part1_question5()
P1.Part1_question5_2()
P1.Part1_question6()
P1.Part1_question6_2()