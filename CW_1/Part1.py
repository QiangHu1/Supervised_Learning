import math
import numpy as np
import LinearRegression as lr
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from matplotlib import pyplot as plt

"""
1.1.a
"""
# Given matrix to store data
data = [[1, 3], [2, 2], [3, 0], [4, 5]]

x = [x[0] for x in data]
y = [x[1] for x in data]
k = [1, 2, 3, 4]
a = np.linspace(0, 5, num=1000)
plt.xlabel('a')
plt.ylabel('f')

plt.ylim(-5, 8)
colour_config = ["green", "red", "blue", "brown"]
# plot 4 figure together using formula for estimate y
for i in k:
    w = np.zeros(4)
    w[0:i] = lr.fit(x, y, i)
    f = w[0] + w[1] * a + w[2] * a ** 2 + w[3] * a ** 3
    plt.plot(a, f, color=colour_config[i - 1], label=f'k = {i}')
# plot 4 actual point of data
plt.scatter(x, y)
plt.legend()
plt.grid()
plt.show()

"""
1.1.b
"""
# solutions for question 1.b

# equations for k = 1 : 2.5
# equations for k = 2 : 1.5 + 0.4x
# equations for k = 3 : 9 - 7.1x + 1.5x^2

"""
1.1.c
"""
# print out the MSE using 4 different K (phi function)
print(lr.MSE(x, y, 1, 4))
print(lr.MSE(x, y, 2, 4))
print(lr.MSE(x, y, 3, 4))
print(lr.MSE(x, y, 4, 4))

"""
2.a.i
"""
# epsilon are created randomly under normal distribution
xi = np.random.uniform(0, 1, 30)
epsilon = np.random.normal(0, 0.07, [1, 30])
# g_c is the y value vector we create randomly
g_c = ((np.sin(2 * np.pi * xi) ** 2) + epsilon)[0]
plt.xlabel('xi')
plt.ylabel('g_c')
# plot point with bias
plt.scatter(xi, g_c)
a_1 = np.linspace(0, 1, num=1000)
# plot the actual function
plt.plot(a_1, np.sin(2 * np.pi * a_1) ** 2)
plt.grid()
plt.show()
"""
2.a.ii
"""
k_1 = [2, 5, 10, 14, 18]
plt.ylim(-2, 3)
# fit the data set using 5 different k (phi function)
for i in k_1:
    f_1 = lr.ef(lr.phi(a_1, i), lr.fit(xi, g_c, i))
    plt.plot(a_1, f_1, label=f"k = {i}")

plt.grid()
plt.legend()
plt.show()

"""
2.b
"""
# K = 1 to 18
k_2 = range(1, 19)
plt.xlabel("x")
plt.ylabel("Y")
mse_1 = []
# plot the training error versus K
for k in k_2:
    mse_1.append(math.log(lr.MSE(xi, g_c, k, 30)))

plt.plot(k_2, mse_1)
plt.show()
"""
2.c
"""
# generate 1000 points with epsilon
xi_2 = np.linspace(0, 1, num=1000)
epsilon_1 = np.random.normal(0, 0.07, [1, 1000])
g_2c = ((np.sin(2 * np.pi * xi_2) ** 2) + epsilon_1)[0]

k_2 = range(1, 19)
plt.xlabel("x")
plt.ylabel("Y")
mse_2 = []
# plot the training error versus K
for k in k_2:
    mse_2.append(math.log(lr.MSE(xi_2, g_2c, k, 1000)))

plt.plot(k_2, mse_2)
plt.show()

"""
2.d
"""

plt.xlabel("x")
plt.ylabel("Y")
# create 4 zero vector to store the value of MSE
mse_avgb = np.zeros(18)
mse_avgc = np.zeros(18)
mse_b = np.zeros(18)
mse_c = np.zeros(18)
# run the same thing as (b) and (c) 100 times
for i in range(100):
    mse_avgb += mse_b
    mse_avgc += mse_c
    mse_b = np.zeros(18)
    mse_c = np.zeros(18)
    for k in k_2:
        mse_b[k - 1] = lr.MSE(xi, g_c, k, 30)
        mse_c[k - 1] = lr.MSE(xi_2, g_2c, k, 1000)
# plot two figure using different data set
plt.plot(k_2, np.log(mse_avgb / 100))
plt.plot(k_2, np.log(mse_avgc / 100))
plt.show()

"""
3
"""

# using another phi function to repeat the calculation like 2 (b-d)
def Question3():
    # repeat 2 (b)
    k_2 = range(1, 19)
    plt.xlabel("x")
    plt.ylabel("Y")
    mse_1 = []
    for k in k_2:
        mse_1.append(math.log(lr.MSE2(xi, g_c, k, 30)))
    # plot the training error versus K
    plt.plot(k_2, mse_1)
    plt.show()

    # repeat 2 (c)
    # generate 1000 points with epsilon
    xi_2 = np.linspace(0, 1, num=1000)
    epsilon_1 = np.random.normal(0, 0.07, [1, 1000])
    g_2c = ((np.sin(2 * np.pi * xi_2) ** 2) + epsilon_1)[0]

    k_2 = range(1, 19)
    plt.xlabel("x")
    plt.ylabel("Y")
    mse_2 = []
    for k in k_2:
        mse_2.append(math.log(lr.MSE2(xi_2, g_2c, k, 1000)))
    # plot the training error versus K
    plt.plot(k_2, mse_2)
    plt.show()

    # repeat 2 (d)
    plt.xlabel("x")
    plt.ylabel("Y")
    mse_avgb = np.zeros(18)
    mse_avgc = np.zeros(18)
    mse_b = np.zeros(18)
    mse_c = np.zeros(18)
    for i in range(100):
        mse_avgb += mse_b
        mse_avgc += mse_c
        mse_b = np.zeros(18)
        mse_c = np.zeros(18)
        for k in k_2:
            mse_b[k - 1] = lr.MSE2(xi, g_c, k, 30)
            mse_c[k - 1] = lr.MSE2(xi_2, g_2c, k, 1000)
    # plot two figure using different data set in (b) and (c)
    plt.plot(k_2, np.log(mse_avgb / 100))
    plt.plot(k_2, np.log(mse_avgc / 100))
    plt.show()

# run Question 3
Question3()

"""
4.a
"""
# read the csv file 'Boston-filtered.csv'
csv_data = pd.read_csv('Boston-filtered.csv')
data = np.array(csv_data)
# divide the data into x and y, which can be used for all the part below
data_x = data[:, 0:12]
data_y = data[:, -1]

# definite a function in order to repeat 4 (a)
def ques4a():
    global data_x, data_y
    mse_train_4a = np.zeros(20)
    mse_test_4a = np.zeros(20)
    # split data 20 times and calculate the average MSE
    for iter in range(20):
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33)
        # create the training set and test set for x which are vectors of ones
        x_train1 = np.ones(len(x_train))
        x_test1 = np.ones(len(x_test))
        mse_train_4a[iter] = lr.MSE4(x_train1, y_train, lr.fit4(x_train1, y_train))
        mse_test_4a[iter] = lr.MSE4(x_test1, y_test, lr.fit4(x_train1, y_train))
    return np.sum(mse_train_4a) / 20, np.sum(mse_test_4a) / 20, \
           np.std(mse_train_4a, ddof=1), np.std(mse_test_4a, ddof=1)


mse_train_4a, mse_test_4a, std_train_4a, std_test_4a = ques4a()
print("The MSE for training set is ", mse_train_4a)
print("The MSE for test set is ", mse_test_4a)

"""
4.c
"""


def ques4c():
    global data_x, data_y
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33)
    # create two matrix to store the data of MSE for train and test set of different attributes
    mse_train_4c_m, mse_test_4c_m = np.zeros((12, 20)), np.zeros((12, 20))
    mse_train_4c, mse_test_4c, std_train_4c, std_test_4c = np.zeros(12), np.zeros(12), np.zeros(12), np.zeros(12)
    bias_train = np.ones(len(x_train))
    bias_test = np.ones(len(x_test))
    # split data 20 times and calculate the average MSE
    for iter in range(20):
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33)
        for i in range(12):
            x_train2 = np.c_[np.asmatrix(x_train[:, i]).T, np.asmatrix(bias_train).T]
            x_test2 = np.c_[np.asmatrix(x_test[:, i]).T, np.asmatrix(bias_test).T]
            mse_train_4c_m[i, iter] = lr.MSE4(x_train2, y_train, lr.fit4(x_train2, y_train))
            mse_test_4c_m[i, iter] = lr.MSE4(x_test2, y_test, lr.fit4(x_train2, y_train))
    # take the MSE for different attributes out
    for i in range(12):
        mse_train_4c[i] = np.sum(mse_train_4c_m[i]) / 20
        mse_test_4c[i] = np.sum(mse_test_4c_m[i]) / 20
        std_train_4c[i] = np.std(mse_train_4c_m[i], ddof=1)
        std_test_4c[i] = np.std(mse_test_4c_m[i], ddof=1)

    return mse_train_4c, mse_test_4c, std_train_4c, std_test_4c

# Print out the average MSE for different attributes
mse_train_4c, mse_test_4c, std_train_4c, std_test_4c = ques4c()
for i in range(12):
    print("MSE for training set of Linear Regression with attribute ", i + 1, " is ", mse_train_4c[i])
    print("MSE for test set of Linear Regression with attribute ", i + 1, " is ", mse_test_4c[i])

"""
4.d
"""

# calculate the MSE when using all attributes
def ques4d():
    global data_x, data_y
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33)
    bias_train = np.ones(len(x_train))
    bias_test = np.ones(len(x_test))
    mse_train_4d, mse_test_4d = np.zeros(20), np.zeros(20)
    # repeat 20 times and take the average value
    for iter in range(20):
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33)
        x_train3 = np.c_[x_train, np.asmatrix(bias_train).T]
        x_test3 = np.c_[x_test, np.asmatrix(bias_test).T]
        mse_train_4d[iter] = lr.MSE4(x_train3, y_train, lr.fit4(x_train3, y_train))
        mse_test_4d[iter] = lr.MSE4(x_test3, y_test, lr.fit4(x_train3, y_train))

    return np.sum(mse_train_4d) / 20, np.sum(mse_test_4d) / 20, \
           np.std(mse_train_4d, ddof=1), np.std(mse_test_4d, ddof=1)

# print out the final solution
mse_train_4d, mse_test_4d, std_train_4d, std_test_4d = ques4d()
print("The MSE for training set is ", mse_train_4d)
print("The MSE for test set is ", mse_test_4d)

"""
5.a
"""


def ques5a():
    global data_x, data_y
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33)
    # create two array to store the value of sigma and gamma
    gamma = np.array([])
    sigma = np.array([])
    for i in range(-40, -25):
        gamma = np.append(gamma, 2 ** i)
    for j in np.arange(7, 13.5, 0.5):
        sigma = np.append(sigma, 2 ** j)
    # lens means the iteration times
    lens = len(gamma) * len(sigma)
    x_gam, y_sig, z_mse = np.zeros(lens), np.zeros(lens), np.zeros(lens)
    # initialize the start value of best MSE, which should be big enough
    best = 1e+10
    iters = 0
    for gam in gamma:
        for sig in sigma:
            # initial the start mse value and KFold
            kf = KFold(n_splits=5)
            mse = 0
            # using 5 folds to split the training and test set
            for train_index, test_index in kf.split(x_train):
                x_tra, x_tes = x_train[train_index], x_train[test_index]
                y_tra, y_tes = y_train[train_index], y_train[test_index]
                ef = lr.ef_kernel(x_tra, y_tra, x_tes, sig, gam)
                mse += lr.MSE_kernel(ef, y_tes)
            MSE = mse / 5
            x_gam[iters], y_sig[iters], z_mse[iters] = gam, sig, MSE
            iters += 1
            if MSE < best:
                best = MSE
                sigma_best = sig
                gamma_best = gam
    # use the best sigma and gamma to calculate the MSE of the train and test set.
    ef_train = lr.ef_kernel(x_train, y_train, x_train, sigma_best, gamma_best)
    MSE_train = lr.MSE_kernel(ef_train, y_train)
    ef_test = lr.ef_kernel(x_train, y_train, x_test, sigma_best, gamma_best)
    MSE_test = lr.MSE_kernel(ef_test, y_test)
    return sigma_best, gamma_best, MSE_train, MSE_test, x_gam, y_sig, z_mse

# print out the solution
sigma_best, gamma_best, MSE_train, MSE_test, x_gam, y_sig, z_mse = ques5a()
print("Best sigma is ", sigma_best, "Best gamma is ", gamma_best)
print("Best test error is ", MSE_test, "Best train error is ", MSE_train)

"""
5.b
"""
# plot MSE versus gamma and sigma as a function
fig = plt.figure()
ax1 = plt.axes(projection='3d')
ax1.set_xlabel('gamma')
ax1.set_ylabel('sigma')
ax1.set_zlabel('MSE')
ax1.scatter3D(x_gam, y_sig, z_mse)
plt.show()

"""
5.c
"""
# use the best sigma and gamma we got in 5 (a) to calculate the MSE for training and test set
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33)
ef_train = lr.ef_kernel(x_train, y_train, x_train, sigma_best, gamma_best)
MSE_train = lr.MSE_kernel(ef_train, y_train)
ef_test = lr.ef_kernel(x_train, y_train, x_test, sigma_best, gamma_best)
MSE_test = lr.MSE_kernel(ef_test, y_test)
print("Best test error is ", MSE_test, "Best train error is ", MSE_train)

"""
5.d
"""
# repeat 4a,c,d (already repeated 20 times in function)
mse_train_4a, mse_test_4a, std_train_4a, std_test_4a = ques4a()
mse_train_4c, mse_test_4c, std_train_4c, std_test_4c = ques4c()
mse_train_4d, mse_test_4d, std_train_4d, std_test_4d = ques4d()
mse_train_5, mse_test_5 = np.zeros(20), np.zeros(20)
# repeat 5a,c 20 times
for i in range(20):
    sigma_best, gamma_best, MSE_train, MSE_test, x_gam, y_sig, z_mse = ques5a()
    mse_train_5[i], mse_test_5[i] = MSE_train, MSE_test

mse_train_5c, mse_test_5c, std_train_5c, std_test_5c = np.sum(mse_train_5) / 20, np.sum(mse_test_5) / 20, \
                                                       np.std(mse_train_5, ddof=1), np.std(mse_test_5, ddof=1)
# Create the table to store all the information
df = pd.DataFrame(columns=['Method', 'MSE train', 'MSE test'])
# add the row for native regression
df.loc[0] = ['Native Regression', str(round(mse_train_4a, 2)) + ' ± ' + str(round(std_train_4a)),
             str(round(mse_test_4a, 2)) + ' ± ' + str(round(std_test_4a))]
# add the row for linear regression with attribute 1 to 12
for i in range(1, 13):
    df.loc[i] = ['Linear Regression (attribute ' + str(i) + ')', str(round(mse_train_4c[i - 1], 2)) + ' ± ' +
                 str(round(std_train_4c[i - 1])),
                 str(round(mse_test_4c[i - 1], 2)) + ' ± ' + str(round(std_test_4c[i - 1]))]
# add the row for linear regression with all attributes
df.loc[13] = ['Linear Regression (all attributes)', str(round(mse_train_4d, 2)) + ' ± ' + str(round(std_train_4d)),
              str(round(mse_test_4d, 2)) + ' ± ' + str(round(std_test_4d))]
# add the row for Kernel Ridge regression
df.loc[14] = ['Kernel Ridge Regression', str(round(mse_train_5c, 2)) + ' ± ' + str(round(std_train_5c)),
              str(round(mse_test_5c, 2)) + ' ± ' + str(round(std_test_5c))]

pd.set_option('display.max_columns', None)
# print out the table
print(df)
