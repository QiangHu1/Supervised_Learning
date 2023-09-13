import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# generate data randomly for x and y (winnow = 0 or 1)
def data_set(m, n, winnow):
    if winnow == 0:
        X = np.random.choice([-1, 1], (m, n))
    elif winnow == 1:
        X = np.random.choice([0, 1], (m, n))
    return X, X[:, 0]


# perform perceptron algorithm
def perceptron(X, y, x_t):
    m, n = X.shape
    w = np.zeros(n)
    for i in range(m):
        y_h = np.sign(w @ X[i])
        if y[i] != y_h:
            # update w
            w += y[i] * X[i]
    return np.sign(x_t @ w)


# perform winnow algorithm
def winnow(X, y, x_t):
    m, n = X.shape
    w = np.ones(n)
    for i in range(m):
        con = w @ X[i]
        if con < n:
            y_h = 0
        else:
            y_h = 1
        if y[i] != y_h:
            # update w
            w *= 2. ** ((y[i] - y_h) * X[i])
    y_p = x_t @ w
    for i in range(len(y_p)):
        if y_p[i] >= n:
            y_p[i] = 1
        else:
            y_p[i] = 0
    return y_p


# perform least_squares algorithm
def least_squares(X, y, x_t):
    w = np.linalg.pinv(X) @ y
    return np.sign(x_t @ w)


# perform 1_nn algorithm
def one_nn(X, y, x_t):
    m, n = x_t.shape
    y_p = np.zeros(m)
    for i in range(m):
        # calculate Euclidean distance
        distances = np.sqrt(np.sum((X - x_t[i]) ** 2, axis=1))
        y_p[i] = y[np.argmin(distances)]
    return y_p


# calculate the mean and std of sample complexity for selected algorithm
def q3a(winnow, algorithm, N):
    # we set a fixed size for data_test instead of 2**n
    max = 1500
    M_matrix = np.zeros((10, N))
    for iter in range(10):
        M = []
        for n in range(1, N + 1):
            # generate test set
            x_t, y_t = data_set(max, n, winnow)
            for m in range(1, max):
                # generate train set
                data_x, data_y = data_set(m, n, winnow)
                # perform the selected algorithm
                y_p = algorithm(data_x, data_y, x_t)
                err = np.count_nonzero(y_p - y_t) / max
                if err <= 0.1:
                    M.append(m)
                    break
        M_matrix[iter] = M
    mean = np.mean(M_matrix, axis=0)
    std = np.std(M_matrix, axis=0)
    return mean, std


# plot the figure for perceptron
mean_p, std_p = q3a(0, perceptron, 100)
n_p = np.linspace(1, 100, 100, dtype=int)
plt.figure()
plt.xlabel('n')
plt.ylabel('m')
# plot of sample complexity m versus n with error bar
plt.errorbar(n_p, mean_p, yerr=std_p, capsize=2, label='sample complexity')
# use the mean and std we got to fit our plot (we observe it is linear)
f_p = np.polyfit(n_p, mean_p, 1)
p_p = np.poly1d(f_p)
print(p_p)
plt.plot(n_p, p_p(n_p), 'r', label='fitted function')
plt.legend()
plt.show()

# plot the figure for winnow
mean_w, std_w = q3a(1, winnow, 100)
n_w = np.linspace(1, 100, 100, dtype=int)
plt.figure()
plt.xlabel('n')
plt.ylabel('m')
# plot of sample complexity m versus n with error bar
plt.errorbar(n_w, mean_w, yerr=std_w, capsize=2, label='sample complexity')
# use the mean and std we got to fit our plot (we observe it is logarithm)
f_w = np.polyfit(np.log(n_w), mean_w, 1)
p_w = np.poly1d(f_w)
print(p_w)
plt.plot(n_w, p_w(np.log(n_w)), 'r', label='fitted function')
plt.legend()
plt.show()

# plot the figure for least squares
mean_l, std_l = q3a(0, least_squares, 100)
n_l = np.linspace(1, 100, 100, dtype=int)
plt.figure()
plt.xlabel('n')
plt.ylabel('m')
# plot of sample complexity m versus n with error bar
plt.errorbar(n_l, mean_l, yerr=std_l, capsize=2, label='sample complexity')
# use the mean and std we got to fit our plot (we observe it is linear)
f_l = np.polyfit(n_l, mean_l, 1)
p_l = np.poly1d(f_l)
print(p_l)
plt.plot(n_l, p_l(n_l), 'r', label='fitted function')
plt.legend()
plt.show()


# we observe this function have the structure m(n) =  a * (b ** n)
def fit(x, a, b):
    return a * (b ** x)


# plot the figure for 1-nn
# in 1-nn sample complexity become huge as n increases, so we only plot
# part of the figure where n ∈ (0,18)
mean_1, std_1 = q3a(0, one_nn, 17)
n_1 = np.linspace(1, 17, 17, dtype=int)
plt.figure()
plt.xlabel('n')
plt.ylabel('m')
# plot of sample complexity m versus n with error bar
plt.errorbar(n_1, mean_1, yerr=std_1, capsize=2, label='sample complexity')
# in curve fit, use the function structure we observed
popt, pcov = curve_fit(fit, n_1, mean_1)
print(popt)
# get the value y of our fitted function
y_fit = [fit(i, popt[0], popt[1]) for i in n_1]
plt.plot(n_1, y_fit, 'r', label='fitted function')
plt.legend()
plt.show()
