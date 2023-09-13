import numpy as np


# Build function return phi(x) with parameter k
def phi(x, k):
    out = []
    for i in range(len(x)):
        row = []
        for j in range(k):
            row.append(pow(x[i], j))
        out.append(row)
    return np.asarray(out)


# This function return a K-dimensional vector w
def fit(x, y, k):
    X = phi(x, k)
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    return w


# Calculate the estimate y-vector
def ef(x, w):
    return x @ w


# Given the parameter x, y, k, m, return the MSE
def MSE(x, y, k, m):
    X = phi(x, k)
    w = fit(x, y, k)
    return ((X @ w - y).transpose() @ (X @ w - y)) / m


# Calculate the MSE use different phi function
def MSE2(x, y, k, m):
    out = []
    for i in range(len(x)):
        row = []
        for j in range(1, k + 1):
            row.append(np.sin(j * np.pi * x[i]))
        out.append(row)
    X = np.asarray(out)
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    return ((X @ w - y).transpose() @ (X @ w - y)) / m


# return w with given X matrix
def fit4(x, y):
    return np.linalg.inv(np.asmatrix(x.T @ x)) @ np.asmatrix(x.T @ y).T


# Calculate the MSE with given phi function
def MSE4(x, y, w):
    if len(w) == 1:
        ef = w @ np.asmatrix(x)
    else:
        ef = (np.asmatrix(x) @ w).T
    mse_sum = 0
    for i in range(len(y)):
        mse_sum += (ef[0, i] - y[i]) ** 2
    return mse_sum / len(y)


# Gaussian Kernel formula
def Gaussian_kernel(xi, xj, sigma):
    return np.exp((-np.linalg.norm(xi - xj) ** 2) / (2 * sigma ** 2))


# Calculate Kernel matrix using Gaussian Kernel
def kernel_matrix(x_train, sigma):
    k = np.zeros((len(x_train), len(x_train)))
    for i in range(len(x_train)):
        for j in range(len(x_train)):
            k[i, j] = Gaussian_kernel(x_train[i], x_train[j], sigma)
    return np.asmatrix(k)


# Return alpha matrix
def fit_kernel(x_train, y_train, sigma, gamma):
    alpha = np.linalg.inv(kernel_matrix(x_train, sigma) + (gamma * len(x_train) * np.identity(len(x_train)))) @ y_train
    return np.asarray(alpha.T)


# Calculate the estimate y-vector
def ef_kernel(x_train, y_train, x_test, sigma, gamma):
    ef = np.zeros(len(x_test))
    alpha = fit_kernel(x_train, y_train, sigma, gamma)
    for j in range(len(x_test)):
        for i in range(len(x_train)):
            ef[j] += alpha[i] * Gaussian_kernel(x_train[i], x_test[j], sigma)
    return ef


# Calculate the final MSE using estimate y and actual Y
def MSE_kernel(ef, y):
    mse_sum = 0
    for i in range(len(y)):
        mse_sum += (ef[i] - y[i]) ** 2
    return mse_sum / len(y)
