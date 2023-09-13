import numpy as np
import matplotlib.pyplot as plt

# import data form all .dat files
data_50 = np.loadtxt('dtrain13_50.dat')
data_100 = np.loadtxt('dtrain13_100.dat')
data_200 = np.loadtxt('dtrain13_200.dat')
data_400 = np.loadtxt('dtrain13_400.dat')
# create list to store those datasets
data_all = [data_50, data_100, data_200, data_400]
L_list = [1, 2, 4, 8, 16]


# generate x_data and y_data
def dataset(data, L):
    m_all, n = data.shape
    m = round(m_all / 2)
    x_data, y1 = data[:, 1:], data[:, 0]
    y_data = 2 * (y1 == 1) - 1
    # for each class select random labeled y
    index_1 = np.unique(np.random.randint(m, size=L))
    index_3 = np.unique(np.random.randint(m, size=L) + m)
    index_l = np.append(index_1, index_3)
    return x_data, y_data, index_l


# calculate Euclidean distance
def distance(x_l, x_t):
    dist = []
    for i in x_l:
        d = np.sqrt(np.sum((x_t - i) ** 2))
        dist.append(d)
    return np.array(dist)


# use 3-nn to calculate the weight matrix W
def three_nn(X):
    m, _ = X.shape
    W, D = np.zeros((m, m)), np.zeros((m, m))
    for i in range(m):
        dist = distance(X, X[i])
        index = np.argsort(dist)[:4]
        W[i, index] = 1
        W[index, i] = 1
    # Wii = 0
    for i in range(m):
        W[i, i] = 0
    # L = D - W
    for i in range(m):
        val = np.sum(W[i])
        D[i, i] = val

    return W, D - W


# calculate error using LaplacianInterpolation
def LaplacianInterpolation(x_data, y_data, index):
    # get unlabeled value y
    y_t = np.delete(y_data, index)
    # get labeled value y
    y_l = y_data[index]
    W, mat_L = three_nn(x_data)
    # using function (5) in provided paper where:
    # (D_uu - W_uu) @ f_u = W_ul @ f_l
    mat_Lt = np.delete(np.delete(mat_L, index, axis=0), index, axis=1)
    W_t = np.delete(W, index, axis=0)
    v = np.linalg.lstsq(mat_Lt, W_t[:, index] @ y_l, rcond=None)[0]
    y_p = np.sign(v)
    err = np.count_nonzero(y_p - y_t) / len(y_p)
    return err


# calculate error using LaplacianKernelInterpolation
def LaplacianKernelInterpolation(x_data, y_data, index):
    m = len(x_data)
    # get unlabeled value y
    y_t = np.delete(y_data, index)
    W, mat_L = three_nn(x_data)
    mat1 = mat_L[index, :]
    # get Kernel Matrix K
    mat_K = np.linalg.pinv(mat1[:, index])
    # get labeled value y
    y_l = y_data[index]
    alpha = np.linalg.pinv(mat_K) @ y_l
    v = np.zeros(m)
    # get v = sum(α_i*e_i@L+), where i ∈ L
    for i in range(len(index)):
        e_vec = np.zeros(m)
        e_vec[index[i]] = 1
        v += alpha[i] * (e_vec.T @ np.linalg.pinv(mat_L))
    y_p = np.sign(v)
    y_i = np.delete(y_p, index)
    err = np.count_nonzero(y_i - y_t) / len(y_i)
    return err


# using the structure in figure 3.
def q2():
    global data_all, L_list
    mean_LI, mean_LKI = np.zeros((4, 5)), np.zeros((4, 5))
    std_LI, std_LKI = np.zeros((4, 5)), np.zeros((4, 5))
    for i in range(4):
        for j in range(5):
            errLI_a, errLKI_a = [], []
            for iter in range(20):
                x_data, y_data, index = dataset(data_all[i], L_list[j])
                errLI = LaplacianInterpolation(x_data, y_data, index)
                errLKI = LaplacianKernelInterpolation(x_data, y_data, index)
                errLI_a.append(errLI)
                errLKI_a.append((errLKI))
            errLI_a = np.array(errLI_a)
            errLKI_a = np.array(errLKI_a)
            mean1 = np.mean(errLI_a, axis=0)
            mean2 = np.mean(errLKI_a, axis=0)
            std1 = np.std(errLI_a, axis=0)
            std2 = np.std(errLKI_a, axis=0)
            mean_LI[i, j], std_LI[i, j] = mean1, std1
            mean_LKI[i, j], std_LKI[i, j] = mean2, std2

    return mean_LI, std_LI, mean_LKI, std_LKI


mean_LI, std_LI, mean_LKI, std_LKI = q2()

#  headers for the tables
col = ['1', '2', '4', '8', '16']
rowL = ["50", "100", "200", "400"]


# convert the data to string
def table_data(mean, std):
    table_data = []
    for i in range(4):
        row = []
        for j in range(5):
            row.append(str(round(mean[i, j], 3)) + ' ± ' +
                       str(round(std[i, j], 3)))
        table_data.append(row)
    return table_data


# plot table1 for LaplacianInterpolation
table_data1 = table_data(mean_LI, std_LI)
table1 = plt.table(cellText=table_data1, colLabels=col,
                   rowLabels=rowL, cellLoc='center', rowLoc='center', loc="center")

plt.axis('off')
plt.show()

# plot table2 for LaplacianKernelInterpolation
table_data2 = table_data(mean_LKI, std_LKI)
table2 = plt.table(cellText=table_data2, colLabels=col,
                   rowLabels=rowL, cellLoc='center', rowLoc='center', loc="center")

plt.axis('off')
plt.show()
