import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

file_names = ['polydata_data_sampx.txt', 'polydata_data_sampy.txt']


def load_data(file):
    with open(f'datasets/PA-1-data-text/{file}', 'r') as file:
        x = file.read()
        numbers_str = x.split(' ')
        numbers = [float(num[:-1]) for num in numbers_str if num != '']
        return np.array(numbers, dtype=np.float32)


def poly_encode(fea_x, k):
    return np.array([[fea_x[i] ** n for n in range(k + 1)] for i in range(len(fea_x))]).T


def poly_closed_regression(fea_x, fea_k, fea_y, method, lam=1):
    encoded_x = poly_encode(fea_x, fea_k)
    if method == 'LS':
        return (np.linalg.inv(encoded_x @ encoded_x.T) @ encoded_x) @ fea_y.T
        # return np.matmul(np.matmul(np.linalg.inv(np.matmul(encoded_x, encoded_x.T)), encoded_x), fea_y.T)

    elif method == 'RLS':
        return (np.linalg.inv(encoded_x @ encoded_x.T + lam) @ encoded_x) @ fea_y.T
        # return np.matmul(np.matmul(np.linalg.inv(np.matmul(encoded_x, encoded_x.T) + lam), encoded_x), fea_y.T)


def lasso(fea_x, fea_y, fea_k, method, lam=1):
    zeta = poly_encode(fea_x, fea_k)
    zeta_qua = np.array(zeta @ zeta.T)
    H = [
        [zeta_qua, -zeta_qua],
        [-zeta_qua, zeta_qua]
    ]
    f = np.block([zeta @ fea_y, -zeta @ fea_y]) + lam
    H = np.block(H)
    G = -np.eye((fea_k + 1), H.shape[0])
    h = np.zeros(zeta_qua.shape[0])
    sv = solvers.qp(P=matrix(H), q=matrix(f), G=matrix(G), h=matrix(h))
    return - sv['x'][0:5] + sv['x'][5:]

def rr(fea_x, fea_y, fea_k, method, lam=1):
    zeta = np.matrix(poly_encode(fea_x, fea_k))
    D = fea_k + 1
    n = len(fea_y)
    f = np.block([np.zeros(shape=D), np.ones(shape=n)])
    A = np.block([
        [-zeta.T, -np.identity(n)],
        [zeta.T, -np.identity(n)]
    ])
    b = matrix(np.concatenate([y*-1, y]).astype(np.double))
    A = matrix(A)
    f = matrix(f)
    sv = solvers.lp(c=f, G=A, h=b)
    return sv['x'][:D]


def test_model(data_x, data_y, fea_k, method, lam=1):
    reg_fun = poly_closed_regression
    if method == 'LASSO':
        reg_fun = lasso
    elif method == 'RR':
        reg_fun = rr
    elif method == 'LS' or 'RLS':
        reg_fun = poly_closed_regression
    zeta = reg_fun(fea_x=data_x, fea_k=fea_k, fea_y=data_y, method=method, lam=lam)
    print(zeta)
    x_grid = np.array(list(range(-30, 30))) / 10
    y_pred = np.matmul(poly_encode(x_grid, k).T, zeta)
    plt.scatter(x, y)
    plt.plot(x_grid, y_pred)
    plt.show()


if __name__ == '__main__':
    k = 5
    x = load_data(file_names[0])
    y = load_data(file_names[1])
    test_model(data_x=x, data_y=y, fea_k=k, method='RR')
