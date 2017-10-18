import numpy as np

def custo_regressao_linear(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

def custo_regressao_linear_regularizada(theta, X, y, reg):
    m = y.size
    return custo_regressao_linear(X, y, theta) +  (reg / (2 * m)) * np.sum(np.square(theta[1:]))

def gradiente_descendente_batch(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = custo_regressao_linear(X, y, theta)

    return theta, cost
