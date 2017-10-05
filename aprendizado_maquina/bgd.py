import numpy as np

from aprendizado_maquina.computar_custo.regressao_logistica import custo_regressao_logistica


def gradiente_descendente_batch(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = custo_regressao_logistica(X, y, theta)

    return theta, cost