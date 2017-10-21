import numpy as np
from scipy.optimize import minimize

def custo_regressao_linear(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

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

def custo_regressao_linear_regularizada(theta, X, y, reg):
    m = y.size
    h = X.dot(theta)
    J = (1/(2*m))*np.sum(np.square(h-y)) + (reg/(2*m))*np.sum(np.square(theta[1:]))
    return(J)

def gradiente_descendente_batch_com_regularicao(theta, X, y, reg):
    m = y.size
    h = X.dot(theta.reshape(-1,1))
    grad = (1/m)*(X.T.dot(h-y))+ (reg/m)*np.r_[[[0]],theta[1:].reshape(-1,1)]    
    return(grad.flatten())

def treinar_modelo(X, y, regularizacao):        
    initial_theta = np.array([[15],[15]])
    resultado = minimize(custo_regressao_linear_regularizada, initial_theta, args=(X,y,regularizacao), 
               method=None, jac=gradiente_descendente_batch_com_regularicao, options={'maxiter':5000})
    return resultado