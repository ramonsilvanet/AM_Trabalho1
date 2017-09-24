import numpy as np
import pandas as pandas

def computarCusto(x, y, alpha = 0.01, convergencia = 0.01, max_iteracoes = 1000):
    convergiu = False
    iteracoes = 0

    m = x.shape[0]

    #calculando os thetas iniciais
    th0 = 0
    th1 = 0

    #calculando o erro total, J(theta)
    J = sum([(th0 + th1 * x[i] - y[i]) ** 2 for i in range(m)])

    while not convergiu:
        # para cada exemplo de treinamento, computar o gradiente (d/d_theta_j(theta)
        grad0 = 1.0 / m * sum([(th0 + th1 * x[i] - y[i]) for i in range(m)])
        grad1 = 1.0 / m * sum([(th0 + th1 * x[i] - y[i]) * x[i] for i in range(m)])

        # atualizando os thteas temporarios theta_temp
        temp0 = th0 - alpha * grad0
        temp1 = th1 - alpha * grad1

        #atualizando os thetas
        th0 = temp0
        th1 = temp1

        # erro medio quadratico
        erro = sum([(th0 + th1 * x[i] - y[i]) ** 2 for i in range(m)])

        #verificando o criterio de convergencia
        if abs(J - erro) <= convergencia:
            print('Convergiu após', iter, 'iterações !!!')
            convergiu = True

        #atualizando o erro (J)
        J = erro

        #atualizando o numero de iteraçoes
        iteracoes += 1

        #segundo criterio de parada, maximo de iterações atingidas
        if iteracoes == max_iteracoes:
            print("Maximo de iteracoes atingidas", max_iteracoes)
            convergiu = True

    return th0,th1


# lendo os dados do dataset
dataframe = pandas.read_csv("data/ex1data1.txt", header=None)
dataset = dataframe.values

x = dataset[:, 0:1].astype(float)
y = dataset[:, 1].astype(float)

th0, th1 = computarCusto(x, y)

print("theta0", th0, "theta1", th1)
