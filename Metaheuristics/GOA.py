from random import *
from math import *
from Problem.SCP.problem import SCP  # Para solucion de SCP
from Problem.Benchmark.Problem import fitness as f   # Para solucion de funciones Benchmark

import numpy as np
import math

# ========= PARAMETROS =========
M = 2.5  # Peso en Kg del Gannet
VEL = 1.5  # Velocidad en el agua en m/s del Gannet
C = 0.2  # Determina si se ejejuta movimiento levy o ajustes de trayectoria
FACTOR_FASE = 0.5  # Determina que fase exploracion o explotacion usa el algoritmo en la iteracion

BETA = 1.5



# ========= FUNCIONES =========
def v(x):
    if x > math.pi:
        return ((1 / pi) * x - 1)
    return (-(1 / pi) * x + 1)


def levy():
    mu = uniform(0, 1)
    v = uniform(0, 1)
    gamma1 = math.gamma(1 + BETA)
    gamma2 = math.gamma((1+BETA)/2)
    seno = math.sin(math.pi * BETA / 2)
    expo = 1 / BETA
    sigma = ((gamma1 * seno) / (gamma2 * BETA * 2**((BETA-1)/2))) ** expo
    resultado = (0.01 * mu * sigma) / (abs(v) ** expo)
    return resultado


def obtener_individuo_random(poblacion):
    index_random = randint(0, len(poblacion) - 1)
    return poblacion[index_random]


def obtener_individuo_promedio(poblacion, dim):
    individuo = []
    for j in range(dim):
        suma = 0
        for i in range(len(poblacion)):
            suma += poblacion[i][j]
        promedio = suma / len(poblacion)
        individuo.append(promedio)
    return individuo


# ========= ALGORITMO =========

# x = iniciar_poblacion_random(N, DIM)

# mx = x.copy()  # Crear matriz de memoria


# Busca mejor solucion
def iterarGOA(maxIter, iter, dim, poblacion, individuo_mejor, fitness, function):
    instance = SCP(function)
    t = 1 - (iter / maxIter)
    rand = uniform(0, 1)
    mx = poblacion.copy()
    # Calculos constantes por iteracion

    individuo_random = obtener_individuo_random(poblacion)
    individuo_promedio = obtener_individuo_promedio(poblacion, dim)

    # ========= Exploracion =========
    if rand > FACTOR_FASE:
        for i in range(len(poblacion)):
            q = uniform(0, 1)
            if q >= 0.5:
                for j in range(dim):
                    r2 = uniform(0, 1)
                    r4 = uniform(0, 1)

                    a = 2 * cos(2 * pi * r2) * t
                    a_may = (2 * r4 - 1) * a

                    u1 = uniform(-a, a)
                    u2 = a_may * (poblacion[i][j] - individuo_random[j])

                    # Ecuacion 7a
                    mx[i][j] = poblacion[i][j] + u1 + u2
            else:
                for j in range(dim):
                    r3 = uniform(0, 1)
                    r5 = uniform(0, 1)

                    b = 2 * v(2 * pi * r3) * t
                    b_may = (2 * r5 - 1) * b

                    v1 = uniform(-b, b)
                    v2 = b_may * (poblacion[i][j] - individuo_promedio[j])
                    # Ecuacion 7b
                    mx[i][j] = poblacion[i][j] + v1 + v2

    # ========= Explotacion =========
    else:
        t2 = 1 + (iter / maxIter)
        for i in range(len(poblacion)):
            r6 = uniform(0, 1)
            l = 0.2 + (2 - 0.2) * r6
            r = (M * VEL**2) / l
            capturability = 1 / (r * t2)
            # Caso ajustes exitosos
            if capturability >= C:
                for j in range(dim):
                    delta = capturability * abs(poblacion[i][j] - individuo_mejor[j])
                    # Ecuacion 17a
                    mx[i][j] = t * delta * (poblacion[i][j] - individuo_mejor[j]) + poblacion[i][j]

            # Caso movimiento Levy
            else:
                for j in range(dim):
                    p = levy()
                    # Ecuacion 17b
                    mx[i][j] = individuo_mejor[j] - (poblacion[i][j] - individuo_mejor[j]) * p * t
    for i in range(len(mx)):
        if (instance.fitness(poblacion[i]) > fitness[i]):
            poblacion[i] = mx[i].copy()
    return np.array(poblacion)
    # for i in range(n):
    # Calculate fitnes of Mxi
    # if fitnessMx > fintensX:
    # xi = Mxi
    # fitness[i] = instance.fitness(poblacion[i])
