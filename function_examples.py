import math
import numpy as np


def ode(t, x):
    return x


def ode_second_order(t, x):
    x_array = np.array([x[1], -x[0]])
    return x_array


def exact_second_order(t, x):
    x = x[0]
    y = x[1]
    x_array = np.array([x*math.cos(t) + y*math.sin(t), -x*math.sin(t) + y*math.cos(t)])
    return x_array


def exponential(t, x):
    return math.exp(t)


def pred_prey(t, x_values, a, b, d):
    x = x_values[0]
    y = x_values[1]
    x_array = np.array([x*(1-x) - (a*x*y)/(d+x), b*y*(1-(y/x))])
    return x_array


def hopf_bif(t, x_values, beta, sigma):
    x = x_values[0]
    y = x_values[1]
    sigma = -1
    x_array = np.array([beta*x - y + sigma*x*(x**2 + y**2), x + beta*y + sigma*y*(x**2 + y**2)])
    return x_array


def exact_hopf_bif(t, x_values, beta, theta):
    x_array = np.array([(beta**0.5)*math.cos(t + theta), (beta**0.5)*math.sin(t + theta)])
    return x_array
