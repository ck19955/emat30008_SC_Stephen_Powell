import math
import numpy as np


def ode_first_order(t, u):
    return u


def ode_second_order(t, u):
    u_array = np.array([u[1], -u[0]])
    return u_array


def exact_second_order(t, u):
    x = u[0]
    y = u[1]
    u_array = np.array([x*math.cos(t) + y*math.sin(t), -x*math.sin(t) + y*math.cos(t)])
    return u_array


def exponential(t, u):
    return math.exp(t)


def pred_prey(t, u_values, a, b, d):
    x = u_values[0]
    y = u_values[1]
    u_array = np.array([x*(1-x) - (a*x*y)/(d+x), b*y*(1-(y/x))])
    return u_array


def hopf_bif(t, u_values, beta):
    x = u_values[0]
    y = u_values[1]
    u_array = np.array([beta*x - y - x*(x**2 + y**2), x + beta*y - y*(x**2 + y**2)])
    return u_array


def exact_hopf_bif(t, u_values, beta, theta):
    u_array = np.array([(beta**0.5)*math.cos(t + theta), (beta**0.5)*math.sin(t + theta)])
    return u_array


def alg_cubic(t, x, c):
    return x**3 - x + c


def mod_hopf_bif(t, u_values, beta):
    x = u_values[0]
    y = u_values[1]
    u_array = np.array([beta*x - y + x*(x**2 + y**2) - x*(x**2 + y**2)**2,
                        x + beta*y + y*(x**2 + y**2) - y*(x**2 + y**2)**2])
    return u_array

