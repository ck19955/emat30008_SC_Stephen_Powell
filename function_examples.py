import math
import numpy as np


def ode(t, x):
    """
    The ode() function gives the equation of the ODE that will be analysed

    Parameters:
        t         - The value of the independent variable
        x         - The value of the dependant variable

    Returns:
        x           - The value of the differential of the dependant variable
    """

    return x


def ode_second_order(t, x):
    x_array = np.array([x[1], -x[0]])
    return x_array


def exact(t, x):
    """
    The exact() function calculates the exact value of the dependant variable given the value of x and t

    Parameters:
        t             - The value of the independent variable
        x             - The value of the dependant variable

    Returns:
        The exact value of the dependant variable using analytical methods
    """

    a = x[0]
    b = x[1]
    return np.array([a*math.cos(t) + b*math.sin(t), -a*math.sin(t) + b*math.cos(t)])


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
