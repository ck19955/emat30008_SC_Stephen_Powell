import numpy as np
import matplotlib.pyplot as plt
from ODE_Solver import *


def ode_num(t,x_values,a,b,d):
    x = x_values[0]
    y = x_values[1]
    x_array = np.array([(x)*(1-x) - (a*x*y)/(d+x), b*y*(1-(y/x))])
    return x_array



