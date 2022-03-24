import numpy as np
import matplotlib.pyplot as plt
from ODE_Solver import *
from scipy.signal import argrelextrema
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from function_examples import *
from Numerical_Shooting import *


def natural_parameter():
    # Isolate an orbit with specific alpha value
    # Find the initial conditions needed
    # Change alpha by small increment delta
    # Assume the intial conditions for new alpha are the same as the intital values given by previous solution
    # Find the 'actual' initial conditions for the new value of alpha
    # Repeat until all the alpha values are covered
    return


def pseudo_arclength():
    return




