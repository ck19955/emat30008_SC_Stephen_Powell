import numpy as np
import matplotlib.pyplot as plt
from ODE_Solver import *
from scipy.signal import argrelextrema
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from function_examples import *
from Numerical_Shooting import *


def natural_parameter(ode, initial_guess, vary_par_index, vary_range, args):
    # Find the initial solution using isolate_orbit()
    args[vary_par_index] = vary_range[0]
    vary_count = 50  # Number of different variable values
    vary_values = np.linspace(vary_range[0], vary_range[1], vary_count)
    times = np.linspace(0, 400, num=1000)  # Range of t_values to find orbit
    RK4_values = np.asarray(solve_ode(times, initial_guess, 0.1, RK4, ode, args))
    init_vals = isolate_orbit(RK4_values, times)
    list_of_solutions = [init_vals]
    for i in vary_values:
        # vary_par += vary_par*delta
        print(i)
        args[vary_par_index] = i
        init_vals = shooting(ode, init_vals, args)
        list_of_solutions.append(init_vals)

    return list_of_solutions
    # Isolate an orbit with specific alpha value
    # Find the initial conditions needed
    # Change alpha by small increment delta
    # Assume the intial conditions for new alpha are the same as the intital values given by previous solution
    # Find the 'actual' initial conditions for the new value of alpha using shooter
    # Repeat until all the alpha values are covered


def pseudo_arclength():
    return


list_param = natural_parameter(pred_prey, np.array([1, 1]), 1, [0.1, 0.25], np.array([1, 0.1, 0.1]))
print(list_param)