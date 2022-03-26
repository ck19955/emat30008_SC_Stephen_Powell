import numpy as np
import matplotlib.pyplot as plt
from ODE_Solver import *
from scipy.signal import argrelextrema
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from function_examples import *
from Numerical_Shooting import *


def natural_parameter(ode, initial_guess, vary_par_index, vary_range, args):
    """
    :param ode: Example ODE
    :param initial_guess: The initial guess for isolating an orbit
    :param vary_par_index: The index of the parameter to vary found in args
    :param vary_range: The range of values to vary over
    :param args: The arguments for tbe constants of the ODE
    :return: A list of solutions
    """
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
        init_vals = shooting(ode, init_vals, [], args)
        list_of_solutions.append(init_vals)

    return list_of_solutions
    # Isolate an orbit with specific alpha value
    # Find the initial conditions needed
    # Change alpha by small increment delta
    # Assume the intial conditions for new alpha are the same as the intital values given by previous solution
    # Find the 'actual' initial conditions for the new value of alpha using shooter
    # Repeat until all the alpha values are covered


def pseudo_arclength(ode, initial_guess, vary_par_index, vary_range, args):
    # Find first initial solution
    args[vary_par_index] = vary_range[0]
    vary_count = 50  # Number of different variable values
    vary_values = np.linspace(vary_range[0], vary_range[1], vary_count)
    times = np.linspace(0, 400, num=1000)  # Range of t_values to find orbit
    RK4_values = np.asarray(solve_ode(times, initial_guess, 0.1, RK4, ode, args))
    solution_0 = isolate_orbit(RK4_values, times)
    u0 = np.array(solution_0)
    p0 = vary_range[0]

    # Find second initial solution
    args[vary_par_index] = vary_values[1]
    RK4_values = np.asarray(solve_ode(times, np.array(solution_0)[:-1], 0.1, RK4, ode, args))
    solution_1 = isolate_orbit(RK4_values, times)
    u1 = np.array(solution_1)
    p1 = vary_values[1]
    state_secant = u1 - u0
    predict_ui = u1 + state_secant
    param_secant = p1 - p0
    predict_pi = p1 + param_secant
    while p1 < vary_range[1]:
        u0 = u1
        p0 = p1
        init_vals = shooting(ode, np.append(np.array(solution_1), p1), [1, predict_ui, state_secant,
                                                                        predict_pi, param_secant], args)
        print(init_vals)
        u1 = init_vals[:-1]
        state_secant = u1 - u0
        predict_ui = u1 + state_secant

        p1 = init_vals[-1]
        param_secant = p1 - p0
        predict_pi = p1 + param_secant

    return


# list_param = natural_parameter(pred_prey, np.array([1, 1]), 1, [0.1, 0.25], np.array([1, 0.1, 0.1]))
pseudo_arclength(pred_prey, np.array([1, 1]), 1, [0.1, 0.25], np.array([1, 0.1, 0.1]))
# print(list_param)
