import numpy as np
import matplotlib.pyplot as plt
from ODE_Solver import RK4, solve_ode
from function_examples import *
from Numerical_Shooting import isolate_orbit, shooting, shooting_conditions
from scipy.optimize import fsolve
from pde_solver import pde_solver, forward_euler, backward_euler, crank_nicholson


def natural_parameter(ode, initial_guess, step_size, vary_par_index, vary_range, orbit, args):
    """
    :param ode: Example ODE
    :param initial_guess: The initial guess for isolating an orbit
    :param vary_par_index: The index of the parameter to vary found in args
    :param vary_range: The range of values to vary over
    :param args: The arguments for tbe constants of the ODE
    :return: A list of solutions
    """

    vary_count = int((np.diff(vary_range))/step_size)  # Number of different variable values
    vary_values = np.linspace(vary_range[0], vary_range[1], vary_count)
    args[vary_par_index] = vary_values[1]

    if orbit:
        # Find the initial solution using isolate_orbit()
        times = np.linspace(0, 100, num=1000)  # Range of t_values to find orbit
        RK4_values = np.asarray(solve_ode(times, initial_guess, 0.01, RK4, ode, args))
        init_vals = isolate_orbit(RK4_values, times)

    else:
        # If shooting is not required
        # Find initial solution
        c = vary_range[0]
        init_vals = fsolve(lambda x: ode(0, x, c), np.array([initial_guess]))

    list_of_solutions = [0]*len(vary_values)
    list_of_solutions[0] = init_vals
    for i in range(1, len(vary_values)):
        if orbit:
            args[vary_par_index] = vary_values[i]
            init_vals = shooting(ode, init_vals, [], orbit, args)
            list_of_solutions[i] = init_vals
        else:
            init_vals = fsolve(lambda x: ode(0, x, vary_values[i]), np.array([init_vals]))
            list_of_solutions[i] = init_vals
    return list_of_solutions, vary_values


def pseudo_arclength(ode, initial_guess, step_size, vary_par_index, vary_range, orbit, args):
    """
    Executes a single step of the forward euler method for given value, t_n

    Parameters:
    ----------
        ode : function
            The ODE for which the euler step predicts
        initial_guess : numpy array
            The initial guess to find an orbit
        vary_par_index : integer
            The index of the parameter from args
        vary_range : list
            The lower and upper limit for the values the parameter can take
        args : list
            The parameters of the ODE

    Returns:
    ----------
        x : numpy array
            The new value of the dependant variable after an euler step
    """

    vary_count = int((np.diff(vary_range))/step_size)  # Number of different variable values
    vary_values = np.linspace(vary_range[0], vary_range[1], vary_count)
    args[vary_par_index] = vary_values[1]

    if orbit:
        # Find first initial solution
        times = np.linspace(0, 500, num=2000)  # Range of t_values to find orbit
        RK4_values = np.asarray(solve_ode(times, initial_guess, 0.1, RK4, ode, args))

        #plt.plot(RK4_values)
        #plt.show()

        # First known solution
        u0 = np.array(isolate_orbit(RK4_values, times))
        p0 = vary_values[1]

        # Find second initial solution
        args[vary_par_index] = vary_values[2]
        RK4_values = np.asarray(solve_ode(times, u0[:-1], 0.1, RK4, ode, args))

        # Second known solution
        u1 = np.array(isolate_orbit(RK4_values, times))
        p1 = vary_values[2]
    else:
        p0, p1 = vary_values[1], vary_values[2]
        u0 = np.array(fsolve(lambda x: ode(0, x, p0), np.array([initial_guess])))
        u1 = np.array(fsolve(lambda x: ode(0, x, p1), np.array([initial_guess])))
    state_secant = u1 - u0
    predict_ui = u1 + state_secant
    param_secant = p1 - p0
    predict_pi = p1 + param_secant
    list_of_solutions = [np.append(u0, p0), np.append(u1, p1)]

    if vary_range[1] < vary_range[0]:
        statement = p1 > vary_range[1]

    else:
        statement = p1 < vary_range[1]

    while statement:
        u0 = u1
        p0 = p1
        if orbit:
            init_vals = shooting(ode, np.append(u1, p1), [vary_par_index, predict_ui, state_secant,
                                                          predict_pi, param_secant], orbit, args)
        else:
            init_vals = fsolve(lambda x: shooting_conditions(ode, x, [vary_par_index, predict_ui, state_secant, predict_pi, param_secant], orbit, args=args), np.append(u1, p1))
        # Update current state
        u1 = init_vals[:-1]
        state_secant = u1 - u0
        predict_ui = u1 + state_secant
        p1 = init_vals[-1]
        param_secant = p1 - p0
        predict_pi = p1 + param_secant
        print(init_vals)
        list_of_solutions.append(init_vals)
        if vary_range[1] < vary_range[0]:
            statement = p1 > vary_range[1]
        else:
            statement = p1 < vary_range[1]
    u_values = [item[:-1] for item in list_of_solutions]
    param_values = [item[-1] for item in list_of_solutions]
    return u_values, param_values


def continuation(diff_eq, initial_guess, step_size, vary_par_index, vary_range, orbit, discretisation, method,
                 boundary, L, T, p_func, q_func, plot, args):
    if discretisation == 'pseudo_arclength':
        list_param, param_values = pseudo_arclength(diff_eq, initial_guess, step_size, vary_par_index, vary_range, orbit, args)
        if len(list_param[0]) == 1:
            plt.plot(param_values, list_param)
        else:
            t_remove = 0
            if orbit:
                t_remove = 1
            for i in range(len(list_param[0])-t_remove):
                x_values = [item[i] for item in list_param]
                plt.plot(param_values, x_values)
        plt.show()

    elif discretisation == 'natural_parameter':
        list_param, param_values = natural_parameter(diff_eq, initial_guess, step_size, vary_par_index, vary_range, orbit, args)
        if len(list_param[0]) == 1:
            plt.plot(param_values, list_param)
        else:
            t_remove = 0
            if orbit:
                t_remove = 1
            for i in range(len(list_param[0])-t_remove):
                x_values = [item[i] for item in list_param]
                plt.plot(param_values, x_values)
        plt.show()

    elif discretisation == 'pde_solve':
        # Set problem parameters/functions
        # k = 1.0   # diffusion constant
        # L = 1.0         # length of spatial domain
        # T = 0.5         # total time to solve for

        # Boundary examples
        boundary_cond1 = 'homogenous'
        boundary_cond2 = 'dirichlet'
        boundary_cond3 = 'neumann'
        boundary_cond4 = 'periodic'

        pde_solver(diff_eq, L, T, method, boundary, p_func, q_func, args)

    elif discretisation == 'shooting':
        u_vect = shooting(diff_eq, initial_guess, False, orbit, args)

    elif discretisation == 'ode_solve':
        times = np.linspace(0, T, num=1000)
        u_values = np.asarray(solve_ode(times, initial_guess, step_size, method, diff_eq, args))


if __name__ == '__main__':

    '''
    continuation(hopf_bif, np.array([0.5, 0.5]), 0.1, 0, [0, 2], True, 'natural_parameter', RK4,
                 'n/a', 0, 0, p, q, False, np.array([0], dtype=float))

    continuation(mod_hopf_bif, np.array([0.5, 0.5]), 0.1, 0, [2, -1], True, 'natural_parameter', RK4,
                 'n/a', 0, 0, p, q, False, np.array([0], dtype=float))

    continuation(hopf_bif, np.array([0.5, 0.5]), 0.1, 0, [0, 2], True, 'pseudo_arclength', RK4,
                 'n/a', 0, 0, p, q, False, np.array([0], dtype=float))

    continuation(mod_hopf_bif, np.array([0.5, 0.5]), 0.1, 0, [2, -1], True, 'pseudo_arclength', RK4,
                 'n/a', 0, 0, p, q, False, np.array([0], dtype=float))


    continuation(alg_cubic, np.array([1]), 0.1, 0, [-2, 2], False, 'pseudo_arclength', RK4,
                 'n/a', 0, 0, p, q, False, np.array([0], dtype=float))

    continuation(alg_cubic, np.array([1]), 0.1, 0, [-2, 2], False, 'natural_parameter', RK4,
                 'n/a', 0, 0, p, q, False, np.array([0], dtype=float))
'''


