import numpy as np
import matplotlib.pyplot as plt
from ode_solver import rk4, solve_ode
from function_examples import *
from numerical_shooting import isolate_orbit, shooting, shooting_conditions, pseudo_arclength_conditions
from scipy.optimize import fsolve
from pde_solver import pde_solver, forward_euler, backward_euler, crank_nicholson


def natural_parameter(ode, initial_guess, step_size, vary_par_index, vary_range, orbit, args):
    """
    Natural parameter continuation finds solutions or initial values of orbits. The solutions are found for parameter
    values between the range specified.

    Parameters:
    ----------
        ode : function
            ODE to find solution for
        initial_guess : array
            Initial guess for the solution
        step_size : float
            The step size to vary between parameter values
        vary_par_index : integer
            The index of the parameter to vary found in args
        vary_range : tuple
            The range of parameter values to vary over
        orbit : Boolean
            Decides whether an orbit is being found or not
        args : numpy array
            The parameters of the ODE


    Returns:
    ----------
        list_of_solutions : list
            List of solutions of the ODE for each parameter value
        parameter_values : list
            List of parameter values for each solution found
    """

    vary_count = int(abs((np.diff(vary_range))) / step_size)  # Number of different variable values
    parameter_values = np.linspace(vary_range[0], vary_range[1], vary_count)
    # List of parameter values to find solutions for

    # Set the new value to the desired parameter to vary
    args[vary_par_index] = parameter_values[0]

    if orbit:
        # Find the initial solution using isolate_orbit()
        times = np.linspace(0, 100, num=1000)  # Range of t_values to find orbit
        rk4_values = np.asarray(solve_ode(times, initial_guess, 0.01, rk4, ode, args))
        init_solution = isolate_orbit(rk4_values, times)

    else:
        # If shooting is not required
        # Find the initial solution using fsolve()
        args[vary_par_index] = vary_range[0]
        init_solution = fsolve(lambda x: ode(0, x, args), np.array([initial_guess]))

    # Set up an list of zeros
    list_of_solutions = [0] * len(parameter_values)
    list_of_solutions[0] = init_solution

    for i in range(1, len(parameter_values)):
        # Set the new value to the desired parameter to vary
        args[vary_par_index] = parameter_values[i]
        if orbit:
            # Find the orbit using the previous solution
            init_solution = shooting(ode, init_solution, shooting_conditions, [], orbit, args)
        else:
            # If shooting is not required find the initial solution using fsolve()
            init_solution = fsolve(lambda x: ode(0, x, args), np.array([init_solution]))
        list_of_solutions[i] = init_solution

    # If list_of_solutions has only one dimension then it is a first order ODE where solutions have been found
    if len(list_of_solutions[0]) == 1:
        plt.plot(parameter_values, list_of_solutions)
    else:
        t_remove = 0
        if orbit:
            t_remove = 1  # If orbits are being found then the period needs to be removed for plotting
        for i in range(len(list_of_solutions[0]) - t_remove):
            x_values = [item[i] for item in list_of_solutions]  # Code is generalised for different dimensions
            plt.plot(parameter_values, x_values)
    plt.ylabel("u", fontsize=14)
    plt.xlabel("Parameter Value", fontsize=14)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.show()

    return list_of_solutions, parameter_values


def pseudo_arclength(ode, initial_guess, step_size, vary_par_index, vary_range, orbit, args):
    """
    Pseudo-arclength continuation finds solutions or initial values of orbits. The solutions are found for parameter
    values between the range specified.

    Parameters:
    ----------
        ode : function
            ODE to find solution for
        initial_guess : array
            Initial guess for the solution
        step_size : float
            The step size to vary between parameter values
        vary_par_index : integer
            The index of the parameter to vary found in args
        vary_range : tuple
            The range of parameter values to vary over
        orbit : Boolean
            Decides whether an orbit is being found or not
        args : numpy array
            The parameters of the ODE


    Returns:
    ----------
     u_values : list
         List of solutions of the ODE for each parameter value
     param_values : list
         List of parameter values for each solution found
    """
    vary_count = int(abs((np.diff(vary_range))) / step_size)  # Number of different variable values
    vary_values = np.linspace(vary_range[0], vary_range[1], vary_count)
    args[vary_par_index] = vary_values[0]

    if orbit:
        # Find first initial solution
        times = np.linspace(0, 500, num=2000)  # Range of t_values to find orbit
        rk4_values = np.asarray(solve_ode(times, initial_guess, 0.1, rk4, ode, args))


        # First known solution
        u0 = np.array(isolate_orbit(rk4_values, times))
        p0 = vary_values[0]

        # Find second initial solution
        args[vary_par_index] = vary_values[1]
        rk4_values = np.asarray(solve_ode(times, u0[:-1], 0.1, rk4, ode, args))

        # Second known solution
        u1 = np.array(isolate_orbit(rk4_values, times))
        p1 = vary_values[1]
    else:
        # If shooting is not required then find the two initial solutions using fsolve()
        p0, p1 = vary_values[0], vary_values[1]
        u0 = np.array(fsolve(lambda x: ode(0, x, p0), np.array([initial_guess])))
        u1 = np.array(fsolve(lambda x: ode(0, x, p1), np.array([initial_guess])))

    # Calculate the different parts required for the pseudo-arclength equation
    state_secant = u1 - u0
    predict_ui = u1 + state_secant
    param_secant = p1 - p0
    predict_pi = p1 + param_secant
    list_of_solutions = [np.append(u0, p0), np.append(u1, p1)]

    # Depending on whether the parameter values starts off smaller or larger than the final parameter value, the
    # current parameter value must be between the upper and lower limit.
    if vary_range[1] < vary_range[0]:
        statement = vary_range[0] > p1 > vary_range[1]
    else:
        statement = vary_range[0] < p1 < vary_range[1]

    while statement:
        # Set the new initial solution and parameter values
        u0 = u1
        p0 = p1
        if orbit:
            # If shooting is required
            init_vals = shooting(ode, np.append(u1, p1), pseudo_arclength_conditions, [vary_par_index, predict_ui, state_secant,
                                                          predict_pi, param_secant], orbit, args)
        else:
            # If shooting is not required, use fsolve()
            init_vals = fsolve(lambda x: pseudo_arclength_conditions(ode, x,
                                                             [vary_par_index, predict_ui, state_secant, predict_pi,
                                                              param_secant], orbit, args=args), np.append(u1, p1))
        # Update current state
        u1 = init_vals[:-1]
        state_secant = u1 - u0
        predict_ui = u1 + state_secant
        p1 = init_vals[-1]
        param_secant = p1 - p0
        predict_pi = p1 + param_secant

        # Use append to the list of solutions. Usually append is bad due to the danger of inefficiently using memory.
        # However, in this case it is difficult to create an empty array of the correct size because pseudo-arclength
        # iterates in an non-linear manner.
        list_of_solutions.append(init_vals)

        # Check if the new parameter remains in the range
        if vary_range[1] < vary_range[0]:
            statement = vary_range[0] > p1 > vary_range[1]
        else:
            statement = vary_range[0] < p1 < vary_range[1]

    # Extract the list of initial values and parameter values from the list of solutions
    u_values = [item[:-1] for item in list_of_solutions]
    param_values = [item[-1] for item in list_of_solutions]

    # If the list of initial values is one dimensional then a first order ODE has been solved without the need of
    # finding orbits. This can be plotted without any more coding.
    if len(u_values[0]) == 1:
        plt.plot(param_values, u_values)
    else:
        t_remove = 0
        if orbit:
            t_remove = 1  # If orbits have been found, then the period of the orbits should be removed when plotting
        for i in range(len(u_values[0]) - t_remove):
            x_values = [item[i] for item in u_values]  # Code is generalised for different dimensions
            plt.plot(param_values, x_values)
    plt.ylabel("u", fontsize=14)
    plt.xlabel("Parameter Value", fontsize=14)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.show()

    return u_values, param_values


def pde_continuation(pde, step_size, vary_par_index, vary_range, final_space_value, final_time_value, num_of_x,
                     num_of_t, method, boundary_cond, p_func, q_func, args):
    """
    PDE continuation finds the steady states of a PDE for varying parameter values

    Parameters:
    ----------
        pde : function
            PDE to find solution for
        step_size : float
            The step size to vary between parameter values
        vary_par_index : integer
            The index of the parameter to vary found in args
        vary_range : tuple
            The range of parameter values to vary over
        final_space_value : integer
            The final space value to consider for the matrix of solutions
        final_time_value : integer
            The final time value to consider for the matrix of solutions
        num_of_x : integer
            The number of data points in space
        num_of_t : integer
            The number of data points in time
        method : function
            The pde discretisation used to solve the PDE
        boundary_cond : string
            The type of boundary condition
        p_func : function
            Function of the left boundary condition
        q_func : function
            Function of the right boundary con
        args : numpy array
            The parameters of the PDE


    Returns:
    ----------
     list_of_solutions : list
         List of solutions of the PDE for each parameter value
     param_values : list
         List of parameter values for each solution found
    """

    vary_count = int((np.diff(vary_range)) / step_size)  # Number of different variable values

    # Set up lists for solutions
    param_values = np.linspace(vary_range[0], vary_range[1], vary_count)
    list_of_solutions = [0] * len(param_values)
    for i in range(len(param_values)):
        # 'vary_p' and 'vary_q' refer to the boundary conditions being varied
        if boundary_cond == 'vary_p':
            solution_matrix = pde_solver(pde, final_space_value, final_time_value, num_of_x, num_of_t, method, boundary_cond, lambda x: param_values[i], q_func, args)
        elif boundary_cond == 'vary_q':
            solution_matrix = pde_solver(pde, final_space_value, final_time_value, num_of_x, num_of_t, method, boundary_cond, p_func, lambda x: param_values[i], args)
        else:
            # Vary the heat diffusion constant
            args[vary_par_index] = param_values[i]
            solution_matrix = pde_solver(pde, final_space_value, final_time_value, num_of_x, num_of_t, method, boundary_cond, p_func, q_func, args)
        list_of_solutions[i] = solution_matrix[-1]

    # If the thermal diffusion constant is being varied, plot normally
    if boundary_cond != 'vary_p' and 'vary_q':
        plt.plot(list_of_solutions)
        plt.show()
    else:
        plt.plot(list_of_solutions)
        plt.show()
        solution_matrix = pde_solver(pde, final_space_value, final_time_value, num_of_x, num_of_t, method, boundary_cond, p_func, q_func, args)
        plt.plot(solution_matrix)
        plt.show()
    return list_of_solutions, param_values


if __name__ == '__main__':

    # Plot the bifurcation diagrams for the cubic equation
    natural_parameter(alg_cubic, np.array([1]), 0.1, 0, [-2, 2], False, np.array([0], dtype=float))
    pseudo_arclength(alg_cubic, np.array([1]), 0.1, 0, [-2, 2], False, np.array([0], dtype=float))

    # Plot the bifurcation diagrams for the hopf bifurcation diagram
    natural_parameter(hopf_bif, np.array([0.5, 0.5]), 0.1, 0, [2, 0], True, np.array([0], dtype=float))
    pseudo_arclength(hopf_bif, np.array([0.5, 0.5]), 0.1, 0, [2, 0], True, np.array([0], dtype=float))

    # Plot the bifurcation diagrams for the modified hopf bifurcation diagram
    natural_parameter(mod_hopf_bif, np.array([0.5, 0.5]), 0.1, 0, [2, -1], True, np.array([0], dtype=float))
    pseudo_arclength(mod_hopf_bif, np.array([0.5, 0.5]), 0.1, 0, [2, -1], True, np.array([0], dtype=float))

    # Plot the solutions for varying the boundary condition
    pde_continuation(u_I, 0.1, 0, [3, 3.5], 1, 0.5, 10, 1000, forward_euler, 'vary_p', p, q, np.array([3], dtype=float))
