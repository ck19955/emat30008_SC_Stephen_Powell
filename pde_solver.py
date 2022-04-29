import numpy as np
from math import pi
import matplotlib.pyplot as plt
from function_examples import *


def forward_euler(pde, final_space_value, lmbda, num_of_x, num_of_t, bound_cond, p_func, q_func):
    """
    Solves a given pde using the forward euler method

    Parameters:
    ----------
        pde : function
            The PDE to be solved
        final_space_value : float
            The length of the spatial domain
        lmbda : float
            Mesh fourier number
        num_of_x : integer
            Number of gridpoints in space
        num_of_t : integer
            Number of gridpoints in time
        bound_cond : string
            boundary condition type for the PDE
        p_func : function
            The value at u(0, t)
        q_func : function
            The value at u(L, t)

    Returns:
    ----------
        solution_matrix : numpy array
            Matrix of all data points in the space at any given time
    """

    # Check whether forward_euler method is suitable
    if not 0 < lmbda < 1/2:
        raise ValueError("Invalid value for lmbda")
    # This is used to avoid getting invalid values for the forward Euler method

    # Evaluate initial solution values
    u_vect = np.linspace(0, final_space_value, num_of_x + 1)
    for i in range(num_of_x + 1):
        u_vect[i] = pde(u_vect[i], final_space_value)

    # Initialise the matrix of solutions with the initial solution values
    solution_matrix = np.zeros((num_of_t, num_of_x + 1))
    solution_matrix[0] = u_vect

    # Check boundary conditions
    if bound_cond == 'dirichlet' or bound_cond == 'vary_p' or bound_cond == 'vary_q':
        A = np.diag([1-2*lmbda] * (num_of_x - 1)) + np.diag([lmbda] * (num_of_x - 2), -1) + np.diag([lmbda] * (num_of_x - 2), 1)
        additive_vector = np.zeros(num_of_x - 1)

    elif bound_cond == 'neumann':
        A = np.diag([1-2*lmbda] * (num_of_x + 1)) + np.diag([lmbda] * num_of_x, -1) + np.diag([lmbda] * num_of_x, 1)
        A[0, 1] = 2*A[0, 1]
        A[-1, -2] = 2*A[-1, -2]
        deltax = final_space_value / num_of_x
        additive_vector = np.zeros(num_of_x + 1)

    elif bound_cond == 'periodic':
        A = np.diag([1-2*lmbda] * num_of_x) + np.diag([lmbda] * (num_of_x - 1), -1) + np.diag([lmbda] * (num_of_x - 1), 1)
        A[0, -1] = lmbda
        A[-1, 0] = lmbda
        solution_matrix = np.zeros((num_of_t, num_of_x))
        solution_matrix[0] = u_vect[:-1]

    elif bound_cond == 'homogenous':
        A = np.diag([1-2*lmbda] * (num_of_x - 1)) + np.diag([lmbda] * (num_of_x - 2), -1) + np.diag([lmbda] * (num_of_x - 2), 1)

    # Iterate over time values
    for i in range(0, num_of_t - 1):
        if bound_cond == 'dirichlet':
            additive_vector[0] = p_func(i)
            additive_vector[-1] = q_func(i)
            solution_matrix[i+1][1:-1] = np.dot(A, solution_matrix[i][1:-1]) + lmbda*additive_vector
            solution_matrix[i+1][0] = additive_vector[0]
            solution_matrix[i+1][-1] = additive_vector[-1]
        elif bound_cond == 'neumann':
            additive_vector[0] = -p_func(i)
            additive_vector[-1] = q_func(i)
            solution_matrix[i+1] = np.dot(A, solution_matrix[i]) + 2*deltax*lmbda*additive_vector

        elif bound_cond == 'periodic':
            solution_matrix[i+1] = np.dot(A, solution_matrix[i])

        else:
            solution_matrix[i+1][1:-1] = np.dot(A, solution_matrix[i][1:-1])

    return solution_matrix


def backward_euler(pde, final_space_value, lmbda, num_of_x, num_of_t, bound_cond, p_func, q_func):
    """
    Solves a given pde using the backward euler method

    Parameters:
    ----------
        pde : function
            The PDE to be solved
        final_space_value : float
            The length of the spatial domain
        lmbda : float
            Mesh fourier number
        num_of_x : integer
            Number of gridpoints in space
        num_of_t : integer
            Number of gridpoints in time
        bound_cond : string
            boundary condition type for the PDE
        p_func : function
            The value at u(0, t)
        q_func : function
            The value at u(L, t)

    Returns:
    ----------
        solution_matrix : numpy array
            Matrix of all data points in the space at any given time
    """

    # Evaluate initial solution values
    u_vect = np.linspace(0, final_space_value, num_of_x + 1)
    for i in range(num_of_x + 1):
        u_vect[i] = pde(u_vect[i], final_space_value)

    # Initialise the matrix of solutions with the initial solution values
    solution_matrix = np.zeros((num_of_t, num_of_x + 1))
    solution_matrix[0] = u_vect

    # Check boundary conditions
    if bound_cond == 'dirichlet':
        A = np.diag([1+2*lmbda] * (num_of_x - 1)) + np.diag([-lmbda] * (num_of_x - 2), -1) + np.diag([-lmbda] * (num_of_x - 2), 1)
        additive_vector = np.zeros(num_of_x - 1)

    elif bound_cond == 'neumann':
        A = np.diag([1+2*lmbda] * (num_of_x + 1)) + np.diag([-lmbda] * num_of_x, -1) + np.diag([-lmbda] * num_of_x, 1)
        A[0, 1] = 2*A[0, 1]
        A[-1, -2] = 2*A[-1, -2]
        deltax = final_space_value / num_of_x
        additive_vector = np.zeros(num_of_x + 1)

    elif bound_cond == 'periodic':
        A = np.diag([1+2*lmbda] * num_of_x) + np.diag([-lmbda] * (num_of_x - 1), -1) + np.diag([-lmbda] * (num_of_x - 1), 1)
        A[0, -1] = -lmbda
        A[-1, 0] = -lmbda
        solution_matrix = np.zeros((num_of_t, num_of_x))
        solution_matrix[0] = u_vect[:-1]

    else:
        A = np.diag([1+2*lmbda] * (num_of_x - 1)) + np.diag([-lmbda] * (num_of_x - 2), -1) + np.diag([-lmbda] * (num_of_x - 2), 1)

    # Iterate over time values
    for i in range(0, num_of_t - 1):
        if bound_cond == 'dirichlet':
            additive_vector[0] = p_func(i)
            additive_vector[-1] = q_func(i)
            solution_matrix[i+1][1:-1] = np.linalg.solve(A, solution_matrix[i][1:-1] + lmbda*additive_vector)
            solution_matrix[i+1][0] = additive_vector[0]
            solution_matrix[i+1][-1] = additive_vector[-1]
        elif bound_cond == 'neumann':
            additive_vector[0] = -p_func(i)
            additive_vector[-1] = q_func(i)
            solution_matrix[i+1] = np.linalg.solve(A, solution_matrix[i] + 2*deltax*lmbda*additive_vector)
        elif bound_cond == 'periodic':
            solution_matrix[i+1] = np.linalg.solve(A, solution_matrix[i])
        else:
            solution_matrix[i+1][1:-1] = np.linalg.solve(A, solution_matrix[i][1:-1])
    return solution_matrix


def crank_nicholson(pde, final_space_value, lmbda, num_of_x, num_of_t, bound_cond, p_func, q_func):
    """
    Solves a given pde using the Crank Nicholson method

    Parameters:
    ----------
        pde : function
            The PDE to be solved
        final_space_value : float
            The length of the spatial domain
        lmbda : float
            Mesh fourier number
        num_of_x : integer
            Number of gridpoints in space
        num_of_t : integer
            Number of gridpoints in time
        bound_cond : string
            boundary condition type for the PDE
        p_func : function
            The value at u(0, t)
        q_func : function
            The value at u(L, t)

    Returns:
    ----------
        solution_matrix : numpy array
            Matrix of all data points in the space at any given time
    """

    # Evaluate initial solution values
    u_vect = np.linspace(0, final_space_value, num_of_x + 1)
    for i in range(num_of_x + 1):
        u_vect[i] = pde(u_vect[i], final_space_value)

    # Initialise the matrix of solutions with the initial solution values
    solution_matrix = np.zeros((num_of_t, num_of_x + 1))
    solution_matrix[0] = u_vect

    # Check boundary conditions
    if bound_cond == 'dirichlet':
        A = np.diag([1+lmbda] * (num_of_x - 1)) + np.diag([-lmbda / 2] * (num_of_x - 2), -1) + np.diag([-lmbda / 2] * (num_of_x - 2), 1)
        B = np.diag([1-lmbda] * (num_of_x - 1)) + np.diag([lmbda / 2] * (num_of_x - 2), -1) + np.diag([lmbda / 2] * (num_of_x - 2), 1)
        additive_vector = np.zeros(num_of_x - 1)

    elif bound_cond == 'neumann':
        A = np.diag([1+lmbda] * (num_of_x + 1)) + np.diag([-lmbda / 2] * num_of_x, -1) + np.diag([-lmbda / 2] * num_of_x, 1)
        B = np.diag([1-lmbda] * (num_of_x + 1)) + np.diag([lmbda / 2] * num_of_x, -1) + np.diag([lmbda / 2] * num_of_x, 1)
        A[0, 1] = 2*A[0, 1]
        A[-1, -2] = 2*A[-1, -2]
        B[0, 1] = 2*B[0, 1]
        B[-1, -2] = 2*B[-1, -2]
        deltax = final_space_value / num_of_x
        additive_vector = np.zeros(num_of_x + 1)

    elif bound_cond == 'periodic':
        A = np.diag([1+lmbda] * num_of_x) + np.diag([-lmbda / 2] * (num_of_x - 1), -1) + np.diag([-lmbda / 2] * (num_of_x - 1), 1)
        B = np.diag([1-lmbda] * num_of_x) + np.diag([lmbda / 2] * (num_of_x - 1), -1) + np.diag([lmbda / 2] * (num_of_x - 1), 1)
        A[0, -1] = -lmbda/2
        A[-1, 0] = -lmbda/2
        B[0, -1] = lmbda/2
        B[-1, 0] = lmbda/2
        solution_matrix = np.zeros((num_of_t, num_of_x))
        solution_matrix[0] = u_vect[:-1]

    else:
        A = np.diag([1+lmbda] * (num_of_x - 1)) + np.diag([-lmbda / 2] * (num_of_x - 2), -1) + np.diag([-lmbda / 2] * (num_of_x - 2), 1)
        B = np.diag([1-lmbda] * (num_of_x - 1)) + np.diag([lmbda / 2] * (num_of_x - 2), -1) + np.diag([lmbda / 2] * (num_of_x - 2), 1)

    # Iterate over time values
    for i in range(0, num_of_t - 1):
        if bound_cond == 'dirichlet':
            additive_vector[0] = p_func(i)
            additive_vector[-1] = q_func(i)
            solution_matrix[i+1][1:-1] = np.linalg.solve(A, np.dot(B, solution_matrix[i][1:-1]) + lmbda*additive_vector)
            solution_matrix[i+1][0] = additive_vector[0]
            solution_matrix[i+1][-1] = additive_vector[-1]
        elif bound_cond == 'neumann':
            additive_vector[0] = -p_func(i)
            additive_vector[-1] = q_func(i)
            solution_matrix[i+1] = np.linalg.solve(A, np.dot(B, solution_matrix[i]) +
                                                   2*deltax*lmbda*additive_vector)
        elif bound_cond == 'periodic':
            solution_matrix[i+1] = np.linalg.solve(A, np.dot(B, solution_matrix[i]))
        else:
            solution_matrix[i+1][1:-1] = np.linalg.solve(A, np.dot(B, solution_matrix[i][1:-1]))
    return solution_matrix


def pde_solver(pde, final_space_value, final_time_value, num_of_x, num_of_t, method, bound_cond, p_func, q_func, args):
    """
    Uses a given finite difference method to solve a given PDE

    Parameters:
    ----------
        pde : function
            The PDE to be solved
        final_space_value : float
            The length of the spatial domain
        final_time_value : float
            The length of the time domain
        lmbda : float
            Mesh fourier number
        num_of_x : integer
            Number of gridpoints in space
        num_of_t : integer
            Number of gridpoints in time
        method : function
            The method to solve the PDE
        bound_cond : string
            boundary condition type for the PDE
        p_func : function
            The value at u(0, t)
        q_func : function
            The value at u(L, t)
        args : list
            arguments required for given PDE (E.g for heat diffusion this includes kappa)
    Returns:
    ----------
        u_vect : numpy array
            Final solution vector at time, T
    """

    # Set up the numerical environment variables
    x_val_array = np.linspace(0, final_space_value, num_of_x + 1)     # mesh points in space
    t_val_array = np.linspace(0, final_time_value, num_of_t + 1)     # mesh points in time
    deltax = x_val_array[1] - x_val_array[0]            # gridspacing in x
    deltat = t_val_array[1] - t_val_array[0]            # gridspacing in t
    lmbda = args[0]*deltat/(deltax**2)    # mesh fourier number

    # Solve PDE
    solution_matrix = method(pde, final_space_value, lmbda, num_of_x, num_of_t, bound_cond, p_func, q_func)
    return solution_matrix


def pde_error_plot(pde, final_space_value, final_time_value, thermal_const, mx_range, mt_range, number_of_runs, bound_cond, p_func,
                   q_func, args):
    """
    Plots the error of the different pde solvers (not including forward euler) as the number of data points vary

    Parameters:
    ----------
        pde : function
            The PDE to be solved
        final_space_value : float
            The length of the spatial domain
        final_time_value : float
            The length of the time domain
        thermal_const : float
            Thermal diffusion constant
        mx_range : tuple
            Range for the number of space data point values
        mt_range : tuple
            Range for the number of time data point values
        number_of_runs : Integer
            The number of runs to consider with the range of data values
        bound_cond : string
            boundary condition type for the PDE
        p_func : function
            The value at u(0, t)
        q_func : function
            The value at u(L, t)
        args : list
            arguments required for given PDE (E.g for heat diffusion this includes kappa)
    """
    # Create lists of values to evaluate over
    mx_values = np.linspace(mx_range[0], mx_range[1], number_of_runs)
    mt_values = np.linspace(mt_range[0], mt_range[1], number_of_runs)

    # Create empty lists for the errors
    backward_error_mx = np.zeros(len(mx_values))
    crank_error_mx = np.zeros(len(mx_values))
    backward_error_mt = np.zeros(len(mt_values))
    crank_error_mt = np.zeros(len(mt_values))

    # Vary mx values
    for i in range(len(mx_values)):
        x_data = int(mx_values[i])
        t_data = 2000  # Set t_data to be large so that accurate steady states are found

        # Find the exact solution
        exact_solution = np.zeros(x_data + 1)
        for j in range(x_data+1):
            exact_solution[j] = u_exact(j * final_space_value / x_data, final_time_value, thermal_const, final_space_value)

        # Create solution a matrix for each method
        backward_sol = pde_solver(pde, final_space_value, final_time_value, x_data, t_data, backward_euler, bound_cond, p_func, q_func, args)
        crank_sol = pde_solver(pde, final_space_value, final_time_value, x_data, t_data, crank_nicholson, bound_cond, p_func, q_func, args)

        # Calculate the mean squared error of the methods
        backward_error_mx[i] = abs(np.mean(exact_solution - backward_sol[-1]))
        crank_error_mx[i] = abs(np.mean(exact_solution - crank_sol[-1]))

    plt.plot(mx_values, backward_error_mx, label='Backward Euler')
    plt.plot(mx_values, crank_error_mx, label='Crank Nicholson')
    plt.legend()
    plt.xlabel('Number of space data points', fontsize=14)
    plt.ylabel('Mean Square Error', fontsize=14)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.show()

    # Vary mt values
    for i in range(len(mt_values)):
        t_data = int(mt_values[i])
        x_data = 100 # Set x_data to be large so that accurate steady states are found

        # Find the exact solution
        exact_solution = np.zeros(x_data + 1)
        for j in range(x_data+1):
            exact_solution[j] = u_exact(j * final_space_value / x_data, final_time_value, thermal_const, final_space_value)

        # Create solution a matrix for each method
        backward_sol = pde_solver(pde, final_space_value, final_time_value, x_data, t_data, backward_euler, bound_cond, p_func, q_func, args)
        crank_sol = pde_solver(pde, final_space_value, final_time_value, x_data, t_data, crank_nicholson, bound_cond, p_func, q_func, args)

        # Calculate the mean squared error of the methods
        backward_error_mt[i] = abs(np.mean(exact_solution - backward_sol[-1]))
        crank_error_mt[i] = abs(np.mean(exact_solution - crank_sol[-1]))

    plt.plot(mt_values, backward_error_mt, label='Backward Euler')
    plt.plot(mt_values, crank_error_mt, label='Crank Nicholson')
    plt.legend()
    plt.xlabel('Number of time data points', fontsize=14)
    plt.ylabel('Mean Square Error', fontsize=14)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.show()


def pde_solution_plot(solution, exact, label):
    """
    Plots the error of the different pde solvers (not including forward euler) as the number of data points vary

    Parameters:
    ----------
        solution : array
            The steady state calculated
        exact : array
            The exact solution for the steady state
        label : string
            The name of the pde discretisation method used
    """

    plt.plot(solution, 'ro', label=label)
    plt.plot(exact, label='Exact solution')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('u(x,t)', fontsize=14)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.show()


if __name__ == '__main__':
    # Set problem parameters/functions
    k = 3   # diffusion constant
    L = 1.0         # length of spatial domain
    T = 0.5         # total time to solve for
    mx = 10     # number of gridpoints in space
    mt = 1000   # number of gridpoints in time
    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time

    # Boundary examples
    boundary_cond1 = 'homogenous'
    boundary_cond2 = 'dirichlet'
    boundary_cond3 = 'neumann'
    boundary_cond4 = 'periodic'

    # Calculate the exact solution
    exact_solution = np.zeros(mx + 1)
    for j in range(mx+1):
        exact_solution[j] = u_exact(j*L/mx, T, k, L)

    # Plot the solved steady state from the pde
    forward_sol = pde_solver(u_I, L, T, mx, mt, forward_euler, boundary_cond1, p, q, np.array([k]))
    pde_solution_plot(forward_sol[-1], exact_solution, 'Forward Euler')

    # Plot the solved steady state from the alternate pde
    forward_sol_alt = pde_solver(alternate_u_I, L, T, mx, mt, forward_euler, boundary_cond1, p, q, np.array([k]))
    print('The pde solved using the original initial condition: ', forward_sol)
    print('The pde solved using the initial condition as the square of the original: ', forward_sol_alt)

    # Plot the error of each method
    pde_error_plot(u_I, L, T, k, [10, 100], [100, 10000], 30, boundary_cond1, p, q, np.array([k]))

